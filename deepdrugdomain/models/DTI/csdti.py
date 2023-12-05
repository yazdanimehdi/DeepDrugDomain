"""
Implementation of CSDTI for drug-target interaction prediction.

Abstract:
CSDTI addresses the challenges in drug-target interaction (DTI) prediction by 
introducing an interpretable network architecture that combines graph neural 
networks (GNNs) for drug molecule modeling with a cross-attention mechanism 
to capture drug-target interaction features. Traditional GNNs, which focus 
on local neighboring nodes, often miss the global 3D structure and edge information 
of drug molecules. CSDTI overcomes this by integrating a drug molecule aggregator 
to capture high-order dependencies within drug molecular graphs. This allows for 
effective modeling of DTIs, improving performance metrics such as AUC, precision, 
and recall. The model's interpretability is enhanced by visualizing attention weights, 
providing chemical insights into the interactions. CSDTI demonstrates superior 
performance over state-of-the-art methods in DTI prediction tasks.

Citation:
Pan, Y., Zhang, Y., Zhang, J., et al. (2023). CSDTI: an interpretable cross-attention 
network with GNN-based drug molecule aggregation for drug-target interaction prediction. 
Applied Intelligence, 53, 27177-27190. https://doi.org/10.1007/s10489-023-04977-8

GitHub Repository:
Source code available at: https://github.com/ziduzidu/CSDTI.

Note:
Effective use of CSDTI requires careful preprocessing of drug and target data. 
Users should ensure that data is formatted appropriately for accurate interaction 
predictions and analysis.
"""

from typing import Any, List, Sequence, Tuple
from torch import Tensor, nn
import torch
import math
from deepdrugdomain.layers import LinearHead, GraphConvEncoder
from deepdrugdomain.layers.utils.layer_factory import LayerFactory
from deepdrugdomain.layers.utils import ActivationFactory
import torch.nn.functional as F
from ..factory import ModelFactory
from deepdrugdomain.layers import get_node_attr, change_node_attr
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from deepdrugdomain.schedulers import BaseScheduler
from typing import Any, Callable, List, Optional, Sequence, Type
from tqdm import tqdm
import numpy as np
from ..base_model import BaseModel
from deepdrugdomain.utils import batch_graphs
from deepdrugdomain.data import PreprocessingObject
from dgllife.utils import CanonicalAtomFeaturizer

@LayerFactory.register("conv_sequence_encoder")
class ConvolutionalSequenceEncoder(nn.Module):
    # todo: make more generic
    def __init__(self, input_dim, hidden_dims, conv_kernel_sizes, dropout_rate, scaling_factor):
        """
        Convolutional Sequence Encoder for encoding protein sequences.

        Parameters:
            input_dim (int): Dimension of the input protein sequence.
            hidden_dim (int): Dimension of the hidden layers in the network.
            num_layers (int): Number of convolutional layers.
            conv_kernel_size (int): Size of the convolutional kernel.
            dropout_rate (float): Dropout rate for regularization.
            groupnorm_groups (int, optional): Number of groups for Group Normalization. Defaults to 8.
        """
        super(ConvolutionalSequenceEncoder, self).__init__()

        assert all([i % 2 == 1 for i in conv_kernel_sizes]
                   ), "Convolution kernel size must be odd."
        # assert len(hidden_dims) == len(conv_kernel_sizes) + \
        #     1, "Number of hidden dimensions must be equal to number of convolutional layers + 1."

        self.scaling_factor = scaling_factor

        self.input_dimension = input_dim
        self.hidden_dimension = hidden_dims
        self.convolution_kernel_size = conv_kernel_sizes

        self.convolution_layers = nn.ModuleList(
            [nn.Conv1d(hidden_dims[i], 2 * hidden_dims[i + 1], conv_kernel_sizes[i], padding=(conv_kernel_sizes[i] - 1) // 2)
             for i in range(len(hidden_dims) - 1)]
        )

        self.dropout_layer = nn.Dropout(dropout_rate)
        self.input_to_hidden_fc = nn.Linear(input_dim, hidden_dims[0])
        self.layer_normalization = nn.LayerNorm(hidden_dims[-1])

    def forward(self, protein_sequence):
        """
        Forward pass of the model.

        Parameters:
            protein_sequence (Tensor): Input protein sequence.

        Returns:
            Tensor: Encoded protein sequence.
        """
        scaling_factor = torch.sqrt(torch.FloatTensor(
            [self.scaling_factor])).to(protein_sequence.device)

        hidden_representation = self.input_to_hidden_fc(protein_sequence)
        hidden_representation = hidden_representation.permute(0, 2, 1)

        for convolution_layer in self.convolution_layers:
            convolution_output = convolution_layer(
                self.dropout_layer(hidden_representation))
            convolution_output = F.glu(convolution_output, dim=1)
            convolution_output += hidden_representation * scaling_factor
            hidden_representation = convolution_output

        normalized_output = convolution_output.permute(0, 2, 1)
        normalized_output = self.layer_normalization(normalized_output)

        return normalized_output


@ModelFactory.register("csdti")
class CSDTI(BaseModel):
    def __init__(self,
                 num_features_xd: int,
                 num_features_xt: int,
                 graph_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 ligand_graph_conv_11_layer: Sequence[str],
                 ligand_graph_conv_11_layer_args: Sequence[dict],
                 ligand_graph_conv_11_layer_dims: Sequence[int],
                 ligand_graph_conv_11_dropout_rate: Sequence[float],
                 ligand_graph_conv_11_normalization: Sequence[str],

                 ligand_graph_conv_12_layer: Sequence[str],
                 ligand_graph_conv_12_layer_args: Sequence[dict],
                 ligand_graph_conv_12_layer_dims: Sequence[int],
                 ligand_graph_conv_12_dropout_rate: Sequence[float],
                 ligand_graph_conv_12_normalization: Sequence[str],

                 ligand_graph_conv_21_layer: Sequence[str],
                 ligand_graph_conv_21_layer_args: Sequence[dict],
                 ligand_graph_conv_21_layer_dims: Sequence[int],
                 ligand_graph_conv_21_dropout_rate: Sequence[float],
                 ligand_graph_conv_21_normalization: Sequence[str],

                 ligand_graph_conv_22_layer: Sequence[str],
                 ligand_graph_conv_22_layer_args: Sequence[dict],
                 ligand_graph_conv_22_layer_dims: Sequence[int],
                 ligand_graph_conv_22_dropout_rate: Sequence[float],
                 ligand_graph_conv_22_normalization: Sequence[str],

                 ligand_pooling_layer_type: str,
                 ligand_pooling_layer_kwargs: dict,

                 linear_activations: str,
                 linear_dropout: float,
                 protein_sequence_length: int,
                 protein_conv_out_channels: int,
                 protein_conv_kernel_size: int,


                 protein_encoder_block: str,
                 protein_encoder_block_args: dict,

                 attention_interaction_layer: str,
                 attention_interaction_layer_args: dict,

                 head_output_dim: int,
                 head_dims: Sequence[int],
                 head_dropout: Sequence[float],
                 head_activation: Sequence[str],
                 head_normalization: Sequence[str],
                 ):
        super(CSDTI, self).__init__()
        self.protein_conv_out_channels = protein_conv_out_channels
        self.embedding_dim = embedding_dim
        self.head_output_dim = head_output_dim
        self.head_dims = head_dims
        self.head_dropout = head_dropout
        self.head_activation = head_activation
        self.head_normalization = head_normalization
        self.output_dim = output_dim
        self.graph_dim = graph_dim
        # Initialize ligand encoder layers
        self.drug_encoder11 = GraphConvEncoder(ligand_graph_conv_11_layer, num_features_xd, graph_dim, ligand_graph_conv_11_layer_dims, None, {
        }, ligand_graph_conv_11_layer_args, ligand_graph_conv_11_dropout_rate, ligand_graph_conv_11_normalization)

        self.drug_encoder12 = GraphConvEncoder(ligand_graph_conv_12_layer, num_features_xd, graph_dim, ligand_graph_conv_12_layer_dims, None, {
        }, ligand_graph_conv_12_layer_args, ligand_graph_conv_12_dropout_rate, ligand_graph_conv_12_normalization)

        self.drug_encoder21 = GraphConvEncoder(ligand_graph_conv_21_layer, graph_dim * 2, graph_dim, ligand_graph_conv_21_layer_dims, ligand_pooling_layer_type,
                                               ligand_pooling_layer_kwargs, ligand_graph_conv_21_layer_args, ligand_graph_conv_21_dropout_rate, ligand_graph_conv_21_normalization)

        self.drug_encoder22 = GraphConvEncoder(ligand_graph_conv_22_layer, graph_dim * 2, graph_dim, ligand_graph_conv_22_layer_dims, ligand_pooling_layer_type,
                                               ligand_pooling_layer_kwargs, ligand_graph_conv_22_layer_args, ligand_graph_conv_22_dropout_rate, ligand_graph_conv_22_normalization)

        self.fc1_xd = nn.Linear(graph_dim * 2, output_dim)

        self.activation = ActivationFactory.create(linear_activations)
        self.linear_dropout = linear_dropout

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embedding_dim)

        self.conv_xt_1 = nn.Conv1d(in_channels=protein_sequence_length,
                                   out_channels=protein_conv_out_channels, kernel_size=protein_conv_kernel_size)

        self.protein_encoder = LayerFactory.create(
            protein_encoder_block, embedding_dim, **protein_encoder_block_args)

        self.fc1_xt = nn.Linear(
            protein_conv_out_channels * embedding_dim, output_dim)

        # cross attention
        self.att = LayerFactory.create(
            attention_interaction_layer, output_dim, **attention_interaction_layer_args)

        # output
        self.head = LinearHead(self.output_dim * 3, self.head_output_dim, self.head_dims,
                               self.head_activation, self.head_dropout, self.head_normalization)

    def forward(self, ligand_g1, ligand_g2, target):
        # drug
        ligand_g1 = self.drug_encoder11(ligand_g1)
        ligand_g2 = self.drug_encoder12(ligand_g2)

        x1 = get_node_attr(ligand_g1)
        x2 = get_node_attr(ligand_g2)

        x12 = torch.cat((x1, x2), dim=1)

        ligand_g1 = change_node_attr(ligand_g1, x12)
        ligand_g2 = change_node_attr(ligand_g2, x12)

        x1 = self.drug_encoder21(ligand_g1)
        x2 = self.drug_encoder22(ligand_g2)

        x = torch.cat((x1, x2), dim=1)
        x = self.activation(self.fc1_xd(x))
        x = F.dropout(x, p=self.linear_dropout)
        x = x.unsqueeze(1)
        # protein
        embedded_xt = self.embedding_xt(target)
        xt = self.conv_xt_1(embedded_xt)
        xt = self.protein_encoder(xt)
        xt = xt.view(-1, self.protein_conv_out_channels * self.embedding_dim)
        xt = self.fc1_xt(xt).unsqueeze(1)

        # cross attention
        att = self.att(x, xt)

        # mix
        feature = torch.cat((x, att, xt), dim=1).view(-1, self.output_dim * 3)
        # output
        out = self.head(feature)

        return out

    def collate(self, batch: List[Tuple[Any, Any, torch.Tensor]]) -> Tuple[Tuple[List[Any], List[Any]], torch.Tensor]:
        """
            Collate function for the AMMVF model.
        """
        # Unpacking the batch data
        ligand_graph1, ligand_graph2, protein, targets = zip(*batch)
        targets = torch.stack(targets, 0)
        ligand_graph1 = batch_graphs(ligand_graph1)
        ligand_graph2 = batch_graphs(ligand_graph2)
        protein = torch.stack(protein, 0)

        return ligand_graph1, ligand_graph2, protein, targets

    def predict(self, *args, **kwargs) -> Any:
        """
            Make predictions using the model.
        """
        return self.forward(*args, **kwargs)

    def train_one_epoch(self, dataloader: DataLoader, device: torch.device, criterion: Callable, optimizer: Optimizer, num_epochs: int, scheduler: Optional[Type[BaseScheduler]] = None, evaluator: Optional[Type[Evaluator]] = None, grad_accum_steps: int = 1, clip_grad: Optional[str] = None, logger: Optional[Any] = None) -> Any:
        """
            Train the model for one epoch.
        """
        accum_steps = grad_accum_steps
        last_accum_steps = len(dataloader) % accum_steps
        updates_per_epoch = (len(dataloader) + accum_steps - 1) // accum_steps
        num_updates = num_epochs * updates_per_epoch
        last_batch_idx = len(dataloader) - 1
        last_batch_idx_to_accum = len(dataloader) - last_accum_steps

        losses = []
        predictions = []
        targets = []
        self.train()
        with tqdm(dataloader) as t:
            t.set_description('Training')
            for batch_idx, (drug_graph1, drug_graph2, protein, target) in enumerate(t):
                last_batch = batch_idx == last_batch_idx
                need_update = last_batch or (batch_idx + 1) % accum_steps == 0
                update_idx = batch_idx // accum_steps

                if batch_idx >= last_batch_idx_to_accum:
                    accum_steps = last_accum_steps
                drug_graph1 = drug_graph1.to(device)
                drug_graph2 = drug_graph2.to(device)
                protein = protein.to(device)
                out = self.forward(drug_graph1, drug_graph2, protein)

                target = target.to(
                    device).view(-1, 1).to(torch.float)

                loss = criterion(out, target)
                loss /= accum_steps

                loss.backward()
                losses.append(loss.detach().cpu().item())
                predictions.append(out.detach().cpu())
                targets.append(target.detach().cpu())
                metrics = evaluator(predictions, targets) if evaluator else {}

                metrics["loss"] = np.mean(losses) * accum_steps
                lrl = [param_group['lr']
                       for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                metrics["lr"] = lr

                t.set_postfix(**metrics)

                if logger is not None:
                    logger.log(metrics)

                optimizer.step()

                num_updates += 1
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step_update(
                        num_updates=num_updates, metric=metrics["loss"])

    def evaluate(self, dataloader: DataLoader, device: torch.device, criterion: Callable, evaluator: Optional[Type[Evaluator]] = None, logger: Optional[Any] = None) -> Any:

        losses = []
        predictions = []
        targets = []
        self.eval()
        with tqdm(dataloader) as t:
            t.set_description('Testing')
            for batch_idx, (drug_graph1, drug_graph2, protein, target) in enumerate(t):
                drug_graph1 = drug_graph1.to(device)
                drug_graph2 = drug_graph2.to(device)
                protein = protein.to(device)

                with torch.no_grad():
                    out = self.forward(drug_graph1, drug_graph2, protein)

                target = target.to(
                    device).view(-1, 1).to(torch.float)

                loss = criterion(out, target)
                losses.append(loss.detach().cpu().item())
                predictions.append(out.detach().cpu())
                targets.append(target.detach().cpu())
                metrics = evaluator(predictions, targets) if evaluator else {}
                metrics["loss"] = np.mean(losses)
                t.set_postfix(**metrics)

        metrics = evaluator(predictions, targets) if evaluator else {}
        metrics["loss"] = np.mean(losses)

        metrics = {"val_" + str(key): val for key, val in metrics.items()}

        if logger is not None:
            logger.log(metrics)
        return metrics

    def reset_head(self) -> None:
        self.head = LinearHead(self.output_dim * 3, self.head_output_dim, self.head_dims,
                               self.head_activation, self.head_dropout, self.head_normalization)

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)

    def default_preprocess(self, smile_attr, target_seq_attr, label_attr) -> List[PreprocessingObject]:
        feat = CanonicalAtomFeaturizer()
        preprocess_drug1 = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="graph", preprocessing_settings={
            "fragment": False, "node_featurizer":  feat, "consider_hydrogen": False, "consider_hydrogen": True}, in_memory=True, online=False)

        preprocess_drug2 = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="graph", preprocessing_settings={
            "fragment": False, "node_featurizer":  feat, "consider_hydrogen": False, "hops": 2, "consider_hydrogen": True}, in_memory=True, online=False)

        preprocess_protein = PreprocessingObject(
            attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="kmers_encoded_tensor", preprocessing_settings={"ngram": 1, "max_length": 1200}, in_memory=True, online=False)

        preprocess_label = PreprocessingObject(
            attribute=label_attr,  from_dtype="binary", to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)

        return [preprocess_drug1, preprocess_drug2, preprocess_protein, preprocess_label]