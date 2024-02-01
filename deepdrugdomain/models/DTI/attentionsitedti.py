"""
Implementation of AttentionSiteDTI for drug-target interaction prediction.

Abstract:
AttentionSiteDTI is an interpretable graph-based deep learning prediction model 
for drug-target interaction (DTI) prediction. Drawing inspiration from NLP sentence 
classification, it treats the drug-target complex as a sentence, understanding the 
relational meaning between protein pockets and the drug molecule. The model utilizes 
protein binding sites combined with a self-attention mechanism, enabling interpretability 
by identifying key protein binding sites in drug-target interactions. AttentionSiteDTI 
has shown improved performance on benchmark datasets and exceptional generalizability 
when tested on new proteins. The model's practical potential was experimentally evaluated, 
showing a high agreement between computational predictions and laboratory observations, 
demonstrating its effectiveness as a pre-screening tool in drug repurposing applications.

Citation:
Mehdi Yazdani-Jahromi, Niloofar Yousefi, Aida Tayebi, Elayaraja Kolanthai, Craig J Neal, Sudipta Seal, Ozlem Ozmen Garibay,
AttentionSiteDTI: an interpretable graph-based model for drug-target interaction prediction using NLP sentence-level relation classification,
Briefings in Bioinformatics, Volume 23, Issue 4, July 2022, bbac272, https://doi.org/10.1093/bib/bbac272

GitHub Repository:
Source code available at: https://github.com/yazdanimehdi/AttentionSiteDTI.

Note:
Proper preprocessing and understanding of input data are essential for the effective 
use of AttentionSiteDTI. The model's input should be formatted correctly to accurately 
predict drug-target interactions.
"""


import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from deepdrugdomain.layers import LayerFactory, ActivationFactory, GraphConvEncoder, LinearHead
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from deepdrugdomain.utils.weight_init import trunc_normal_
from ..factory import ModelFactory
from ..base_model import BaseModel
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from deepdrugdomain.schedulers import BaseScheduler
from deepdrugdomain.data import PreprocessingList, PreprocessingObject


@ModelFactory.register("attentionsitedti")
class AttentionSiteDTI(BaseModel):
    """
        An Interpretable Graph-Based Model for Drug-Target Interaction (DTI) Prediction.

        The AttentionSiteDTI model is a sophisticated deep learning architecture that represents 
        both drugs and targets as graphs. It leverages graph convolutional layers for feature 
        extraction, an attention mechanism to emphasize crucial interaction regions, and optionally, 
        LSTM layers to account for sequential patterns in the combined drug-target representation.

        Attributes:
        ------------
        protein_graph_conv : nn.ModuleList
            List of graph convolutional layers dedicated to processing protein representations.

        ligand_graph_conv : nn.ModuleList
            List of graph convolutional layers for processing ligand representations.

        pool_protein : nn.Module
            Pooling layer for protein graph representations.

        pool_ligand : nn.Module
            Pooling layer for ligand graph representations.

        lstm : nn.Module
            LSTM layers for capturing sequential patterns. Used if `use_lstm_layer` is True.

        attention : nn.Module
            Attention mechanism layer emphasizing regions critical for drug-target interaction.

        fc : nn.ModuleList
            List of fully connected layers leading to the prediction layer.

        fc_out : nn.Linear
            The final prediction layer.

        activation : nn.Module
            Activation function applied after the fully connected layers.

        Parameters:
        ------------
        protein_graph_conv_layer : str
            Specifies the type of graph convolutional layer to be used for processing protein graph representations.

        ligand_graph_conv_layer : str
            Specifies the type of graph convolutional layer to be used for processing ligand (drug) graph representations.

        protein_input_size : int
            Initial dimensionality of the input features for proteins.

        ligand_input_size : int
            Initial dimensionality of the input features for ligands (drugs).

        protein_graph_conv_dims : Sequence[int]
            Dimensions for each subsequent graph convolutional layer dedicated to protein representations.

        ligand_graph_conv_dims : Sequence[int]
            Dimensions for each subsequent graph convolutional layer dedicated to ligand representations.

        sequence_length : int
            Expected length of the combined drug-target sequence representation.

        embedding_dim : int
            Desired dimensionality of the embeddings after combining drug and target representations.

        ligand_graph_pooling : str
            Defines the type of graph pooling mechanism to be used on ligand graphs.

        protein_graph_pooling : str
            Defines the type of graph pooling mechanism to be used on protein graphs.

        use_lstm_layer : bool
            Flag to decide if LSTM layers should be used in the model for capturing sequential patterns.

        use_bilstm : bool
            If set to True, a bidirectional LSTM will be used. Relevant only if `use_lstm_layer` is True.

        lstm_input : Optional[int]
            Size of input features for the LSTM layer.

        lstm_output : Optional[int]
            Output size from the LSTM layer.

        lstm_num_layers : Optional[int]
            Number of LSTM layers to be used.

        lstm_dropout_rate : Optional[float]
            Dropout rate to be applied to LSTM layers.

        head_dims : Sequence[int]
            Defines the dimensions for each subsequent fully connected layer leading to the final prediction.

        attention_layer : str
            Specifies the type of attention layer to be used in the model.

        attention_head : int
            Number of attention heads in the attention mechanism.

        attention_dropout : float
            Dropout rate applied in the attention mechanism.

        qk_scale : Optional[float]
            Scaling factor for the query-key dot product in the attention mechanism.

        proj_drop : float
            Dropout rate applied after the projection in the attention mechanism.

        attention_layer_bias : bool
            If True, biases will be included in the attention mechanism computations.

        protein_conv_dropout_rate : Sequence[float]
            Dropout rates for each protein graph convolutional layer.

        protein_conv_normalization : Sequence[str]
            Normalization types (e.g., 'batch', 'layer') for each protein graph convolutional layer.

        ligand_conv_dropout_rate : Sequence[float]
            Dropout rates for each ligand graph convolutional layer.

        ligand_conv_normalization : Sequence[str]
            Normalization types (e.g., 'batch', 'layer') for each ligand graph convolutional layer.

        head_dropout_rate : float
            Dropout rate applied before the final prediction layer.

        head_activation_fn : Optional[str]
            Activation function applied after the fully connected layers. If None, no activation is applied.

        protein_graph_conv_kwargs : Sequence[dict]
            Additional keyword arguments for each protein graph convolutional layer.

        ligand_graph_conv_kwargs : Sequence[dict]
            Additional keyword arguments for each ligand graph convolutional layer.

        ligand_graph_pooling_kwargs : dict
            Additional keyword arguments for the ligand graph pooling layer.

        protein_graph_pooling_kwargs : dict
            Additional keyword arguments for the protein graph pooling layer.

        **kwargs : 
            Other additional keyword arguments not explicitly listed above.

    """

    def __init__(self,
                 protein_graph_conv_layer: Sequence[str],
                 ligand_graph_conv_layer: Sequence[str],
                 protein_input_size: int,
                 ligand_input_size: int,
                 protein_graph_conv_dims: Sequence[int],
                 ligand_graph_conv_dims: Sequence[int],
                 sequence_length: int,
                 embedding_dim: int,
                 ligand_graph_pooling: str,
                 protein_graph_pooling: str,
                 use_lstm_layer: bool,
                 use_bilstm: bool,
                 lstm_input: Optional[int],
                 lstm_hidden: Optional[int],
                 lstm_num_layers: Optional[int],
                 lstm_dropout_rate: Optional[float],
                 head_dims: Sequence[int],
                 attention_layer: str,
                 attention_head: int,
                 attention_dropout: float,
                 qk_scale: Optional[float],
                 proj_drop: float,
                 attention_layer_bias: bool,
                 protein_conv_dropout_rate: Sequence[float],
                 protein_conv_normalization: Sequence[bool],
                 ligand_conv_dropout_rate: Sequence[float],
                 ligand_conv_normalization: Sequence[bool],
                 head_dropout_rate: float,
                 head_activation_fn: Optional[str],
                 protein_graph_conv_kwargs: Sequence[dict],
                 ligand_graph_conv_kwargs: Sequence[dict],
                 ligand_graph_pooling_kwargs: dict,
                 protein_graph_pooling_kwargs: dict,
                 **kwargs
                 ) -> None:
        """Initialize the AttentionSiteDTI model."""
        super(AttentionSiteDTI, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.head_dims = head_dims
        self.head_dropout_rate = head_dropout_rate
        self.head_activation_fn = head_activation_fn

        # Initialize target encoder layers
        self.target_encoder = GraphConvEncoder(protein_graph_conv_layer, protein_input_size, embedding_dim, protein_graph_conv_dims, protein_graph_pooling,
                                               protein_graph_pooling_kwargs, protein_graph_conv_kwargs, protein_conv_dropout_rate, protein_conv_normalization, **kwargs)

        # Initialize ligand encoder layers
        self.drug_encoder = GraphConvEncoder(ligand_graph_conv_layer, ligand_input_size, embedding_dim, ligand_graph_conv_dims, ligand_graph_pooling,
                                             ligand_graph_pooling_kwargs, ligand_graph_conv_kwargs, ligand_conv_dropout_rate, ligand_conv_normalization, **kwargs)

        # Sequence encoder layers
        self.lstm = self.lstm = nn.LSTM(lstm_input, self.embedding_dim, lstm_hidden,
                                        lstm_num_layers, lstm_dropout_rate, use_bilstm) if use_lstm_layer else None

        assert self.embedding_dim % attention_head == 0, "The embedding dimension must be advisable by number of \
                                                          attention heads"

        # Attention layer
        self.attention = LayerFactory.create(attention_layer, self.embedding_dim, num_heads=attention_head,
                                             qkv_bias=attention_layer_bias, qk_scale=qk_scale,
                                             attn_drop=attention_dropout, proj_drop=proj_drop, **kwargs)

        # Prediction layer
        self.head = LinearHead(self.embedding_dim * self.sequence_length, 1, self.head_dims,
                               self.head_activation_fn, self.head_dropout_rate, normalization=None)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
            Initialize weights for the layers.

            This method applies a truncated normal initialization for linear layers and 
            sets up biases/weights for LayerNorm layers.

            Parameters:
            - m (nn.Module): A PyTorch module whose weights need to be initialized.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_attention_mask(self, sequence_size):
        """
            Generate an attention mask based on sequence size.

            This mask is designed to pay attention to valid parts of the sequence and ignore padding.

            Parameters:
            - sequence_size (int): Size of the valid sequence.

            Returns:
            - Tensor: The generated attention mask.
        """
        mask = torch.eye(self.sequence_length, dtype=torch.uint8).view(1, self.sequence_length,
                                                                       self.sequence_length)
        mask[0, sequence_size:self.sequence_length, :] = 0
        mask[0, :, sequence_size:self.sequence_length] = 0
        mask[0, :, sequence_size - 1] = 1
        mask[0, sequence_size - 1, :] = 1
        mask[0, sequence_size - 1, sequence_size - 1] = 0

        return mask

    def sequence_creator(self, target_rep, drug_rep):
        sequence = torch.cat((drug_rep, target_rep), dim=0).view(
            1, -1, self.embedding_dim)

        mask = self.generate_attention_mask(
            sequence.size()[1]).to(sequence.device)

        sequence = F.pad(input=sequence, pad=(
            0, 0, 0, self.sequence_length - sequence.size()[1]), mode='constant', value=0)

        return sequence, mask

    def forward(self, drug, target):
        """
            Forward pass of the model.

            Process the drug and target graphs through the network, making use of the attention mechanism, 
            and optionally, LSTM layers, to predict their interaction.

            Parameters:
            - g (tuple): A tuple containing the drug and target graphs.

            Returns:
            - Tuple[Tensor, Tensor]: Model output and attention weights.
        """

        protein_rep = self.target_encoder(target)
        ligand_rep = self.drug_encoder(drug)

        sequence, mask = self.sequence_creator(protein_rep, ligand_rep)

        if self.lstm is not None:
            sequence.permute(1, 0, 2)
            output, _ = self.lstm(sequence)
            output = output.permute(1, 0, 2)

        else:
            output = sequence

        out, att = self.attention(output, mask=mask, return_attn=True)

        out = out.view(-1, out.size()[1] * out.size()[2])

        return self.head(out), att

    def predict(self, drug: List[Any], target: List[Any]) -> Any:
        """
            Make predictions using the model.

            Parameters:
            - g (tuple): A tuple containing the drug and target graphs.

            Returns:
            - Tensor: The predictions.
        """
        return self.forward(drug, target)

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
            for batch_idx, (drug, protein, target) in enumerate(t):
                last_batch = batch_idx == last_batch_idx
                need_update = last_batch or (batch_idx + 1) % accum_steps == 0
                update_idx = batch_idx // accum_steps

                if batch_idx >= last_batch_idx_to_accum:
                    accum_steps = last_accum_steps

                outs = []
                for item in range(len(drug)):
                    d = drug[item].to(device)
                    p = protein[item].to(device)
                    out = self.forward(d, p)

                    if isinstance(out, tuple):
                        out = out[0]

                    outs.append(out)

                out = torch.stack(outs, dim=0).squeeze(1)
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
        """
            Evaluate the model on the given dataset.
        """
        losses = []
        predictions = []
        targets = []
        self.eval()
        with tqdm(dataloader) as t:
            t.set_description('Testing')
            for batch_idx, (drug, protein, target) in enumerate(t):
                outs = []
                with torch.no_grad():
                    for item in range(len(drug)):
                        d = drug[item].to(device)
                        p = protein[item].to(device)
                        out = self.forward(d, p)
                        if isinstance(out, tuple):
                            out = out[0]

                        outs.append(out)

                out = torch.stack(outs, dim=0).squeeze(1)
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

    def collate(self, batch: List[Tuple[Any, Any, torch.Tensor]]) -> Tuple[Tuple[List[Any], List[Any]], torch.Tensor]:
        """
            Collate function for the AttentionSiteDTI model.
        """
        # Unpacking the batch data
        drug, protein, targets = zip(*batch)
        targets = torch.stack(targets, 0)

        return drug, protein, targets

    def reset_head(self) -> None:
        self.head = LinearHead(self.embedding_dim * self.sequence_length, 1, self.head_dims,
                               self.head_activation_fn, self.head_dropout_rate, normalization=None)

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)

    def default_preprocess(self, smile_attr, pdb_id_attr, label_attr) -> List[PreprocessingObject]:
        feat = CanonicalAtomFeaturizer()
        preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="graph", preprocessing_settings={
            "fragment": False, "max_block": 6, "max_sr": 8, "min_frag_atom": 1, "node_featurizer": feat}, in_memory=True, online=False)
        preprocess_protein = PreprocessingObject(attribute=pdb_id_attr, from_dtype="pdb_id", to_dtype="binding_pocket_graph", preprocessing_settings={
            "pdb_path": "data/pdb/", "protein_size_limit": 10000}, in_memory=False, online=False)
        preprocess_label = PreprocessingObject(
            attribute=label_attr, from_dtype="binary", to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)

        return [preprocess_drug, preprocess_protein, preprocess_label]
