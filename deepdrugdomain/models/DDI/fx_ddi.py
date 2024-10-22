"""
Implementation of FragXsiteDTI for drug-target interaction prediction.

Abstract:
FragXsiteDTI is a novel transformer-based model aimed at improving the interpretability 
and performance of drug-target interaction (DTI) prediction. It is the first DTI model 
to use both drug molecule fragments and protein pockets, providing rich representations 
for detailed interaction analysis. Inspired by the Perceiver IO framework, FragXsiteDTI 
employs a learnable latent array for initial cross-attention with protein binding site 
embeddings and subsequent self-attention for drug fragments. This approach ensures 
seamless information translation and captures critical nuances in drug-protein interactions. 
The model demonstrates superior predictive power on benchmark datasets and offers 
interpretability by identifying critical components in drug-target pairs.

GitHub Repository:
Source code available at: https://github.com/yazdanimehdi/FragXsiteDTI.

Citation:
Yalabadi, A. K., Yazdani-Jahromi, M., Yousefi, N., Tayebi, A., Abdidizaji, S., & Garibay, O. O. 
(2023). FragXsiteDTI: Revealing Responsible Segments in Drug-Target Interaction with 
Transformer-Driven Interpretation. arXiv preprint arXiv:2311.02326.

Note:
For optimal use of FragXsiteDTI, ensure accurate preprocessing of drug molecules and 
protein pocket data. The model requires precise input formats for effective 
interpretation and prediction of drug-target interactions.

[Implementation details and methods go here.]
"""
import time
from dgl.nn.pytorch.glob import MaxPooling
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple
from deepdrugdomain.layers.modules.graph_encoders.graph_conv import GraphConvEncoder
from deepdrugdomain.layers.modules.heads.linear import LinearHead
from torch.nn.utils.rnn import pad_sequence
from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory
import dgl
from ..factory import ModelFactory
from ..base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from deepdrugdomain.utils.weight_init import trunc_normal_
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from deepdrugdomain.schedulers import BaseScheduler
from typing import Any, Callable, List, Optional, Sequence, Type
from tqdm import tqdm
import numpy as np
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from deepdrugdomain.data import PreprocessingList, PreprocessingObject

@ModelFactory.register('fx_ddi')
class FragXSiteDDI(BaseModel):
    def __init__(self,
                 protein_graph_conv_layer: str,
                 ligand_graph_conv_layer: str,
                 protein_input_size: int,
                 ligand_input_size: int,
                 protein_graph_conv_dims: Sequence[int],
                 ligand_graph_conv_dims: Sequence[int],
                 protein_conv_dropout_rate: Sequence[float],
                 protein_conv_normalization: Sequence[str],
                 ligand_conv_dropout_rate: Sequence[float],
                 ligand_conv_normalization: Sequence[str],
                 head_dropout_rate: float,
                 head_activation_fn: Optional[str],
                 head_normalization: Optional[str],
                 protein_graph_conv_kwargs: Sequence[dict],
                 ligand_graph_conv_kwargs: Sequence[dict],
                 ligand_graph_pooling_kwargs: dict,
                 protein_graph_pooling_kwargs: dict,
                 embedding_dim: int,
                 ligand_graph_pooling: str,
                 protein_graph_pooling: str,
                 self_attention_depth: str,
                 self_attention_num_heads: str,
                 self_attention_mlp_ratio: str,
                 self_attention_qkv_bias: bool,
                 self_attention_qk_scale: Optional[float],
                 self_attention_drop_rate: float,
                 self_attn_drop_rate: float,
                 self_drop_path_rate: float,
                 self_norm_layer: str,
                 input_norm_layer: str,
                 output_norm_layer: str,
                 block_layers: str,
                 input_block_layers: str,
                 output_block_layers: str,
                 self_act_layer: str,
                 input_act_layer: str,
                 output_act_layer: str,
                 attention_block: str,
                 self_mlp_block: str,
                 input_mlp_block: str,
                 output_mlp_block: str,
                 input_cross_att_block: str,
                 output_cross_att_block: str,
                 input_cross_attention_num_heads: int,
                 input_cross_attention_mlp_ratio: float,
                 input_cross_attention_qkv_bias: bool,
                 input_cross_attention_qk_scale: Optional[float],
                 input_cross_attention_drop_rate: float,
                 input_cross_attn_drop_rate: float,
                 input_cross_drop_path_rate: float,
                 output_cross_attention_num_heads: int,
                 output_cross_attention_mlp_ratio: float,
                 output_cross_attention_qkv_bias: bool,
                 output_cross_attention_qk_scale: Optional[float],
                 output_cross_attention_drop_rate: float,
                 output_cross_attn_drop_rate: float,
                 output_cross_drop_path_rate: float,
                 input_stages: int,
                 output_stages: int,
                 latent_space: int,
                 head_dims: Sequence[int]):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initialize ligand encoder layers
        self.drug_encoder = GraphConvEncoder(ligand_graph_conv_layer, ligand_input_size, embedding_dim, ligand_graph_conv_dims, ligand_graph_pooling,
                                             ligand_graph_pooling_kwargs, ligand_graph_conv_kwargs, ligand_conv_dropout_rate, ligand_conv_normalization)

        self.latent_query = nn.Parameter(
            torch.zeros(1, latent_space, embedding_dim))
        dpr = [self_drop_path_rate for i in range(self_attention_depth)]
        self.blocks = nn.ModuleList([
            LayerFactory.create(block_layers,
                                dim=embedding_dim, num_heads=self_attention_num_heads, mlp_ratio=self_attention_mlp_ratio, qkv_bias=self_attention_qkv_bias, qk_scale=self_attention_qk_scale,
                                drop=self_attention_drop_rate, attn_drop=self_attn_drop_rate, drop_path=dpr[
                                    i], norm_layer=self_norm_layer,
                                act_layer=self_act_layer, Attention_block=attention_block, Mlp_block=self_mlp_block)
            for i in range(self_attention_depth)])

        dpr = [input_cross_drop_path_rate for i in range(input_stages)]
        self.blocks_ca_input = nn.ModuleList([
            LayerFactory.create(input_block_layers,
                                dim=embedding_dim, num_heads=input_cross_attention_num_heads, mlp_ratio=input_cross_attention_mlp_ratio, qkv_bias=input_cross_attention_qkv_bias, qk_scale=input_cross_attention_qk_scale,
                                drop=input_cross_attention_drop_rate, attn_drop=input_cross_attn_drop_rate, drop_path=dpr[
                                    i], norm_layer=input_norm_layer,
                                act_layer=input_act_layer, Attention_block=input_cross_att_block, Mlp_block=input_mlp_block) for i in
            range(input_stages)
        ])

        dpr = [output_cross_drop_path_rate for i in range(output_stages)]
        self.blocks_ca_output = nn.ModuleList([
            LayerFactory.create(output_block_layers, dim=embedding_dim, num_heads=output_cross_attention_num_heads, mlp_ratio=output_cross_attention_mlp_ratio, qkv_bias=output_cross_attention_qkv_bias, qk_scale=output_cross_attention_qk_scale,
                                drop=output_cross_attention_drop_rate, attn_drop=output_cross_attn_drop_rate, drop_path=dpr[
                                    i], norm_layer=output_norm_layer,
                                act_layer=output_act_layer, Attention_block=output_cross_att_block, Mlp_block=output_mlp_block) for i in
            range(output_stages)
        ])

        self.head = LinearHead(embedding_dim, 87, head_dims,
                               head_activation_fn, head_dropout_rate, head_normalization)

        trunc_normal_(self.latent_query, std=.02)

        self.apply(self._init_weights)
        self.pooling = MaxPooling()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'latent_query'}

    def get_classifier(self):
        return self.head

    def prepare_and_pool(self, graph, batch_number):
        pooled = self.pooling(graph, graph.ndata['h'])
        graph_pooled = pad_sequence(torch.split(pooled, batch_number), batch_first=True, padding_value=0)

        return graph_pooled
    
    def forward(self, drug, target, drug_batch_number, protein_batch_number):
        protein_graph = self.drug_encoder(target)
        ligand_graph = self.drug_encoder(drug)

        ligand_rep = self.prepare_and_pool(ligand_graph, drug_batch_number)
        protein_rep = self.prepare_and_pool(protein_graph, protein_batch_number)
        x = self.latent_query.expand(protein_rep.size(0), -1, -1)

        attn_binding = []
        attn_frag = []

        for i, blk in enumerate(self.blocks_ca_input):
            x,  attn = blk(x, protein_rep, return_attn=True)
            attn_binding.append(attn)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        for i, blk in enumerate(self.blocks_ca_output):
            x, attn = blk(x, ligand_rep, return_attn=True)
            attn_frag.append(attn)

        x = torch.mean(x, dim=1)
        x = torch.dropout(x, p=0.4, train=self.training)
        return self.head(x), attn_binding, attn_frag
    
    def predict(self, drug: List[Any], target: List[Any]) -> Any:
        """
            Make predictions using the model.

            Parameters:
            - g (tuple): A tuple containing the drug and target graphs.

            Returns:
            - Tensor: The predictions.
        """
        return self.forward(drug, target)[0]

    def train_one_epoch(self, dataloader: DataLoader, device: torch.device, criterion: Callable, optimizer: Optimizer, num_epochs: int, scheduler: Optional[Type[BaseScheduler]] = None, evaluator: Optional[Type[Evaluator]] = None, grad_accum_steps: int = 1, clip_grad: Optional[str] = None, logger: Optional[Any] = None) -> Any:

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
            for batch_idx, x in enumerate(t):
                
                last_batch = batch_idx == last_batch_idx
                need_update = last_batch or (batch_idx + 1) % accum_steps == 0
                update_idx = batch_idx // accum_steps

                if batch_idx >= last_batch_idx_to_accum:
                    accum_steps = last_accum_steps

                out = self.forward(x[0].to(device), x[1].to(device) , x[2], x[3])[0]   
                # out = torch.stack(outs, dim=0).squeeze(1)
                target = x[4].to(
                    device).view(-1).to(torch.int64)

                loss = criterion(out, target)
                loss /= accum_steps

                loss.backward()
                losses.append(loss.detach().cpu().item())
                predictions.append(torch.argmax(out.detach().cpu(), dim=1))

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
        return metrics

    def evaluate(self, dataloader: DataLoader, device: torch.device, criterion: Callable, evaluator: Optional[Type[Evaluator]] = None, logger: Optional[Any] = None) -> Any:

        losses = []
        predictions = []
        targets = []
        self.eval()
        with tqdm(dataloader) as t:
            t.set_description('Testing')
            for batch_idx, x in enumerate(t):
                with torch.no_grad():
                    out = self.forward(x[0].to(device), x[1].to(device) , x[2], x[3])[0]    

                target = x[4].to(
                    device).view(-1).to(torch.int64)

                loss = criterion(out, target)
                losses.append(loss.detach().cpu().item())
                predictions.append(torch.argmax(out.detach().cpu(), dim=1))
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
            Collate function for the FragXsiteDTI model.
        """
        # Unpacking the batch data
        drug, protein, targets = zip(*batch)
        drug_count = [len(d) for d in drug]
        drug = [dgl.batch(d) for d in drug]
        
        drug = dgl.batch(drug)
        protein_count = [len(p) for p in protein]
        protein = [dgl.batch(p) for p in protein]
        protein = dgl.batch(protein)
        targets = torch.stack(targets, 0)

        return drug, protein, drug_count, protein_count, targets
    
    def reset_head(self) -> None:
        pass

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)

    def default_preprocess(self, smile_attr1, smile_attr2, label_attr) -> List[PreprocessingObject]:
        feat = CanonicalAtomFeaturizer()
        preprocess_drug = PreprocessingObject(attribute=smile_attr1, from_dtype="smile", to_dtype="graph", preprocessing_settings={
                                                   "fragment": True, "max_block": 6, "max_sr": 8, "min_frag_atom": 1, "node_featurizer": feat}, in_memory=True, online=False)
        preprocess_protein = PreprocessingObject(attribute=smile_attr2, from_dtype="smile", to_dtype="graph", preprocessing_settings={
                                                   "fragment": True, "max_block": 6, "max_sr": 8, "min_frag_atom": 1, "node_featurizer": feat}, in_memory=True, online=False)
        preprocess_label = PreprocessingObject(
            attribute=label_attr, from_dtype="binary", to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)
        
        return [preprocess_drug, preprocess_protein, preprocess_label]