from functools import partial
from typing import Any, Optional, Sequence

from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory

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


@ModelFactory.register('fragxsitedti')
class FragXSiteDTI(BaseModel):
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

        # Initialize protein graph convolution layers
        p_dims = [protein_input_size] + list(protein_graph_conv_dims)
        assert len(p_dims) - 1 == len(protein_conv_dropout_rate) == len(protein_graph_conv_kwargs) == len(
            protein_conv_normalization), "The number of protein graph convolution layers parameters must be the same"
        self.protein_graph_conv = nn.ModuleList([
            LayerFactory.create(protein_graph_conv_layer,
                                p_dims[i],
                                p_dims[i + 1],
                                normalization=protein_conv_normalization[i],
                                dropout=protein_conv_dropout_rate[i],
                                **protein_graph_conv_kwargs[i]) for i in range(len(p_dims) - 1)])

        # Initialize drug graph convolution layers
        l_dims = [ligand_input_size] + list(ligand_graph_conv_dims)
        assert len(l_dims) - 1 == len(ligand_conv_dropout_rate) == len(ligand_graph_conv_kwargs) == len(
            ligand_conv_normalization), "The number of ligand graph convolution layers parameters must be the same"
        self.ligand_graph_conv = nn.ModuleList([
            LayerFactory.create(ligand_graph_conv_layer,
                                l_dims[i],
                                l_dims[i + 1],
                                normalization=ligand_conv_normalization[i],
                                dropout=ligand_conv_dropout_rate[i],
                                **ligand_graph_conv_kwargs[i]) for i in range(len(l_dims) - 1)])

        # Graph pooling layers
        self.pool_ligand = LayerFactory.create(
            ligand_graph_pooling, **ligand_graph_pooling_kwargs)
        self.pool_protein = LayerFactory.create(
            protein_graph_pooling, **protein_graph_pooling_kwargs)

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

        self.norm = LayerFactory.create(
            head_normalization, embedding_dim) if head_normalization is not None else nn.Identity()
        self.feature_info = [
            dict(num_chs=embedding_dim, reduction=0, module='head')]
        # Prediction layer
        self.head = nn.ModuleList()
        neuron_list = [self.embedding_dim] + list(head_dims)
        for item in range(len(neuron_list) - 1):
            self.head.append(nn.Dropout(head_dropout_rate))
            self.head.append(
                nn.Linear(neuron_list[item], neuron_list[item + 1]))
            self.head.append(ActivationFactory.create(
                head_activation_fn) if head_activation_fn else nn.Identity())

        trunc_normal_(self.latent_query, std=.02)

        self.apply(self._init_weights)

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

    def forward(self, drug, target):

        for module in self.protein_graph_conv:
            target = module(target)

        for module in self.ligand_graph_conv:
            drug = module(drug)

        protein_rep = self.pool_protein(target).view(1, -1, self.embedding_dim)
        ligand_rep = self.pool_ligand(drug).view(1, -1, self.embedding_dim)
        x = self.latent_query.expand(1, -1, -1)
        attn_binding = []
        attn_frag = []
        for i, blk in enumerate(self.blocks_ca_input):
            x,  attn = blk(x, protein_rep)
            attn_binding.append(attn)

        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)

        for i, blk in enumerate(self.blocks_ca_output):
            x, attn = blk(x, ligand_rep)
            attn_frag.append(attn)

        x = self.norm(x)
        x = torch.mean(x, dim=1)
        return torch.sigmoid(self.head(x)), attn_binding, attn_frag

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

        losses = []
        predictions = []
        targets = []
        self.eval()
        with tqdm(dataloader) as t:
            t.set_description('Testing')
            for batch_idx, (drug, protein, target) in enumerate(t):
                outs = []
                for item in range(len(drug)):
                    d = drug[item].to(device)
                    p = protein[item].to(device)
                    out = self.forward(d, p)
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

        metrics = {"val" + str(key): val for key, val in metrics.items()}

        if logger is not None:
            logger.log(metrics)

    def reset_head(self) -> None:
        pass

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)
