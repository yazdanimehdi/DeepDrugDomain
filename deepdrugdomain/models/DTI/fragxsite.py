from functools import partial
from typing import Optional, Sequence

from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory

from ..factory import ModelFactory
import torch
from torch import nn
import torch.nn.functional as F
from deepdrugdomain.utils.weight_init import trunc_normal_


@ModelFactory.register('fragxsite')
class FragXSiteDTI(nn.Module):
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
                 self_attention_depth=2,
                 self_attention_num_heads=4,
                 self_attention_mlp_ratio=4.,
                 self_attention_qkv_bias=True,
                 self_attention_qk_scale=None,
                 self_attention_drop_rate=0.3,
                 self_attn_drop_rate=0.,
                 self_drop_path_rate=0.3,
                 self_norm_layer="layer_norm",
                 input_norm_layer="layer_norm",
                 output_norm_layer="layer_norm",
                 block_layers="transformer_block",
                 input_block_layers="transformer_cross_attention_block",
                 output_block_layers="transformer_cross_attention_block",
                 self_act_layer="gelu",
                 input_act_layer="gelu",
                 output_act_layer="gelu",
                 attention_block="transformer_attention",
                 self_mlp_block="transformer_mlp",
                 input_mlp_block="transformer_mlp",
                 output_mlp_block="transformer_mlp",
                 input_cross_att_block="transformer_cross_attention",
                 output_cross_att_block="transformer_cross_attention",
                 input_cross_attention_num_heads=4,
                 input_cross_attention_mlp_ratio=4.,
                 input_cross_attention_qkv_bias=True,
                 input_cross_attention_qk_scale=None,
                 input_cross_attention_drop_rate=0.3,
                 input_cross_attn_drop_rate=0.,
                 input_cross_drop_path_rate=0.3,
                 output_cross_attention_num_heads=4,
                 output_cross_attention_mlp_ratio=4.,
                 output_cross_attention_qkv_bias=True,
                 output_cross_attention_qk_scale=None,
                 output_cross_attention_drop_rate=0.3,
                 output_cross_attn_drop_rate=0.,
                 output_cross_drop_path_rate=0.3,
                 input_stages=2,
                 output_stages=2,
                 latent_space=200,
                 head_dims=[],
                 **kwargs):
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
        for item in range(len(neuron_list) - 2):
            self.fc.append(nn.Linear(neuron_list[item], neuron_list[item + 1]))

        self.fc_out = nn.Linear(neuron_list[-2], neuron_list[-1])

        self.activation = ActivationFactory.create(
            head_activation_fn) if head_activation_fn else nn.Identity()

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

    def forward(self, g):
        feature_protein = g[0].ndata['h']
        feature_smile = g[1].ndata['h']
        for module in self.protein_graph_conv:
            feature_protein = module(g[0], feature_protein)

        for module in self.ligand_graph_conv:
            feature_smile = module(g[1], feature_smile)

        protein_rep = self.pool_protein(
            g[0], feature_protein).view(1, -1, self.embedding_dim)
        ligand_rep = self.pool_ligand(
            g[1], feature_smile).view(1, -1, self.embedding_dim)
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
        return torch.sigmoid(self.head(x))
