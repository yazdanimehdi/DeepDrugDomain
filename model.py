from functools import partial

from dgl.nn.pytorch import GATConv, MaxPooling, TAGConv

from layers import Block_CA, Block, Attention, Attention_CA, Mlp
import torch
from torch import nn
import torch.nn.functional as F
from weight_init import trunc_normal_


class PerceiverIODTI(nn.Module):
    def __init__(self, embed_dim=256, depth=2, num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.05, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=None, block_layers=Block,
                 act_layer=nn.GELU, attention_block=Attention, mlp_block=Mlp, cross_att_block=Block_CA, input_stages=2,
                 output_stages=2, latent_space=300, dpr_constant=True, mlp_ratio_ca=6.0,  drop_rate_ca=0.0, **kwargs):
        super().__init__()
        self.embedding_dim = embed_dim
        self.protein_graph_conv = nn.ModuleList()
        # self.protein_graph_conv.append(TAGConv(74, 74, 4))
        # self.protein_graph_conv.append(TAGConv(74, 74, 4))
        self.protein_graph_conv.append(TAGConv(74, 74, 4))
        self.protein_graph_conv.append(TAGConv(74, embed_dim // 2, 4))
        self.protein_graph_conv.append(GATConv(embed_dim // 2, embed_dim, 2))

        self.ligand_graph_conv = nn.ModuleList()

        self.ligand_graph_conv.append(TAGConv(74, 74, 4))
        self.ligand_graph_conv.append(TAGConv(74, embed_dim // 2, 4))
        self.ligand_graph_conv.append(GATConv(embed_dim // 2, embed_dim, 2))

        self.latent_query = nn.Parameter(torch.zeros(1, latent_space, embed_dim))
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=attention_block, Mlp_block=mlp_block)
            for i in range(depth)])

        self.blocks_ca_input = nn.ModuleList([
            Block_CA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_ca, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop_rate_ca, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                     act_layer=act_layer, Attention_block=cross_att_block, Mlp_block=mlp_block) for i in
            range(input_stages)
        ])

        self.blocks_ca_output = nn.ModuleList([
            Block_CA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                     act_layer=act_layer, Attention_block=cross_att_block, Mlp_block=mlp_block) for i in
            range(output_stages)
        ])

        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, 1)
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
            feature_protein = F.relu(module(g[0], feature_protein))
            feature_protein = F.dropout(feature_protein, 0.05)
            # feature_protein = F.normalize(feature_protein)

        for module in self.ligand_graph_conv:
            feature_smile = F.relu(module(g[1], feature_smile))
            feature_smile = F.dropout(feature_smile, 0.05)
            # feature_smile = F.normalize(feature_smile)

        pool_ligand = MaxPooling()
        pool_protein = MaxPooling()
        protein_rep = pool_protein(g[0], feature_protein).view(1, -1, self.embedding_dim)
        ligand_rep = pool_ligand(g[1], feature_smile).view(1, -1, self.embedding_dim)
        x = self.latent_query.expand(1, -1, -1)
        for i, blk in enumerate(self.blocks_ca_input):
            x, _ = blk(x, protein_rep)

        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)

        for i, blk in enumerate(self.blocks_ca_output):
            x, _ = blk(x, ligand_rep)

        x = self.norm(x)
        x = torch.mean(x, dim=1)
        return torch.sigmoid(self.head(x))



