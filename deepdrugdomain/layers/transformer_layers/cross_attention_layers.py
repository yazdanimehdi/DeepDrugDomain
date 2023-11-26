import warnings
from ..utils import LayerFactory, ActivationFactory
from torch import nn
import torch


@LayerFactory.register('transformer_cross_attention')
class Attention_CA(nn.Module):
    """
        Implements Cross Attention mechanism, extracted from Vision Transformer models.

        Args:
        dim (int): Dimension of the input.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If set to True, uses bias in QKV layers. Defaults to False.
        qk_scale (float, optional): Scaling factor for QK. If None, uses head_dim**-0.5. Defaults to None.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
        proj_drop (float, optional): Dropout rate for the output projection. Defaults to 0.
    """
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        defaults = {
            "qkv_bias": False,
            "qk_scale": None,
            "attn_drop": 0.,
            "proj_drop": 0.,
        }

        self.num_heads = num_heads
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        head_dim = dim // num_heads

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        qkv_bias = kwargs["qkv_bias"]
        qk_scale = kwargs["qk_scale"]
        attn_drop = kwargs["attn_drop"]
        proj_drop = kwargs["proj_drop"]

        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask=None, return_attn=False):
        B, M, C = kv.shape
        kv = self.kv(kv).reshape(B, M, 2, self.num_heads, C //
                                 self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C //
                              self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = torch.squeeze(attn, dim=0)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -9e15)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x


@LayerFactory.register('transformer_cross_attention_block')
class Block_CA(nn.Module):
    """
    Implements a block with Cross Attention mechanism. Based on Vision Transformer models.

    Args:
        dim (int): Dimension of the input.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio to scale the MLP hidden dimensions. Defaults to 4.
        qkv_bias (bool, optional): If set to True, uses bias in QKV layers. Defaults to False.
        qk_scale (float, optional): Scaling factor for QK. If None, uses head_dim**-0.5. Defaults to None.
        drop (float, optional): Dropout rate for the output. Defaults to 0.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
        drop_path (float, optional): Drop path rate. Defaults to 0.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        Attention_block (nn.Module, optional): Attention block module. Defaults to Attention_CA.
        Mlp_block (nn.Module, optional): MLP block module. Defaults to Mlp.
        init_values (float, optional): Initial values for weights. Defaults to 1e-4.
    """
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        defaults = {
            "mlp_ratio": 4.,
            "qkv_bias": False,
            "qk_scale": None,
            "drop": 0.,
            "attn_drop": 0.,
            "drop_path": 0.,
            "act_layer": "gelu",
            "norm_layer": "layer_norm",
            "Attention_block": "transformer_cross_attention",
            "Mlp_block": "transformer_mlp",
        }
        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        mlp_ratio = kwargs["mlp_ratio"]
        drop = kwargs["drop"]
        qkv_bias = kwargs["qkv_bias"]
        qk_scale = kwargs["qk_scale"]
        attn_drop = kwargs["attn_drop"]
        drop_path = kwargs["drop_path"]
        act_layer = kwargs["act_layer"]
        norm_layer = kwargs["norm_layer"]
        mlp_block = kwargs["Mlp_block"]
        attention_block = kwargs["Attention_block"]

        self.norm1 = LayerFactory.create(
            norm_layer, dim) if norm_layer is not None else nn.Identity()
        self.norm3 = LayerFactory.create(
            norm_layer, dim) if norm_layer is not None else nn.Identity()
        self.attn = LayerFactory.create(
            attention_block, dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)
        self.drop_path = LayerFactory.create(
            "drop_path", drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerFactory.create(
            norm_layer, dim) if norm_layer is not None else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LayerFactory.create(
            mlp_block, dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, x, mask=None, return_attn=False):
        v, attn = self.attn(self.norm3(q), self.norm1(x), mask, True)
        x = q + self.drop_path(v)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x
