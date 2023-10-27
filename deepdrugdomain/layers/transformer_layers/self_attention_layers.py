from functools import partial
import warnings
from ..utils import LayerFactory, ActivationFactory
from torch import nn


@LayerFactory.register('transformer_attention')
class Attention(nn.Module):
    """ 
    Implements multi-head self attention mechanism as used in Vision Transformers.

    Args:
        dim (int): Dimension of input embeddings.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to use bias for qkv linear transformation. Defaults to False.
        qk_scale (float, optional): Manual scaling for qk. If None, scale is set to `head_dim ** -0.5`. Defaults to None.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.
        proj_drop (float, optional): Dropout rate after the linear projection. Defaults to 0.

    Inputs:
        x (Tensor): Input embeddings tensor of shape (batch_size, num_patches, embed_dim).
        mask (Tensor, optional): Optional mask tensor for attention mechanism.
        return_attn (bool, optional): Whether to return attention weights. Defaults to True.

    Outputs:
        x (Tensor): Tensor after self-attention operation.
        attn (Tensor, optional): Attention weights tensor.
    """
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        defaults = {
            'qkv_bias': False,
            'qk_scale': None,
            'attn_drop': 0.,
            'proj_drop': 0.,
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = kwargs["qk_scale"] or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=kwargs["qkv_bias"])
        self.attn_drop = nn.Dropout(kwargs["attn_drop"])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(kwargs["proj_drop"])

    def forward(self, x, mask=None, return_attn=True):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

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


@LayerFactory.register('transformer_attention_block')
class Block(nn.Module):
    """
    Implements an attention block followed by MLP in Vision Transformers.

    Args:
        dim (int): Dimension of input embeddings.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Scaling factor for MLP hidden dimension. Defaults to 4.0.
        qkv_bias (bool, optional): Whether to use bias for qkv linear transformation in attention mechanism. Defaults to False.
        qk_scale (float, optional): Manual scaling for qk in attention mechanism. Defaults to None.
        drop (float, optional): Dropout rate for MLP. Defaults to 0.
        attn_drop (float, optional): Dropout rate for attention weights in attention mechanism. Defaults to 0.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        act_layer (nn.Module, optional): Activation function for MLP. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        Attention_block (nn.Module, optional): Custom Attention module. Defaults to Attention.
        Mlp_block (nn.Module, optional): Custom MLP module. Defaults to Mlp.
        init_values (float, optional): Initialization values for weights. Defaults to 1e-4.

    Inputs:
        x (Tensor): Input embeddings tensor of shape (batch_size, num_patches, embed_dim).

    Outputs:
        x (Tensor): Tensor after attention and MLP operations.
        attn (Tensor): Attention weights tensor.
    """
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        defaults = {
            'mlp_ratio': 4.,
            'qkv_bias': False,
            'qk_scale': None,
            'drop': 0.,
            'attn_drop': 0.,
            'drop_path': 0.,
            'act_layer': "gelu",
            'norm_layer': "layer_norm",
            'Attention_block': "transformer_attention",
            'Mlp_block': "transformer_mlp",
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
        act_layer = ActivationFactory.create(kwargs["act_layer"])
        norm_layer = kwargs["norm_layer"]

        self.norm1 = LayerFactory.create(
            norm_layer, dim) if norm_layer is not None else nn.Identity()
        self.attn = LayerFactory.create(kwargs['Attention_block'], dim, num_heads,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = LayerFactory.create(
            "drop_path", drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerFactory.create(
            norm_layer, dim) if norm_layer is not None else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LayerFactory.create(
            kwargs["Mlp_block"], dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        v, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(v)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
