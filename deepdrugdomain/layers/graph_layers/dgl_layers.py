"""
graph_conv_layers.py

This module provides wrapper classes around DGL's graph convolution layers.
These wrappers are built on top of the factory pattern provided by the GraphLayerFactory.
It makes instantiating these layers more streamlined and manageable.

Example:
    >>> from deepdrugdomain.layers.graph_layers import GraphLayerFactory
    >>> gcn_layer = GraphLayerFactory.create('dgl_gcn', in_feat=64, out_feat=128)
    >>> gat_layer = GraphLayerFactory.create('dgl_gat', in_feat=64, out_feat=128, num_heads=8)
    >>> tag_layer = GraphLayerFactory.create('dgl_tag', in_feat=64, out_feat=128)

Requirements:
    - dgl (For DGL's graph convolution layers)
    - torch (For tensor operations)
    - deepdrugdomain (For the base factory class and custom exceptions)
"""

from .graph_layer_factory import GraphLayerFactory, AbstractGraphLayer
import torch
from dgl.nn.pytorch import GraphConv, TAGConv, GATConv
import torch.nn.functional as F
from deepdrugdomain.utils.exceptions import MissingRequiredParameterError
import warnings


@GraphLayerFactory.register('dgl_gcn')
class GCN(AbstractGraphLayer):
    """
    Wrapper class for DGL's Graph Convolution (GCN) layer.
    """
    def __init__(self, in_feat, out_feat, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            'norm': 'both',
            'weights': True,
            'bias': True,
            'activation': F.relu,
            'allow_zero_in_degree': False
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.layer = GraphConv(in_feats=in_feat, out_feats=out_feat, **kwargs)

    def forward(self, g, features: torch.Tensor) -> torch.Tensor:
        """ Pass the graph and its features through the GCN layer. """
        return self.layer(g, features)


@GraphLayerFactory.register('dgl_gat')
class GAT(AbstractGraphLayer):
    """
    Wrapper class for DGL's Graph Attention Network (GAT) layer.
    """
    def __init__(self, in_feat, out_feat, **kwargs):
        super().__init__()

        if 'num_heads' not in kwargs:
            raise MissingRequiredParameterError(self.__class__.__name__, 'num_heads')

        # Default parameter values
        defaults = {
            'feat_drop': 0.,
            'attn_drop': 0.,
            'negative_slope': 0.2,
            'residual': False,
            'activation': F.relu,
            'allow_zero_in_degree': False,
            'bias': True
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.layer = GATConv(in_feats=in_feat, out_feats=out_feat, **kwargs)

    def forward(self, g, features: torch.Tensor) -> torch.Tensor:
        """ Pass the graph and its features through the GAT layer. """
        return self.layer(g, features)


@GraphLayerFactory.register('dgl_tag')
class TAG(AbstractGraphLayer):
    """
    Wrapper class for DGL's Topological Adaptive Graph Convolutional (TAG) layer.
    """
    def __init__(self, in_feat, out_feat, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            'k': 2,
            'bias': True,
            'activation': None
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.layer = TAGConv(in_feats=in_feat, out_feats=out_feat, **kwargs)

    def forward(self, g, features: torch.Tensor) -> torch.Tensor:
        """ Pass the graph and its features through the TAG layer. """
        return self.layer(g, features)
