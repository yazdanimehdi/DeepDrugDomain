"""
graph_conv_layers.py

This module provides wrapper classes around DGL's graph convolution layers.
These wrappers are built on top of the factory pattern provided by the LayerFactory.
It makes instantiating these layers more streamlined and manageable.

Example:
    >>> from deepdrugdomain.layers.graph_layers import GraphLayerFactory
    >>> gcn_layer = LayerFactory.create('dgl_gcn', in_feat=64, out_feat=128)
    >>> gat_layer = LayerFactory.create('dgl_gat', in_feat=64, out_feat=128, num_heads=8)
    >>> tag_layer = LayerFactory.create('dgl_tag', in_feat=64, out_feat=128)

Requirements:
    - dgl (For DGL's graph convolution layers)
    - torch (For tensor operations)
    - deepdrugdomain (For the base factory class and custom exceptions)
"""

from ..utils import LayerFactory, ActivationFactory
import torch
from dgl.nn.pytorch import GraphConv, TAGConv, GATConv
import dgl
import torch.nn as nn
import torch.nn.functional as F
import warnings
from deepdrugdomain.utils import MissingRequiredParameterError


@LayerFactory.register('dgl_gcn')
class GCN(nn.Module):
    """
    Wrapper class for DGL's Graph Convolution (GCN) layer.
    """

    def __init__(self, in_feat, out_feat, normalization=False, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            'norm': 'both',
            # 'weights': True,
            # 'bias': True,
            'activation': "relu",
            'allow_zero_in_degree': False
        }

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        kwargs["activation"] = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else None

        self.layer = GraphConv(in_feats=in_feat, out_feats=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """ Pass the graph and its features through the GCN layer. """
        features = g.ndata['h']
        features = self.layer(g, features)
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_g = g
        new_g.ndata['h'] = features

        return new_g


@LayerFactory.register('dgl_gat')
class GAT(nn.Module):
    """
    Wrapper class for DGL's Graph Attention Network (GAT) layer.
    """

    def __init__(self, in_feat, out_feat, normalization=None, dropout=0.0, **kwargs):
        super().__init__()

        # Default parameter values
        defaults = {
            'feat_drop': 0.,
            'attn_drop': 0.,
            'negative_slope': 0.2,
            'residual': False,
            'activation': "relu",
            'allow_zero_in_degree': False,
            'bias': True
        }
        if 'num_heads' not in kwargs:
            raise MissingRequiredParameterError(
                'num_heads', self.__class__.__name__)

        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        kwargs["activation"] = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else None

        self.layer = GATConv(
            in_feats=in_feat, out_feats=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """ Pass the graph and its features through the GAT layer. """
        features = g.ndata['h']
        features = self.layer(g, features)
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_g = g
        new_g.ndata['h'] = torch.mean(features, dim=1)

        return new_g


@LayerFactory.register('dgl_tag')
class TAG(nn.Module):
    """
    Wrapper class for DGL's Topological Adaptive Graph Convolutional (TAG) layer.
    """

    def __init__(self, in_feat, out_feat, normalization=None, dropout=0.0, **kwargs):
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

        kwargs["activation"] = ActivationFactory.create(
            kwargs['activation']) if kwargs['activation'] is not None else None

        self.layer = TAGConv(in_feats=in_feat, out_feats=out_feat, **kwargs)
        self.norm = normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """ Pass the graph and its features through the TAG layer. """
        features = g.ndata['h']
        features = self.layer(g, features)
        if self.norm:
            features = F.normalize(features)
        features = self.dropout(features)

        new_g = g
        new_g.ndata['h'] = features

        return new_g
