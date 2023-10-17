"""
Module implementing various graph convolutional layers using the Deep Graph Library (DGL)
and PyTorch. This module defines GCN, GAT, and TAG layers with custom initialization
defaults and provides the ability to register these layers for use within a factory
pattern. It also handles missing parameters through warnings and sets them to default values.

Usage:
    To define and register a new custom layer:
        @GraphConvLayerFactory.register_layer('custom_layer_name')
        class CustomLayer(AbstractGraphConvLayer):
            # ... rest of the implementation

    To create a layer using the factory:
        layer = GraphConvLayerFactory.create_layer('custom_layer_name', *args, **kwargs)
"""
from torch import nn

from .factory import GraphConvLayerFactory, AbstractGraphConvLayer
import torch
from dgl.nn.pytorch import GraphConv, TAGConv, GATConv
import torch.nn.functional as F
from deepdrugdomain.exceptions import MissingRequiredParameterError
import warnings


@GraphConvLayerFactory.register_layer('dgl_gcn')
class GCN(AbstractGraphConvLayer):
    """
    Defines the Graph Convolutional Network (GCN) layer using DGL's GraphConv.

    Attributes:
        layer (GraphConv): The internal GCN layer from DGL.
    """

    def __init__(self, in_feat, out_feat, **kwargs):
        """
        Initializes the GCN layer.

        Args:
            in_feat (int): Number of input features.
            out_feat (int): Number of output features.
            **kwargs: Additional arguments for the GraphConv layer.
        """
        super().__init__()
        defaults = {
            'norm': 'both',
            'weights': True,
            'bias': True,
            'activation': F.relu,
            'allow_zero_in_degree': False
        }  # Default parameter values

        missing_keys = defaults.keys() - kwargs.keys()  # Identify missing parameters

        for key in missing_keys:
            warnings.warn(
                f"'{key}' parameter is missing. Using default value '{defaults[key]}' for the '{self.__class__.__name__}' layer.")
            kwargs[key] = defaults[key]

        self.layer = GraphConv(in_feats=in_feat, out_feats=out_feat, **kwargs)

    def forward(self, g, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GCN layer.

        Args:
            g (DGLGraph): The input graph.
            features (torch.Tensor): Node features.

        Returns:
            torch.Tensor: Output after GCN layer.
        """
        return self.layer(g, features)


@GraphConvLayerFactory.register_layer('dgl_gat')
class GAT(AbstractGraphConvLayer):
    """
    Defines the Graph Attention Network (GAT) layer using DGL's GATConv.

    Attributes:
        layer (GATConv): The internal GAT layer from DGL.
    """

    def __init__(self, in_feat, out_feat, **kwargs):
        """
        Initializes the GAT layer.

        Args:
            in_feat (int): Number of input features.
            out_feat (int): Number of output features.
            **kwargs: Additional arguments for the GATConv layer.
        """
        super().__init__()

        if 'num_heads' not in kwargs:
            raise MissingRequiredParameterError(self.__class__.__name__, 'num_heads')

        defaults = {
            'feat_drop': 0.,
            'attn_drop': 0.,
            'negative_slope': 0.2,
            'residual': False,
            'activation': F.relu,
            'allow_zero_in_degree': False,
            'bias': True
        }

        missing_keys = defaults.keys() - kwargs.keys()

        for key in missing_keys:
            warnings.warn(
                f"'{key}' parameter is missing. Using default value '{defaults[key]}' for the '{self.__class__.__name__}' layer.")
            kwargs[key] = defaults[key]

        self.layer = GATConv(in_feats=in_feat, out_feats=out_feat, **kwargs)

    def forward(self, g, features: torch.Tensor) -> torch.Tensor:
        return self.layer(g, features)


@GraphConvLayerFactory.register_layer('dgl_tag')
class TAG(AbstractGraphConvLayer):
    """
    Defines the Topology Adaptive Graph Convolutional (TAG) network layer using DGL's TAGConv.

    Attributes:
        layer (TAGConv): The internal TAGConv layer from DGL.
    """

    def __init__(self, in_feat, out_feat, **kwargs):
        """
        Initializes the TAG layer.

        Args:
            in_feat (int): Number of input features.
            out_feat (int): Number of output features.
            **kwargs: Additional arguments for the TAGConv layer.
        """
        super().__init__()
        defaults = {
            'k': 2,
            'bias': True,
            'activation': None
        }

        missing_keys = defaults.keys() - kwargs.keys()

        for key in missing_keys:
            warnings.warn(
                f"'{key}' parameter is missing. Using default value '{defaults[key]}' for the '{self.__class__.__name__}' layer.")
            kwargs[key] = defaults[key]

        self.layer = TAGConv(in_feats=in_feat, out_feats=out_feat, **kwargs)

    def forward(self, g, features: torch.Tensor) -> torch.Tensor:
        return self.layer(g, features)


