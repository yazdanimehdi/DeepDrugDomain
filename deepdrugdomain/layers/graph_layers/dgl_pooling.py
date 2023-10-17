"""
graph_pooling_layers.py

This module provides a set of classes that act as wrappers around DGL's graph pooling layers.
By leveraging the factory pattern provided by the GraphLayerFactory,
it becomes straightforward to instantiate these layers in a unified manner.

Example:
    >>> from deepdrugdomain.layers.graph_layers import GraphLayerFactory
    >>> max_pool_layer = GraphLayerFactory.create('dgl_maxpool')
    >>> sum_pool_layer = GraphLayerFactory.create('dgl_sumpool')
    >>> attention_pool_layer = GraphLayerFactory.create('dgl_attentionpool', gate_nn_dims=[32, 1])

Requirements:
    - dgl (For DGL's graph pooling layers)
    - torch (For neural network operations)
    - deepdrugdomain (For the base factory class and custom exceptions)
"""

import warnings
from dgl.nn.pytorch.glob import MaxPooling, SumPooling, AvgPooling, GlobalAttentionPooling
import torch.nn as nn
from deepdrugdomain.utils.exceptions import MissingRequiredParameterError
from .graph_layer_factory import GraphLayerFactory, AbstractGraphLayer


@GraphLayerFactory.register('dgl_maxpool')
class MaxPoolLayer(AbstractGraphLayer):
    """
    Wrapper class for DGL's MaxPooling layer.
    """
    def __init__(self):
        super().__init__()
        self.layer = MaxPooling()

    def forward(self, g, features):
        """ Perform max pooling on the graph. """
        return self.layer(g, features)


@GraphLayerFactory.register('dgl_sumpool')
class SumPoolLayer(AbstractGraphLayer):
    """
    Wrapper class for DGL's SumPooling layer.
    """
    def __init__(self):
        super().__init__()
        self.layer = SumPooling()

    def forward(self, g, features):
        """ Perform sum pooling on the graph. """
        return self.layer(g, features)


@GraphLayerFactory.register('dgl_avgpool')
class AvgPoolLayer(AbstractGraphLayer):
    """
    Wrapper class for DGL's AvgPooling layer.
    """
    def __init__(self):
        super().__init__()
        self.layer = AvgPooling()

    def forward(self, g, features):
        """ Perform average pooling on the graph. """
        return self.layer(g, features)


@GraphLayerFactory.register('dgl_attentionpool')
class AttentionPoolLayer(AbstractGraphLayer):
    """
    Wrapper class for DGL's GlobalAttentionPooling layer.
    This layer uses a gating mechanism to determine node importance.
    """
    def __init__(self, **kwargs):
        super().__init__()

        if 'gate_nn_dims' not in kwargs:
            raise MissingRequiredParameterError(self.__class__.__name__, 'gate_nn_dims')

        if 'feat_nn_dims' not in kwargs:
            warnings.warn(
                f"feat_nn_dims parameter is missing. Using default value '{None}' for the '{self.__class__.__name__}''s feat nn.")
            feat_nn = None
        else:
            dims = kwargs['feat_nn_dims']
            layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            feat_nn = nn.ModuleList(layers)

        dims = kwargs['gate_nn_dims']
        layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        gate_nn = nn.ModuleList(layers)

        self.layer = GlobalAttentionPooling(feat_nn=feat_nn, gate_nn=gate_nn)

    def forward(self, g, features):
        """ Perform attention pooling on the graph. """
        return self.layer(g, features)
