"""
graph_pooling_layers.py

Description:
    This module provides a collection of graph pooling layers designed for deep
    drug discovery applications. Each layer here is a wrapper around a specific
    pooling layer provided by the DGL library, with certain enhancements to provide
    additional functionality, error-checking, or convenience. The layers in this
    module are intended to be used in graph neural networks to aggregate node
    features into a unified graph representation.

    The pooling layers leverage the `GraphPoolingLayerFactory` for easy registration,
    instantiation, and management. This also allows external developers to potentially
    register their own custom pooling layers if needed.

Available Layers:
    - MaxPoolLayer: Implements max pooling over node features.
    - SumPoolLayer: Implements sum pooling over node features.
    - AvgPoolLayer: Implements average pooling over node features.
    - GlobalAttentionPooling: Implements pooling with global attention mechanism.

Dependencies:
    - dgl: The Deep Graph Library, used for the base pooling implementations.
    - torch.nn: PyTorch's neural network module, utilized for creating neural network
                layers when needed, such as in the attention mechanism.

Usage:
To define and register a new custom pooling mechanism:
        @GraphPoolingLayerFactory.register_layer('custom_pooling_name')
        class CustomPooling(AbstractGraphPoolingLayer):
            # ... rest of the implementation

    To create a layer using the factory:
        layer = GraphPoolingLayerFactory.create_layer('custom_pooling_name', **kwargs)

"""

import warnings

from dgl.nn.pytorch.glob import MaxPooling, SumPooling, AvgPooling, GlobalAttentionPooling
import torch.nn as nn

from deepdrugdomain.exceptions import MissingRequiredParameterError
from .factory import GraphPoolingLayerFactory, AbstractGraphPoolingLayer


@GraphPoolingLayerFactory.register_layer('dgl_maxpool')
class MaxPoolLayer(AbstractGraphPoolingLayer):
    """
    Wrapper class for DGL's MaxPooling layer.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MaxPoolLayer instance.
        """
        super().__init__()
        self.layer = MaxPooling()

    def forward(self, g, features):
        """
        Forward pass for the MaxPooling layer.

        Parameters:
        - g: A DGL graph.
        - features: Node features.

        Returns:
        - Tensor: Pooled graph representation.
        """
        return self.layer(g, features)


@GraphPoolingLayerFactory.register_layer('dgl_sumpool')
class MaxPoolLayer(AbstractGraphPoolingLayer):
    """
    Wrapper class for DGL's SumPooling layer.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SumPoolLayer instance.
        """
        super().__init__()
        self.layer = SumPooling()

    def forward(self, g, features):
        """
        Forward pass for the MaxPooling layer.

        Parameters:
        - g: A DGL graph.
        - features: Node features.

        Returns:
        - Tensor: Pooled graph representation.
        """
        return self.layer(g, features)


@GraphPoolingLayerFactory.register_layer('dgl_avgpool')
class MaxPoolLayer(AbstractGraphPoolingLayer):
    """
    Wrapper class for DGL's AvgPooling layer.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MaxPoolLayer instance.
        """
        super().__init__()
        self.layer = AvgPooling()

    def forward(self, g, features):
        """
        Forward pass for the MaxPooling layer.

        Parameters:
        - g: A DGL graph.
        - features: Node features.

        Returns:
        - Tensor: Pooled graph representation.
        """
        return self.layer(g, features)


@GraphPoolingLayerFactory.register_layer('dgl_attentionpool')
class MaxPoolLayer(AbstractGraphPoolingLayer):
    """
    Wrapper class for DGL's GlobalAttentionPooling layer.
    """

    def __init__(self, **kwargs):
        """
        Initialize the MaxPoolLayer instance.
        """
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
        """
        Forward pass for the MaxPooling layer.

        Parameters:
        - g: A DGL graph.
        - features: Node features.

        Returns:
        - Tensor: Pooled graph representation.
        """
        return self.layer(g, features)
