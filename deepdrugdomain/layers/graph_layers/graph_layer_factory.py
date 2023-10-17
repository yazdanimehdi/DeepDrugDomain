"""
graph_layer_factory.py

This module defines an abstract graph layer and a corresponding factory for its instantiation.
By using this factory pattern, it becomes straightforward to register new types of graph layers
and then create them on-the-fly based on configuration or user input.

Example:
    >>> from deepdrugdomain.layers.graph_layers.graph_layer_factory import GraphLayerFactory, AbstractGraphLayer
    >>> import torch.nn.functional as F

    >>> @GraphLayerFactory.register('custom_layer')
    ... class CustomLayer(AbstractGraphLayer):
    ...     def __init__(self, in_features, out_features):
    ...         super(CustomLayer, self).__init__()
    ...         self.fc = nn.Linear(in_features, out_features)
    ...
    ...     def forward(self, x):
    ...         return F.relu(self.fc(x))

    >>> layer = GraphLayerFactory.create('custom_layer', in_features=32, out_features=64)
    >>> print(layer)

Requirements:
    - torch (For neural network operations)
    - deepdrugdomain (For the base factory class)

Attributes:
    - AbstractGraphLayer (class): An abstract class for defining graph layers.
    - GraphLayerFactory (class): Factory class for creating graph layers.
"""

from typing import Type
import torch.nn as nn
from abc import ABC, abstractmethod
from deepdrugdomain import BaseFactory


class AbstractGraphLayer(nn.Module, ABC):
    """
    Abstract base class for graph layers.

    All custom graph layers should inherit from this class
    and implement the forward method.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the graph layer.

        This method must be implemented by all subclasses.

        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        pass


class GraphLayerFactory(BaseFactory):
    """
    Factory for creating graph layers.

    Manages registration and instantiation of custom graph layers.
    """
    _registry = {}  # Internal registry for layer types

    @classmethod
    def register(cls, layer_type: str):
        """
        Decorator for registering a new graph layer type.

        Usage:
            @GraphLayerFactory.register('my_layer_type')
            class MyLayer(AbstractGraphLayer):
                ...

        :param layer_type: The type (or key) of the layer being registered.
        :return: Decorator function.
        """
        def decorator(sub_cls: Type[AbstractGraphLayer]):
            if layer_type in cls._registry:
                raise ValueError(f"Layer type {layer_type} already registered")
            cls._registry[layer_type] = sub_cls
            return sub_cls
        return decorator

    @classmethod
    def create(cls, layer_type: str, *args, **kwargs) -> AbstractGraphLayer:
        """
        Create and return an instance of a graph layer.

        :param layer_type: The type (or key) of the layer to be created.
        :param args: Positional arguments for layer initialization.
        :param kwargs: Keyword arguments for layer initialization.
        :return: Instance of AbstractGraphLayer subclass representing the graph layer.
        """
        if layer_type not in cls._registry:
            raise ValueError(f"Layer type {layer_type} not recognized")
        return cls._registry[layer_type](*args, **kwargs)



