"""
layer_factory.py

This module defines an abstract layer and a corresponding factory for its instantiation.
By using this factory pattern, it becomes straightforward to register new types of layers
and then create them on-the-fly based on configuration or user input.

Example:
    >>> from deepdrugdomain.layers.layer_factory import LayerFactory, AbstractLayer
    >>> import torch.nn.functional as F

    >>> @LayerFactory.register('custom_layer')
    ... class CustomLayer(AbstractLayer):
    ...     def __init__(self, in_features, out_features):
    ...         super(CustomLayer, self).__init__()
    ...         self.fc = nn.Linear(in_features, out_features)
    ...
    ...     def forward(self, x):
    ...         return F.relu(self.fc(x))

    >>> layer = LayerFactory.create('custom_layer', in_features=32, out_features=64)
    >>> print(layer)

Requirements:
    - torch (For neural network operations)
    - deepdrugdomain (For the base factory class)

Attributes:
    - AbstractLayer (class): An abstract class for defining layers.
    - LayerFactory (class): Factory class for creating layers.
"""

from typing import Type
from torch.nn import Module
from deepdrugdomain.utils import BaseFactory


class LayerFactory(BaseFactory):
    """
    Factory for creating layers.

    Manages registration and instantiation of custom layers.
    """
    _registry = {}  # Internal registry for layer types

    @classmethod
    def register(cls, layer_type: str):
        """
        Decorator for registering a new layer type.

        Usage:
            @LayerFactory.register('my_layer_type')
            class MyLayer(AbstractLayer):
                ...

        :param layer_type: The type (or key) of the layer being registered.
        :return: Decorator function.
        """
        def decorator(sub_cls: Type[Module]):
            if layer_type in cls._registry:
                raise ValueError(f"Layer type {layer_type} already registered")
            cls._registry[layer_type] = sub_cls
            return sub_cls
        return decorator

    @classmethod
    def create(cls, layer_type: str, *args, **kwargs) -> Module:
        """
        Create and return an instance of a layer.

        :param layer_type: The type (or key) of the layer to be created.
        :param args: Positional arguments for layer initialization.
        :param kwargs: Keyword arguments for layer initialization.
        :return: Instance of AbstractLayer subclass representing the layer.
        """
        if layer_type not in cls._registry:
            raise ValueError(f"Layer type {layer_type} not recognized")
        return cls._registry[layer_type](*args, **kwargs)
