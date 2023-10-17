
from typing import Type

import torch.nn as nn
from abc import ABC, abstractmethod

from deepdrugdomain import BaseFactory


class AbstractGraphLayer(nn.Module, ABC):
    """
    Abstract base class for graph convolution layers.

    All custom graph convolution layers should inherit from this class
    and implement the forward method.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass for the graph convolution layer.

        This method must be implemented by all subclasses.

        :param args: Positional arguments
        :param kwargs: Keyword arguments
        """
        pass


class GraphConvLayerFactory(BaseFactory):
    """
    Factory for creating graph convolution layers.

    Manages registration and instantiation of custom graph convolution layers.
    """
    _registry = {}  # Internal registry for layer types

    @classmethod
    def register(cls, layer_type: str):
        """
        Decorator for registering a new graph convolution layer type.

        Usage:
            @GraphConvLayerFactory.register('my_layer_type')
            class MyLayer(nn.Module):
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
        Create and return an instance of a graph convolution layer.

        :param layer_type: The type (or key) of the layer to be created.
        :param args: Positional arguments for layer initialization.
        :param kwargs: Keyword arguments for layer initialization.
        :return: Instance of nn.Module subclass representing the graph convolution layer.
        """
        if layer_type not in cls._registry:
            raise ValueError(f"Layer type {layer_type} not recognized")
        return cls._registry[layer_type](*args, **kwargs)


