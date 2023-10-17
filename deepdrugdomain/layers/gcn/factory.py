"""
factory.py

Description:
    This module provides factory functionality to dynamically create and manage
    different types of graph convolution layers, specifically designed for deep
    drug discovery applications. The module leverages the Factory Design Pattern
    to enable flexible instantiation of layer types, while also providing a mechanism
    for external developers to register and define their own custom graph convolution
    layer implementations.

Main Components:
    - AbstractGraphConvLayer: An abstract base class representing the blueprint
                              for any graph convolution layer.

    - AbstractGraphConvLayerFactory: An abstract base class for the layer factory.

    - GraphConvLayerFactory: The main factory class responsible for registering,
                             managing, and creating different graph convolution layers.

Usage:
    from factory import GraphConvLayerFactory

    # Register a custom layer (external developers can use this pattern)
    @GraphConvLayerFactory.register_layer('custom_layer')
    class CustomGraphConvLayer(AbstractGraphConvLayer):
        ...

    # Instantiate a layer using the factory
    layer_instance = GraphConvLayerFactory.create_layer('custom_layer', *args, **kwargs)

"""

import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractGraphConvLayer(nn.Module, ABC):
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


class AbstractGraphConvLayerFactory(ABC):
    """
    Abstract base class for the graph convolution layer factory.

    Defines the expected interface for concrete factories.
    """

    @staticmethod
    @abstractmethod
    def create_layer(layer_type: str, *args, **kwargs) -> AbstractGraphConvLayer:
        """
        Abstract method to create a graph convolution layer instance.

        Must be implemented by concrete factory classes.

        :param layer_type: The type (or key) of the layer to be created.
        :param args: Positional arguments for layer initialization.
        :param kwargs: Keyword arguments for layer initialization.
        :return: Instance of AbstractGraphConvLayer.
        """
        pass


class GraphConvLayerFactory(AbstractGraphConvLayerFactory):
    """
    Factory for creating graph convolution layers.

    Manages registration and instantiation of custom graph convolution layers.
    """

    _registry = {}  # Internal registry for layer types

    @staticmethod
    def register_layer(layer_type: str):
        """
        Decorator for registering a new graph convolution layer type.

        Usage:
            @GraphConvLayerFactory.register_layer('my_layer_type')
            class MyLayer(AbstractGraphConvLayer):
                ...

        :param layer_type: The type (or key) of the layer being registered.
        :return: Decorator function.
        """
        def decorator(cls: type):
            if layer_type in GraphConvLayerFactory._registry:
                raise ValueError(f"Layer type {layer_type} already registered")
            GraphConvLayerFactory._registry[layer_type] = cls
            return cls
        return decorator

    @staticmethod
    def create_layer(layer_type: str, *args, **kwargs) -> AbstractGraphConvLayer:
        """
        Create and return an instance of a graph convolution layer.

        :param layer_type: The type (or key) of the layer to be created.
        :param args: Positional arguments for layer initialization.
        :param kwargs: Keyword arguments for layer initialization.
        :return: Instance of AbstractGraphConvLayer.
        """
        if layer_type not in GraphConvLayerFactory._registry:
            raise ValueError(f"Layer type {layer_type} not recognized")
        return GraphConvLayerFactory._registry[layer_type](*args, **kwargs)
