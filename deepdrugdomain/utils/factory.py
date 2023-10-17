from abc import ABC, abstractmethod
from typing import Type, Any, TypeVar, Dict

T = TypeVar('T')


class BaseFactory(ABC):
    """
    Base abstract factory class to facilitate registration and instantiation of components.

    Example usage (In another file, e.g., "graph_conv_layer_factory.py"):
    >>> from deepdrugdomain.utils import BaseFactory
    >>> import torch.nn as nn
    ...
    >>> class GraphConvLayerFactory(BaseFactory):
    ...     _registry = {}
    ...
    >>> @GraphConvLayerFactory.register('my_layer_type')
    ... class MyLayer(nn.Module):
    ...     def forward(self, x):
    ...         return x
    ...
    >>> instance = GraphConvLayerFactory.create('component_type')

    Note: When subclassing BaseFactory, remember to define _registry for each subclass.
    """

    _registry: Dict[str, Type[T]]

    @classmethod
    @abstractmethod
    def register(cls, key: str):
        """
        Decorator method for registering a component to the factory's registry.

        :param key: Unique key to identify the component in the registry.
        :return: Decorator function.
        """
        pass

    @classmethod
    @abstractmethod
    def create(cls, key: str, *args, **kwargs) -> T:
        """
        Create and return an instance of the component.

        :param key: Unique key to fetch the component class from the registry.
        :param args: Positional arguments for component initialization.
        :param kwargs: Keyword arguments for component initialization.
        :return: Instance of the component.
        """
        pass
