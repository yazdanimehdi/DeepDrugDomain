"""
Activation Factory for Deep Learning Models.

This module provides a mechanism to register and instantiate custom activation functions 
or layers in a structured way. Using the `ActivationFactory`, one can centralize the 
management of various activation functionalities, ensuring flexibility and modularity in 
model development.

Example:
    >>> @ActivationFactory.register("custom_relu")
    ... class CustomReLU(BaseActivation):
    ...     @staticmethod
    ...     def activation_layer():
    ...         return nn.ReLU()
    ...     
    ...     @staticmethod
    ...     def activation_function(x):
    ...         return torch.relu(x)
    ...
    >>> activation_layer = ActivationFactory.create("custom_relu")
    >>> activation_function = ActivationFactory.create("custom_relu", function=True)

Attributes:
    _registry (dict): Internal registry for keeping track of registered activation layers.

Classes:
    ActivationFactory: Manages the registration and instantiation of custom activation layers.

"""

from typing import Type, Union, Callable
from torch.nn import Module
from deepdrugdomain.utils import BaseFactory
from .base_activation import BaseActivation


class ActivationFactory(BaseFactory):
    """
    Factory class for handling custom activation layers.

    This class provides methods to register new activation classes and instantiate 
    them either as layer objects or as callable functions.

    Attributes:
        _registry (dict): A dictionary that holds registered activation layer classes.
    """

    _registry = {}  # Internal registry for layer types

    @classmethod
    def register(cls, activation_type: str):
        """
        Register an activation layer class under a specific activation type.

        Parameters:
            activation_type (str): The name/type of the activation to be registered.

        Returns:
            Callable: A decorator to be applied on the custom activation class.

        Raises:
            ValueError: If the given activation_type is already registered.

        Example:
            >>> @ActivationFactory.register("my_activation")
            ... class MyActivation(BaseActivation):
            ...     ...
        """
        def decorator(sub_cls: Type[BaseActivation]):
            if activation_type in cls._registry:
                raise ValueError(
                    f"Layer type {activation_type} already registered")
            cls._registry[activation_type] = sub_cls
            return sub_cls
        return decorator

    @classmethod
    def create(cls, activation_type: str, function: bool = False, *args, **kwargs) -> Union[Module, Callable]:
        """
        Instantiate and return a registered activation either as a layer or as a function.

        Parameters:
            activation_type (str): The name/type of the desired activation.
            function (bool, optional): If True, return the activation as a function, 
                                       else return it as a layer. Defaults to False.

        Returns:
            Module or Callable: The desired activation, either as a layer or as a function.

        Raises:
            ValueError: If the given activation_type is not registered.

        Example:
            >>> activation_layer = ActivationFactory.create("my_activation")
            >>> activation_function = ActivationFactory.create("my_activation", function=True)
        """
        if activation_type not in cls._registry:
            raise ValueError(f"Layer type {activation_type} not recognized")

        if function:
            return cls._registry[activation_type].activation_function(*args, **kwargs)
        else:
            return cls._registry[activation_type].activation_layer(*args, **kwargs)
