"""
Base Activation Class for Deep Learning Models.

This abstract base class provides a structured template for defining custom activation 
functions or layers in PyTorch. It ensures that any derived class will provide both 
a layer and a function format of the activation, ensuring flexibility in usage.

Example:
    >>> class CustomReLU(BaseActivation):
    ...     @staticmethod
    ...     def activation_layer() -> nn.Module:
    ...         return nn.ReLU()
    ...
    ...     @staticmethod
    ...     def activation_function() -> Callable:
    ...         return torch.relu
    ...
    >>> layer = CustomReLU.activation_layer()
    >>> function = CustomReLU.activation_function()
    >>> output = function(torch.tensor([-1., 0., 1.]))

Attributes:
    - None

Classes:
    BaseActivation: Abstract base class to derive custom activation layers or functions.

"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Callable


class BaseActivation(ABC):
    """
    Abstract Base Class for defining custom activation functions and layers in PyTorch.

    This class ensures that any derived custom activation provides both a layer 
    representation (e.g., a torch.nn.Module) and a function representation (e.g., 
    a callable function).

    Derived classes should implement both `activation_layer` and `activation_function` 
    methods.
    """

    @staticmethod
    @abstractmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        """
        Defines the layer representation of the activation.

        This method should return an instance of `torch.nn.Module` that represents 
        the activation function.

        Returns:
            nn.Module: The activation function in layer format.

        Example:
            >>> relu_layer = CustomReLU.activation_layer()
            >>> output = relu_layer(torch.tensor([-1., 0., 1.]))
        """
        pass

    @staticmethod
    @abstractmethod
    def activation_function(*args, **kwargs) -> Callable:
        """
        Defines the function representation of the activation.

        This method should return a callable function that can be used to apply the 
        activation to a tensor.

        Returns:
            Callable: The activation function in callable format.

        Example:
            >>> relu_function = CustomReLU.activation_function()
            >>> output = relu_function(torch.tensor([-1., 0., 1.]))
        """
        pass
