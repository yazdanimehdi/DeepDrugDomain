"""
model_factory.py - Factory for creating instances of model classes.

The ModelFactory facilitates easy registration of model classes and subsequent instantiation. 
Models must inherit from `torch.nn.Module`. It can instantiate a model using configurations 
provided either via a JSON config file or directly as keyword arguments. For missing arguments,
the factory will issue a warning and use the default values provided during registration.

Example Usage:

Registering a model:
>>> from deepdrugdomain.models import ModelFactory
>>> import torch.nn as nn
>>>
>>> @ModelFactory.register("my_model_type", required_args={"num_layers": 2, "hidden_size": 64})
>>> class MyModel(nn.Module):
>>>     def __init__(self, num_layers, hidden_size):
>>>         super().__init__()
>>>         self.num_layers = num_layers
>>>         self.hidden_size = hidden_size

Instantiating a model:
>>> model = ModelFactory.create("my_model_type", config_path="path_to_config.json")
... # OR
>>> model = ModelFactory.create("my_model_type", num_layers=3, hidden_size=128)

Ensure that the model class is registered with its associated key before instantiation and 
that it inherits from `torch.nn.Module`.
"""

import json
import warnings
from typing import Dict, Type, Any, Optional
from torch.nn import Module
from deepdrugdomain.utils import BaseFactory

class ModelFactory(BaseFactory):
    """
    Factory for registering and creating model instances.

    The factory facilitates easy registration and instantiation of models.
    Models can be instantiated using configurations from a JSON config file or 
    directly via keyword arguments. All registered models must inherit from nn.Module.

    Attributes:
    - _registry: Internal registry for mapping model keys to model classes.
    - _required_args: Dictionary holding the required arguments for initializing each model.
    - _default_args: Dictionary holding the default values for required arguments.
    """

    _registry: Dict[str, Type[Module]] = {}
    _required_args: Dict[str, set] = {}
    _default_args: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, key: str, required_args: Optional[Dict[str, Any]] = None):
        """
        Decorator method for registering a model class.

        Parameters:
        - key: Unique key to identify the model class.
        - required_args: Dictionary of required arguments with their default values for initializing the model.

        Returns:
        - Decorator function.
        """
        if required_args is None:
            required_args = {}

        def decorator(model_class: Type[Module]):
            if not issubclass(model_class, Module):
                raise TypeError(f"Class {model_class.__name__} must be a subclass of torch.nn.Module")
            
            cls._registry[key] = model_class
            cls._required_args[key] = set(required_args.keys())
            cls._default_args[key] = required_args
            return model_class

        return decorator

    @classmethod
    def create(cls, key: str, config_path: str = None, **kwargs) -> Module:
        """
        Create and return an instance of the model.

        Parameters:
        - key: Unique key to fetch the model class from the registry.
        - config_path: Path to JSON file holding the model's configuration.
        - kwargs: Additional keyword arguments for model initialization.

        Returns:
        - Model instance.
        """
        if key not in cls._registry:
            raise ValueError(f"Model key '{key}' not registered.")

        if config_path:
            with open(config_path, 'r') as file:
                config = json.load(file)
                kwargs.update(config)

        for arg in cls._required_args[key]:
            if arg not in kwargs:
                default_value = cls._default_args[key][arg]
                warnings.warn(f"Missing required argument '{arg}' for model '{key}'. Using default value of {default_value}.")
                kwargs[arg] = default_value

        return cls._registry[key](**kwargs)