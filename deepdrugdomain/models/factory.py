"""
model_factory.py - Factory for creating instances of model classes.

The ModelFactory facilitates easy registration of model classes and subsequent instantiation. 
Models must inherit from `torch.nn.Module`. The factory can instantiate models using configurations 
from a JSON config file (either the default one provided during registration or an external one) 
or directly through keyword arguments. If any arguments are missing during instantiation, the 
factory will use the default values provided either directly during registration or from the default 
config file.

Example Usage:

Registering a model with a default config file:
>>> from deepdrugdomain.models import ModelFactory
>>> import torch.nn as nn
>>>
>>> @ModelFactory.register("my_model_type")
>>> class MyModel(nn.Module):
>>>     def __init__(self, num_layers, hidden_size):
>>>         super().__init__()
>>>         self.num_layers = num_layers
>>>         self.hidden_size = hidden_size

Registering a model with directly provided default values:
>>> @ModelFactory.register("my_model_type", default_values={"num_layers": 2, "hidden_size": 64})
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


import os
import json
from typing import Type, Dict, Any, Optional
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
    - _default_args: Dictionary holding the default values for required arguments.
    """

    _registry: Dict[str, Type[Module]] = {}
    _default_args: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, key: str, default_values: Optional[Dict[str, Any]] = None, config_dir: str = 'deepdrugdomain/models/configs'):
        """
        Decorator method for registering a model class.

        Parameters:
        - key: Unique key to identify the model class.
        - default_values: Directly provided default values for the model.
        - config_dir: Directory containing the default JSON configuration files for models.

        Returns:
        - Decorator function.
        """

        def decorator(model_class: Type[Module]):
            if not issubclass(model_class, Module):
                raise TypeError(
                    f"Class {model_class.__name__} must be a subclass of torch.nn.Module")

            if default_values is None:
                # Load default config for the model from the specified directory
                config_path = os.path.join(config_dir, f"{key}.json")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(
                        f"Default config file for model '{key}' not found at {config_path} and no default values provided.")

                with open(config_path, 'r') as file:
                    config = json.load(file)
                cls._default_args[key] = config
            else:
                cls._default_args[key] = default_values

            cls._registry[key] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, key: str, config_path: str = None, **kwargs) -> Module:
        """
        Create and return an instance of the model.

        Parameters:
        - key: Unique key to fetch the model class from the registry.
        - config_path: Path to JSON file holding the model's configuration (overrides default config).
        - kwargs: Additional keyword arguments for model initialization.

        Returns:
        - Model instance.
        """
        if key not in cls._registry:
            raise ValueError(f"Model key '{key}' not registered.")

        default_config = cls._default_args.get(key, {})
        if config_path:
            with open(config_path, 'r') as file:
                override_config = json.load(file)
                default_config.update(override_config)

        kwargs = {**default_config, **kwargs}

        return cls._registry[key](**kwargs)

    @classmethod
    def is_model_registered(cls, model_name: str) -> bool:
        """
            Check if a model with the given name is registered in the factory's registry.

            This method queries the model registry to determine if a model with the specified
            name has been added to the registry. The registry is a class-level dictionary
            storing model names as keys.

            Parameters:
            model_name (str): The name of the model to check in the registry.

            Returns:
            bool: True if the model is registered, False otherwise.
        """
        return model_name in cls._registry
