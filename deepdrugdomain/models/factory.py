"""
model_factory.py - A dynamic factory for registering and creating model instances.

The ModelFactory streamlines the process of registering model classes and instantiating them with ease. 
All models must extend `torch.nn.Module`. The factory enables model instantiation with configurations 
from a JSON config file, which can hold default and version-specific configurations, or through explicit 
keyword arguments. Missing instantiation arguments are supplied from the default values which may be set 
during class registration using a decorator or a JSON configuration file. If a version-specific configuration 
is not registered, the factory issues a warning and uses the default configuration.

Example Usage:

Registering a model with a JSON config file containing multiple version configurations:
>>> from deepdrugdomain.models import ModelFactory
>>> import torch.nn as nn
>>>
>>> @ModelFactory.register("my_model_type", model_config_path="my_model_config.json")
>>> class MyModel(nn.Module):
>>>     def __init__(self, learning_rate, batch_size, num_layers):
>>>         super().__init__()
>>>         # Model layers and operations defined here

Registering a model and providing direct default values, which are not version-specific:
>>> @ModelFactory.register("my_model_type", default_values={"learning_rate": 0.01, "batch_size": 32, "num_layers": 2})
>>> class MyModel(nn.Module):
>>>     def __init__(self, learning_rate, batch_size, num_layers):
>>>         super().__init__()
>>>         # Model layers and operations defined here

Instantiating a model with default parameters when no version is specified:
>>> model_default = ModelFactory.create("my_model_type")
>>> model_human_version = ModelFactory.create("my_model_type", version="human")

Creating a model instance using a specified configuration, overriding the defaults if provided:
>>> model_override = ModelFactory.create("my_model_type", learning_rate=0.002, batch_size=64, num_layers=4)

Models must be registered under a unique key and confirm to `torch.nn.Module` for successful instantiation.
"""


import os
import json
import warnings
from typing import Type, Dict, Any, Optional
from torch.nn import Module
from deepdrugdomain.utils import BaseFactory
from .base_model import BaseModel


class ModelFactory(BaseFactory):
    """
        Factory class for model registration and instantiation.

        The ModelFactory class allows for the registration of model classes and their configurations.
        It supports creating model instances with either default or version-specific configurations.
        A single JSON file may contain multiple version-specific configurations along with a default configuration.

        Attributes:
            _registry (Dict[str, Type[Module]]): Registry mapping model keys to model classes.
            _default_args (Dict[str, Dict[str, Any]]): Default and version-specific configurations for registered models.
            _config_dir (str): Path to the configuration directory where JSON files are stored.
    """

    _registry: Dict[str, Type[BaseModel]] = {}
    _default_args: Dict[str, Dict[str, Any]] = {}
    _config_dir: str = 'deepdrugdomain/configs'

    @classmethod
    def register(cls, key: str, model_config_path: Optional[str] = None):
        """
            Register a model class with the factory.

            This decorator function registers the given model class under the specified key. If a configuration
            path is not provided, it tries to load the configuration from the default configs directory.

            Args:
                key (str): The key associated with the model class.
                model_config_path (Optional[str]): Custom path to the model's configuration file.

            Returns:
                A decorator function that takes a model class and registers it.

            Examples:
                >>> @ModelFactory.register("my_model")
                ... class MyModel(nn.Module):
                ...     # model definition
        """
        def decorator(model_class: Type[BaseModel]):
            if not issubclass(model_class, Module):
                raise TypeError(
                    f"Class {model_class.__name__} must be a subclass of torch.nn.Module")

            # Load config for the model from the specified directory
            config_path = os.path.join(
                cls._config_dir, f"{key}.json") if model_config_path is None else model_config_path
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Config file for model '{key}' not found at {config_path}.")

            with open(config_path, 'r') as file:
                config = json.load(file)

            config = config['model']
            # Register the model with its default config and any version-specific configs
            cls._default_args[key] = config.get('default', {})
            for version, version_config in config.items():
                if version != 'default':  # Skip the 'default' key
                    cls._registry[f"{key}_{version}"] = model_class
                    cls._default_args[f"{key}_{version}"] = version_config

            # Also register the model with the default configuration
            cls._registry[key] = model_class
            return model_class

        return decorator

    @classmethod
    def create(cls, key: str, version: Optional[str] = None, **kwargs) -> BaseModel:
        """
            Create an instance of a registered model with the specified or default configuration.

            If a version is specified and a configuration for it exists, the corresponding configuration
            is used; otherwise, it falls back to the default configuration with a warning.

            Args:
                key (str): The key for the model to instantiate.
                version (Optional[str]): The version name to look for a specific configuration.
                **kwargs: Additional keyword arguments to pass to the model constructor.

            Returns:
                An instance of the requested model with the appropriate configuration.

            Raises:
                ValueError: If the model key is not registered.

            Examples:
                >>> model_instance = ModelFactory.create("my_model", "human")
                >>> model_instance_default = ModelFactory.create("my_model")
        """

        if key not in cls._registry:
            raise ValueError(f"Model '{key}' not registered.")

        model_key = f"{key}_{version}" if version and f"{key}_{version}" in cls._registry else key
        if version and f"{key}_{version}" not in cls._registry:
            warnings.warn(
                f"version '{version}' configuration not found for model '{key}'. Using default configuration.", UserWarning)

        # Use default configuration as a base; override with version-specific if exists
        default_config = cls._default_args.get(key, {})
        version_config = cls._default_args.get(model_key, {})
        combined_config = {**default_config, **
                           version_config, **kwargs}  # Merge configurations

        # Use base model class registry for instantiation
        return cls._registry[key](**combined_config)

    @classmethod
    def is_model_registered(cls, key: str, version: Optional[str] = None) -> bool:
        """
            Check if a model is registered with the factory under the given key and version.

            Args:
                key (str): The key associated with the model class.
                version (Optional[str]): The version name to check for a specific registration.

            Returns:
                bool: True if the model is registered with the specified version, False otherwise.

            Examples:
                >>> ModelFactory.is_model_registered("my_model", "human")
                True
                >>> ModelFactory.is_model_registered("my_model", "alien")
                False
                >>> ModelFactory.is_model_registered("my_model")
                True
        """
        model_key = f"{key}_{version}" if version else key
        return model_key in cls._registry
