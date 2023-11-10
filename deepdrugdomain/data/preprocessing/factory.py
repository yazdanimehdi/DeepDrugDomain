"""
PreprocessorFactory - A factory for registering and creating preprocessors.

The PreprocessorFactory allows for easy registration and instantiation of different preprocessors.
Registered preprocessors can be instantiated by providing the key used during registration.

Example usage:
>>> from deepdrugdomain.data.preprocessing import PreprocessorFactory
>>> from deepdrugdomain.data.preprocessing import BasePreprocessor
...
>>> @PreprocessorFactory.register('simple_preprocessor')
... class SimplePreprocessor(BasePreprocessor):
...     def __init__(self, **kwargs):
...         super().__init__()
...
...     def preprocess(self, data: str) -> str:
...         return data.lower()
...
>>> preprocessor_instance = PreprocessorFactory.create('simple_preprocessor')
>>> processed_data = preprocessor_instance.preprocess("HELLO WORLD")
>>> print(processed_data)
hello world
"""

from typing import Dict, TypeVar
from .base_preprocessor import AbstractBasePreprocessor
from deepdrugdomain.utils import BaseFactory

T = TypeVar('T', bound='AbstractBasePreprocessor')


class PreprocessorFactory(BaseFactory):
    """
    Factory for registering and creating preprocessors.

    Parameters:
    - None

    Attributes:
    - _registry (dict): Internal registry for mapping keys to preprocessors.
    """

    _registry: Dict[str, T] = {}

    @classmethod
    def register(cls, key: str):
        """
        Decorator method for registering a preprocessor subclass to the factory's registry.

        :param key: Unique key to identify the preprocessor subclass in the registry.
        :return: Decorator function.
        """

        def decorator(subclass):
            if not issubclass(subclass, AbstractBasePreprocessor):
                raise TypeError(
                    f"Class {subclass.__name__} is not a subclass of AbstractBasePreprocessor")
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> T:
        """
        Create and return an instance of the preprocessor.

        :param key: Unique key to fetch the preprocessor class from the registry.
        :param args: Positional arguments for preprocessor initialization.
        :param kwargs: Keyword arguments for preprocessor initialization.
        :return: Instance of the preprocessor.
        """
        if key is None:
            return None

        if key not in cls._registry:
            raise ValueError(f"Key '{key}' not registered.")

        return cls._registry[key](*args, **kwargs)
