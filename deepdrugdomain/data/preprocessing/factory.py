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

from typing import Dict, TypeVar, Tuple, Optional
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
    _transform_helper: Dict[Tuple[str, str], str] = {}

    @classmethod
    def register(cls, key: str, from_dtype: str, to_dtype: str):
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
            cls._transform_helper[(from_dtype, to_dtype)] = key
            return subclass

        return decorator

    @classmethod
    def create(cls, from_dtype: Optional[str] = None, to_dtype: Optional[str] = None, key: Optional[str] = None, *args, **kwargs) -> T:
        """
        Create and return an instance of the preprocessor.

        :param key: Unique key to fetch the preprocessor class from the registry.
        :param args: Positional arguments for preprocessor initialization.
        :param kwargs: Keyword arguments for preprocessor initialization.
        :return: Instance of the preprocessor.
        """
        if from_dtype is None or to_dtype is None:
            if key is None:
                raise ValueError(
                    "Either a key or from_dtype and to_dtype must be specified.")

        if from_dtype is not None and to_dtype is not None:
            if from_dtype == to_dtype:
                return None

            if (from_dtype, to_dtype) not in cls._transform_helper:
                raise ValueError(
                    f"Key for transformation from '{from_dtype}' to '{to_dtype}' not registered.")

            key = cls._transform_helper[(from_dtype, to_dtype)]

        if key not in cls._registry:
            raise ValueError(f"Key '{key}' not registered.")

        return cls._registry[key](*args, **kwargs)

    @classmethod
    def get_preprocessor_name(cls, from_dtype: str, to_dtype: str) -> str:
        """
        Get the name of the preprocessor for the specified data type transformation.

        :param from_dtype: Data type to be transformed from.
        :param to_dtype: Data type to be transformed to.
        :return: Name of the preprocessor.
        """
        if (from_dtype, to_dtype) not in cls._transform_helper:
            raise ValueError(
                f"Key for transformation from '{from_dtype}' to '{to_dtype}' not registered.")

        return cls._transform_helper[(from_dtype, to_dtype)]
