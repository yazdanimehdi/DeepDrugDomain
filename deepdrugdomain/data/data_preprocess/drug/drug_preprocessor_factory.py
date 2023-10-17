"""
drug_preprocessor_factory.py

Provides mechanisms to register and retrieve Drug preprocessors.
"""

from .base_preprocessor import BasePreprocessor
from typing import Type, Union, Dict


class DrugPreprocessorFactory:
    """
    Factory class for registering and retrieving drug data preprocessors.
    """
    preprocessors: Dict[str, BasePreprocessor] = {}

    @staticmethod
    def register(data_type: str) -> callable:
        """
        Decorator method to register a new drug preprocessor.

        Parameters:
        - data_type: A string identifier for the drug preprocessor.

        Returns:
        - Decorator function.
        """
        def decorator(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
            if not issubclass(cls, BasePreprocessor):
                raise ValueError("Only subclasses of BasePreprocessor can be registered.")
            DrugPreprocessorFactory.preprocessors[data_type] = cls()
            return cls
        return decorator

    @staticmethod
    def get_preprocessor(data_type: str) -> Union[BasePreprocessor, None]:
        """
        Retrieve a registered drug preprocessor based on its type.

        Parameters:
        - data_type: The type of data (e.g., "simple_drug").

        Returns:
        - The registered drug preprocessor. If not found, returns None.
        """
        return DrugPreprocessorFactory.preprocessors.get(data_type)
