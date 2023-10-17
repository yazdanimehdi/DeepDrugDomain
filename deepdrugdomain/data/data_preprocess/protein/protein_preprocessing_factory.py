"""
drug_preprocessor_factory.py

Provides mechanisms to register and retrieve Drug preprocessors.
"""

from .base_preprocessor import BasePreprocessor
from typing import Type, Union, Dict


class ProteinPreprocessorFactory:
    """
    Factory class for registering and retrieving protein data preprocessors.
    """
    preprocessors: Dict[str, BasePreprocessor] = {}

    @staticmethod
    def register(data_type: str) -> callable:
        """
        Decorator method to register a new protein preprocessor.

        Parameters:
        - data_type: A string identifier for the protein preprocessor.

        Returns:
        - Decorator function.
        """
        def decorator(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
            if not issubclass(cls, BasePreprocessor):
                raise ValueError("Only subclasses of BasePreprocessor can be registered.")
            ProteinPreprocessorFactory.preprocessors[data_type] = cls()
            return cls
        return decorator

    @staticmethod
    def get_preprocessor(data_type: str) -> Union[BasePreprocessor, None]:
        """
        Retrieve a registered protein preprocessor based on its type.

        Parameters:
        - data_type: The type of data (e.g., "simple_drug").

        Returns:
        - The registered protein preprocessor. If not found, returns None.
        """
        return ProteinPreprocessorFactory.preprocessors.get(data_type)
