"""
base_preprocessor.py

Provides an abstract class for defining preprocessing operations for various data types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class BasePreprocessor(ABC):
    """
    An abstract base class for data preprocessors.
    All custom preprocessors should inherit from this class.
    """

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
        Abstract method for preprocessing the given data.

        Parameters:
        - data: The input data to preprocess.

        Returns:
        - Processed data.
        """
        pass

    @abstractmethod
    def shard_and_save(self, data: Any, shard_directory: str, shard_size: Union[int, None]) -> None:
        """
        Abstract method to shard the preprocessed data and save it.

        Parameters:
        - data: The data to shard and save.
        - shard_directory: The directory where shards should be saved.
        - shard_size: The size of each shard. If None, data isn't sharded.

        Returns:
        - None
        """
        pass

    @abstractmethod
    def generate_mapping(self, data: Any, shard_size: Union[int, None]) -> Dict[Any, Any]:
        """
        Abstract method to generate a mapping between original data and shards.

        Parameters:
        - data: The original data.
        - shard_size: The size of each shard. If None, data isn't sharded.

        Returns:
        - Dictionary mapping from data identifier to shard and position.
        """
        pass

    @abstractmethod
    def process_and_get_info(self, data: Any, shard_directory: str, shard_size: Union[int, None]) -> Dict[str, Any]:
        """
        Process the input data and return all relevant details,
        including preprocessed data, shard information, and mappings.

        Parameters:
        - data: The input data to preprocess.
        - shard_directory: The directory where shards should be saved.
        - shard_size: The size of each shard. If None, data isn't sharded.

        Returns:
        - Dictionary with keys:
          - "preprocessed_data": the processed data
          - "shard_info": shard-related details (e.g., file paths)
          - "mapping": dictionary mapping original data to shard and position.
        """
        pass
