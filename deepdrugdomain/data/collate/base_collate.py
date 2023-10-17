"""
base_collate.py - Contains an abstract class for custom collate functions.

The BaseCollate class provides a blueprint for defining custom collate functions 
for PyTorch's DataLoader. By defining subclasses of BaseCollate, different collation 
strategies can be implemented.

Example Usage (In another file, e.g., "my_collates.py"):
>>> from deepdrugdomain.data.collate import BaseCollate
... class MyCollate(BaseCollate):
...     def __call__(self, batch: List[Any]) -> Any:
...         # Collation logic here
...         return processed_batch
"""

from abc import ABC, abstractmethod
from typing import Any, List


class BaseCollate(ABC):
    """
    Abstract class to be used as a blueprint for creating custom collate functions.
    """

    @abstractmethod
    def __call__(self, batch: List[Any]) -> Any:
        """
        Abstract method that defines the collate operation.

        Parameters:
        - batch (List[Any]): List of data items.

        Returns:
        - Any: Collated data.
        """
        pass