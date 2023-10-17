"""
collate_factory.py - A factory for registering and instantiating custom collate functions.

The CollateFactory class simplifies the registration and creation process for collate functions.
Users can register their custom collate functions and then instantiate them using a unique key.

Example Usage (In a file like "my_collates.py"):
>>> from deepdrugdomain.data.collate import BaseCollate, CollateFactory
...
>>> @CollateFactory.register('my_collate')
... class MyCollate(BaseCollate):
...     def collate(self, batch: List[Any]) -> Any:
...         # Collation logic here
...         return processed_batch
...
>>> collate_fn = CollateFactory.create('my_collate')
>>> data_loader = DataLoader(dataset, collate_fn=collate_fn)
"""

from typing import Dict, Type, TypeVar, List
from .base_collate import BaseCollate
from deepdrugdomain.utils import BaseFactory

T = TypeVar('T', bound=BaseCollate)


class CollateFactory(BaseFactory):
    """
    Factory class for registering and creating custom collate functions.
    """

    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, key: str):
        """
        Decorator for registering a collate function.

        Parameters:
        - key (str): Unique identifier for the collate function.

        Returns:
        - Type[T]: The registered collate class.
        """

        def decorator(subclass: Type[T]) -> Type[T]:
            if not issubclass(subclass, BaseCollate):
                raise TypeError(f"Class {subclass.__name__} is not a subclass of BaseCollateAbstract")
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, key: str, *args, **kwargs) -> Type[T]:
        """
        Instantiate and return a collate function.

        Parameters:
        - key (str): Unique identifier for the collate function.
        - *args: Positional arguments for the collate function's initialization.
        - **kwargs: Keyword arguments for the collate function's initialization.

        Returns:
        - BaseCollateAbstract: Instance of the collate function.
        """
        if key not in cls._registry:
            raise ValueError(f"Key '{key}' not registered.")

        collate_instance = cls._registry[key](*args, **kwargs)
        return collate_instance
