"""
This module provides a factory pattern implementation for creating optimizer objects in PyTorch. 

The module defines an `OptimizerFactory` class that inherits from `BaseFactory` to register and instantiate optimizers. 
The registration of optimizer types allows for dynamic creation of optimizers using string keys associated with each 
optimizer class. This can help in managing different optimizer configurations and instantiating them as needed during 
training of machine learning models.

Example:
    >>> from torch.optim import SGD, Adam
    >>> OptimizerFactory.register("sgd")(SGD)
    >>> OptimizerFactory.register("adam")(Adam)
    >>> sgd_optimizer = OptimizerFactory.create("sgd", lr=0.01, momentum=0.9)
    >>> adam_optimizer = OptimizerFactory.create("adam", lr=0.001)
    Here, the `register` class method is used as a decorator to add the SGD and Adam optimizer classes to the factory's 
    registry with respective keys 'sgd' and 'adam'. These can then be instantiated with `create` method calls, 
    providing appropriate arguments such as learning rate (`lr`) and momentum for the SGD optimizer.
"""


from typing import Dict, Type, TypeVar, List
from torch.optim.optimizer import Optimizer
from deepdrugdomain.utils import BaseFactory

T = TypeVar('T', bound=Optimizer)


class OptimizerFactory(BaseFactory):
    """
    A factory class for creating optimizer instances.

    This class provides a registry for optimizer classes, allowing them to be created dynamically based on a key.

    Attributes:
        _registry (Dict[str, Type[T]]): A dictionary mapping keys to optimizer classes.
    """

    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, key: str):
        """
        A decorator method for registering optimizer classes with the factory.

        Args:
            key (str): The key to use for registering the optimizer class.

        Returns:
            A decorator function that takes an optimizer class and registers it with the factory.
        """

        def decorator(subclass: Type[T]) -> Type[T]:
            """
            The decorator function that registers the optimizer class with the factory.

            Args:
                subclass (Type[T]): The optimizer class to register.

            Returns:
                The optimizer class that was passed in.
            """
            if not issubclass(subclass, Optimizer):
                raise TypeError(
                    f"Class {subclass.__name__} is not a subclass of Optimizer")
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, key: str, params, *args, **kwargs) -> Type[T]:
        """
        A method for creating optimizer instances based on a key.

        Args:
            key (str): The key to use for looking up the optimizer class.
            *args: Positional arguments to pass to the optimizer constructor.
            **kwargs: Keyword arguments to pass to the optimizer constructor.

        Returns:
            An instance of the optimizer class corresponding to the given key.
        """
        if key not in cls._registry:
            raise ValueError(f"Key '{key}' not registered.")

        optimizer_instance = cls._registry[key](params, *args, **kwargs)
        return optimizer_instance
