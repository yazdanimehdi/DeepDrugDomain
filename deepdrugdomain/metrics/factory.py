"""
Module for MetricFactory, a centralized registry for evaluation metrics.

This module provides a class `MetricFactory` that offers a centralized registry for custom evaluation metrics.
Metric functions can be registered, fetched, and utilized to compute multiple metrics.

Usage example:

>>> from metric_factory_module import MetricFactory
>>> @MetricFactory.register('mean_absolute_error')
... def mean_absolute_error(preds, targets):
...     return sum(abs(p - t) for p, t in zip(preds, targets)) / len(preds)
...
>>> MetricFactory.create('mean_absolute_error')([3, 5], [4, 6])
1.0
"""


from typing import Callable, Dict, List, Union
from deepdrugdomain.utils.factory import BaseFactory


class MetricFactory(BaseFactory):
    """Centralized registry for evaluation metrics.

    This class facilitates registration and use of custom metric functions.
    Each metric function should take in predictions and targets and return a calculated score.

    Attributes:
        _registry: A dictionary to store registered metric functions.
    """

    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, key: str) -> Callable:
        """Decorator to register a new metric function.

        Args:
            name (str): The name of the metric function.

        Raises:
            ValueError: If a metric with the given name is already registered.
        """

        def inner(func: Callable) -> Callable:
            if key in cls._registry:
                raise ValueError(f"Metric '{key}' already registered.")
            cls._registry[key] = func
            return func

        return inner

    @classmethod
    def create(cls, key: str) -> Union[Callable, None]:
        """Fetch the metric function from the registry.

        Args:
            name (str): The name of the metric function to retrieve.

        Returns:
            Callable: The metric function corresponding to the given name.
            None: If no metric function is found for the given name.
        """
        if key not in cls._registry:
            raise ValueError(f"metric '{key}' not registered.")

        return cls._registry[key]()
