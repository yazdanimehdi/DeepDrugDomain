from abc import ABC, abstractmethod
from typing import List


class BaseMetric(ABC):
    """
    Abstract base class for metrics. 

    Metrics that require state should store their state as instance variables 
    and may optionally implement the `update_state` method. This class provides 
    default implementations for `update_state` and `reset_state` which do 
    nothing. Subclasses should override these as needed.

    Usage example:

    >>> class MeanAbsoluteError(Metric):
    ...     def compute(self, prediction: List[float], target: List[float]) -> float:
    ...         return sum(abs(p - t) for p, t in zip(prediction, target)) / len(prediction)
    ...
    >>> mae = MeanAbsoluteError()
    >>> result = mae.compute([3, 5], [4, 6])
    >>> print(result)
    1.0

    Note: In this example, `MeanAbsoluteError` does not require state, so 
    `update_state` and `reset_state` methods aren't overridden.
    """

    @abstractmethod
    def compute(self, prediction: List[float], target: List[float]) -> float:
        """
        Compute the metric given a prediction and target.

        Args:
            prediction (List[float]): List of predictions.
            target (List[float]): List of ground truth values.

        Returns:
            float: Computed metric value.
        """
        pass

    def update_state(self, prediction: List[float], target: List[float]) -> None:
        """
        Update state with new data. By default, this does nothing. 
        Override in metrics that need to maintain a state.

        Args:
            prediction (List[float]): List of predictions.
            target (List[float]): List of ground truth values.
        """
        pass

    def reset_state(self) -> None:
        """
        Reset any internal state. Override in metrics that maintain a state.
        """
        pass
