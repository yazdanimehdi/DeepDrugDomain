from typing import Callable, Dict, List, Union, Any
from .factory import MetricFactory
import numpy as np
import torch


def create_suitable_vector(vector: Any) -> np.ndarray:
    new_vector = []
    if isinstance(vector, list):
        if isinstance(vector[0], list):
            new_vector = np.array([item for item in zip(*vector)]).reshape(-1)
        elif isinstance(vector[0], float):
            new_vector = np.array(vector).reshape(-1)
        elif isinstance(vector[0], np.ndarray):
            new_vector = np.concatenate(vector).reshape(-1)
        elif isinstance(vector[0], torch.Tensor):
            new_vector = torch.cat(vector).view(-1).numpy()
        else:
            raise TypeError('The type of the vector is not supported.')

    elif isinstance(vector, torch.Tensor):
        new_vector = vector.view(-1).numpy()

    elif isinstance(vector, np.ndarray):
        new_vector = vector.reshape(-1)
    else:
        raise TypeError('The type of the vector is not supported.')

    return new_vector


class Evaluator:
    """
    Evaluator class for computing and updating metrics based on predictions.

    This class facilitates the computation and updating of multiple metrics on predictions.
    Once initialized with the desired metrics, it can be called with predictions and targets 
    to compute and return the metric values in a dictionary format. Metrics can be reset to 
    their initial state using the `reset` method.

    Usage example:

    >>> class MeanAbsoluteError:
    ...     def __init__(self):
    ...         self.reset_state()
    ...
    ...     def update_state(self, preds, targets):
    ...         pass  # Typically, updates some internal state here
    ...
    ...     def compute(self, preds, targets):
    ...         return sum(abs(p - t) for p, t in zip(preds, targets)) / len(preds)
    ...
    ...     def reset_state(self):
    ...         pass  # Reset any internal state to initial conditions
    ...
    >>> metrics = {'mean_absolute_error': MeanAbsoluteError()}
    >>> evaluator = Evaluator(metrics)
    >>> evaluator([3, 5], [4, 6])
    {'mean_absolute_error': 1.0}
    >>> evaluator.reset()  # Resets the internal state of all metrics

    Attributes:
        metrics (Dict[str, Callable]): Dictionary mapping metric names to their respective metric objects.
    """

    def __init__(self, metrics: str, threshold: float = 0.5) -> None:
        """
        Initialize the evaluator with desired metric objects.

        Args:
            metrics (Dict[str, Callable]): Dictionary of metric name to metric objects.
        """
        self.metrics = {metrics: MetricFactory.create(
            metrics) for metrics in metrics}

        self.threshold = threshold

    def __call__(self, prediction: List, target: List) -> Dict[str, float]:
        """Compute and update the desired metrics on the given predictions and targets.

        Args:
            prediction (List): Model predictions.
            target (List): Ground truth targets.

        Returns:
            Dict[str, float]: A dictionary containing metric names and their computed scores.
        """
        results = {}
        for name, metric in self.metrics.items():
            prediction = create_suitable_vector(prediction)
            target = create_suitable_vector(target)
            try:
                metric.update_state(prediction, target)
                results[name] = metric.compute(prediction, target)
            except:
                try:
                    prediction = (prediction > self.threshold).astype(int)
                    metric.update_state(prediction, target)
                    results[name] = metric.compute(prediction, target)
                except ValueError:
                    results[name] = np.nan

        return results

    def reset(self) -> None:
        """
        Reset the state of all metrics.
        """
        for metric in self.metrics.values():
            metric.reset_state()
