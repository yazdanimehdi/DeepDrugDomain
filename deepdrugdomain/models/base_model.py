from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from torch.nn import Module


class BaseModel(ABC, Module):
    """
    Base class for all models in the DeepDrugDomain project.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        None

    Methods:
        forward: Forward pass of the model.
        train_one_epoch: Train the model for one epoch.
        evaluate: Evaluate the model on a given dataset.
        predict: Make predictions using the model.
        save_checkpoint: Save the model checkpoint to a file.
        load_checkpoint: Load the model checkpoint from a file.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The output of the forward pass.
        """
        pass

    @abstractmethod
    def train_one_epoch(self, *args, **kwargs) -> Any:
        """
        Train the model for one epoch.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The training result.
        """
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """
        Evaluate the model on a given dataset.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The evaluation result.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """
        Make predictions using the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The predictions.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs) -> None:
        """
        Save the model checkpoint to a file.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs) -> None:
        """
        Load the model checkpoint from a file.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def reset_head(self, *args, **kwargs) -> None:
        """
        Reset the model head.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def collate(self, *args, **kwargs) -> Any:
        """
        Collate the data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The collated data.
        """
        pass
