from torch import nn
from typing import Sequence, Optional, Union, Dict, Any, List
from deepdrugdomain.layers import LayerFactory


@LayerFactory.register('reshape')
class Reshape(nn.Module):
    """
    Reshape module for reshaping tensors.
    """

    def __init__(self, shape: Sequence[int]) -> None:
        """
        Initialize the reshape module.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x):
        """
        Forward pass of the module.
        """
        return x.view(self.shape)


@LayerFactory.register('permute')
class Permute(nn.Module):
    """
    Reshape module for reshaping tensors.
    """

    def __init__(self, permute: Sequence[int]) -> None:
        """
        Initialize the reshape module.
        """
        super().__init__()
        assert max(permute) < len(
            permute), "Permute indices must be less than the length of the tensor"
        self.permute = permute

    def forward(self, x):
        """
        Forward pass of the module.
        """
        return x.permute(self.permute)
