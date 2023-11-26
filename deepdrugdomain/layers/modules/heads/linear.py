import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional, Union
import torch
from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory


class LinearHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: Sequence[int], activations: Optional[Union[Sequence[str], str]] = None, dropouts: Union[Sequence[float], float] = 0.0, normalization:  Optional[Union[Sequence[str], str]] = None) -> None:
        super().__init__()
        dims = [input_size] + list(hidden_sizes) + [output_size]
        activations = activations if isinstance(activations, list) else [
            activations]*(len(dims) - 1)
        dropouts = dropouts if isinstance(dropouts, list) else [
            dropouts]*(len(dims) - 1)
        normalization = normalization if isinstance(normalization, list) else [
            normalization]*(len(dims) - 1)
        assert len(dims) - 1 == len(dropouts) == len(normalization) == len(
            activations), "The number of linear layers parameters must be the same"

        self.head = []
        for i in range(len(dims) - 1):
            self.head.append(nn.Dropout(dropouts[i]))
            self.head.append(LayerFactory.create(
                normalization[i], dims[i]) if normalization[i] else nn.Identity())
            self.head.append(nn.Linear(dims[i], dims[i + 1]))
            self.head.append(ActivationFactory.create(
                activations[i]) if activations[i] else nn.Identity())

        self.head = nn.Sequential(*self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
