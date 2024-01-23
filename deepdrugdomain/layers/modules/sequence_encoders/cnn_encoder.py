from typing import Sequence, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from ...utils import LayerFactory, ActivationFactory


class CNNEncoder(nn.Module):
    """
    CNN encoder module for encoding sequences of tokens.
    """

    def __init__(self,
                 input_channels: int,
                 hidden_channels: Sequence[int],
                 output_channels: int,
                 kernel_sizes: Sequence[int],
                 strides: Sequence[int],
                 pooling: Optional[Sequence[Optional[str]]] = None,
                 pooling_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
                 paddings: Union[Sequence[int], int] = 0,
                 activations: Optional[Union[str,
                                             Sequence[Optional[str]]]] = None,
                 dropouts: Union[float, Sequence[float]] = 0.0,
                 normalization: Optional[Union[str,
                                               Sequence[Optional[str]]]] = None,
                 input_embedding_dim: Optional[int] = None,
                 permute_embedding_indices: Optional[Sequence[int]] = None,
                 **kwargs) -> None:
        """
        Initialize the CNN encoder.
        """
        super().__init__()

        self.layer_factory = LayerFactory()
        self.activation_factory = ActivationFactory()
        self.input_channels = input_channels

        layers = []
        cnn_channels = [input_channels] + hidden_channels + [output_channels]
        if pooling_kwargs is None:
            pooling_kwargs = [{}] * (len(cnn_channels) - 1)

        if isinstance(pooling_kwargs, dict):
            pooling_kwargs = [pooling_kwargs] * (len(cnn_channels) - 1)

        if isinstance(dropouts, float):
            dropouts = [dropouts] * (len(cnn_channels) - 1)
        if isinstance(normalization, str):
            normalization = [normalization] * (len(cnn_channels) - 1)
        if normalization is None:
            normalization = [None] * (len(cnn_channels) - 1)
        if isinstance(activations, str):
            activations = [activations] * (len(cnn_channels) - 1)
        if activations is None:
            activations = [None] * (len(cnn_channels) - 1)

        if isinstance(pooling, str):
            pooling = [pooling] * (len(cnn_channels) - 1)

        if pooling is None:
            pooling = [None] * (len(cnn_channels) - 1)

        if not isinstance(paddings, list):
            paddings = [paddings] * (len(cnn_channels) - 1)

        # print(len(cnn_channels) - 1, len(kernel_sizes), len(strides), len(dropouts), len(normalization), len(
        #     activations), len(pooling), len(pooling_kwargs))

        assert len(cnn_channels) - 1 == len(kernel_sizes) == len(strides) == len(dropouts) == len(normalization) == len(
            activations) == len(pooling) == len(pooling_kwargs), "The number of CNN layers parameters must be the same"

        if input_embedding_dim is not None:
            layers.append(nn.Embedding(input_embedding_dim, self.input_channels))
            if permute_embedding_indices is not None:
                layers.append(LayerFactory.create("permute", permute_embedding_indices))

        for i in range(len(cnn_channels) - 1):
            layers.append(
                nn.Conv1d(in_channels=cnn_channels[i],
                          out_channels=cnn_channels[i + 1],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=self._get_padding(paddings, i),
                          **kwargs)
            )

            if normalization[i]:
                layers.append(
                    self.layer_factory.create(normalization[i], cnn_channels[i + 1]))

            if activations[i]:
                layers.append(
                    self.activation_factory.create(activations[i]))
            if pooling[i]:
                if pooling[i] == 'max':
                    layers.append(nn.MaxPool1d(**pooling_kwargs[i]))
                elif pooling[i] == 'avg':
                    layers.append(nn.AvgPool1d(**pooling_kwargs[i]))
                elif pooling[i] == 'global_max':
                    layers.append(nn.AdaptiveAvgPool1d(1))
                elif pooling[i] == 'global_avg':
                    layers.append(nn.AdaptiveAvgPool1d(1))
                else:
                    raise ValueError(
                        f"Pooling type '{pooling[i]}' not supported")

        self.cnn_encoder = nn.Sequential(*layers)

    def _get_padding(self, paddings, index):
        return paddings if isinstance(paddings, str) else paddings[index] if isinstance(paddings, Sequence) else paddings

    def _get_sequence_value(self, sequence, index):
        return sequence[index] if isinstance(sequence, Sequence) else sequence

    def forward(self, x):
        """
        Forward pass of the module.
        """
        return self.cnn_encoder(x)

    def get_output_size(self, input_size):
        """
        Returns the output size of the module.
        """
        return self.cnn_encoder(torch.zeros((1, self.input_channels, input_size))).view(1, -1).shape[1]
