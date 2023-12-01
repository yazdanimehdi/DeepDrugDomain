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
                 activations: Optional[Union[str, Sequence[Optional[str]]]] = None, 
                 dropouts: Union[float, Sequence[float]] = 0.0, 
                 normalization: Optional[Union[str, Sequence[Optional[str]]]] = None, 
                 **kwargs) -> None:
        """
        Initialize the CNN encoder.
        """
        super().__init__()

        self.layer_factory = LayerFactory()
        self.activation_factory = ActivationFactory()

        layers = []
        cnn_channels = [input_channels] + hidden_channels + [output_channels]

        if len(cnn_channels) - 1 != len(kernel_sizes) or len(cnn_channels) - 1 != len(strides):
            raise ValueError("Number of channels must be one more than the number of kernel sizes and strides")

        for i in range(len(cnn_channels) - 1):
            layers.append(
                nn.Conv1d(in_channels=cnn_channels[i], 
                          out_channels=cnn_channels[i + 1],  
                          kernel_size=kernel_sizes[i], 
                          stride=strides[i], 
                          padding=self._get_padding(paddings, i), 
                          **kwargs)
            )
            self._add_optional_layer(layers, normalization, i, self.layer_factory)
            self._add_optional_layer(layers, activations, i, self.activation_factory)
            layers.append(nn.Dropout(self._get_sequence_value(dropouts, i)))

            if pooling:
                self._add_pooling_layer(layers, pooling[i], pooling_kwargs, i)

        self.cnn_encoder = nn.Sequential(*layers)

    def _get_padding(self, paddings, index):
        return paddings[index] if isinstance(paddings, Sequence) else paddings

    def _get_sequence_value(self, sequence, index):
        return sequence[index] if isinstance(sequence, Sequence) else sequence

    def _add_optional_layer(self, layers, layer_type, index, factory):
        if layer_type:
            layer = factory.create(layer_type[index] if isinstance(layer_type, Sequence) else layer_type)
            if layer:
                layers.append(layer)

    def _add_pooling_layer(self, layers, pooling_type, pooling_kwargs, index):
        if pooling_type == 'max':
            layers.append(nn.MaxPool1d(**pooling_kwargs[index]))
        elif pooling_type == 'avg':
            layers.append(nn.AvgPool1d(**pooling_kwargs[index]))

    def forward(self, x):
        """
        Forward pass of the module.
        """
        return self.cnn_encoder(x)
