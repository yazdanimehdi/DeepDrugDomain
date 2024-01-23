from ...utils import LayerFactory
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Dict, Any, Optional, Union
from torch.autograd import Variable
import torch


class LSTMEncoder(nn.Module):
    """
        LSTM encoder module for encoding sequences of tokens.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool, batch_size: int = 1, **kwargs) -> None:
        """
            Initializes the LSTM encoder module.
            args:
                input_size (int): The input size of the module.
                output_size (int): The output size of the module.
                hidden_size (int): The hidden size of the module.
                num_layers (int): The number of layers of the module.
                dropout (float): The dropout rate of the module.
                bidirectional (bool): Whether to use a bidirectional LSTM.
                **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kwargs = kwargs
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional, **kwargs)
        self.batch_size = batch_size

    def forward(self, x: Any) -> Any:
        """
            Forward pass of the module.
            args:
                x (Any): The input sequence.
            returns:
                Any: The encoded sequence.
        """
        h0 = Variable(torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1), self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1), self.batch_size, self.hidden_size))
        output, (h0, c0) = self.lstm(x, (h0, c0))

        return output
