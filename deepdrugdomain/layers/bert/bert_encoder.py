"""
This module defines a BERT encoder class that encapsulates the embedding layer and multiple
encoder layers from the transformer architecture. The encoder generates a context-aware
representation of the input sequence, which can be used for various downstream NLP tasks.

The `BERTEncoder` class uses a customizable embedding layer and a stack of transformer encoder
layers to process the input sequence and produce the final encoded representation.

Attributes:
    embedding (nn.Module): The embedding layer created by the LayerFactory based on the provided arguments.
    layers (nn.ModuleList): A list of transformer encoder layers created by the LayerFactory.
    fc (nn.Linear): A fully connected layer that projects the output to the desired hidden dimension.

Args:
    dim (int): The dimensionality of the input embeddings and the output of each encoder layer.
    hidden_dim (int): The dimensionality of the final output vectors.
    n_word (int): The number of words for which embeddings will be created.
    max_len (int): The maximum length of the input sequences.
    depth (int): The number of transformer encoder layers to be stacked.
    embedding_layer (str): The name of the embedding layer to be created by the LayerFactory.
    embedding_layer_args (dict): A dictionary of arguments required for creating the embedding layer.
    encoder_layer (str): The name of the encoder layer to be created by the LayerFactory.
    encoder_layer_args (dict): A dictionary of arguments required for creating the encoder layer.

Example:
    >>> input_ids = torch.randint(0, 1000, (1, 512))  # Example input tensor with random word IDs
    >>> encoder = BERTEncoder()
    >>> output = encoder(input_ids)
    >>> print(output.shape)  # Expected shape: (1, 512, 32), assuming default hidden_dim of 32

Note:
    The `get_attn_pad_mask` function is used internally to create a mask for padding tokens
    to exclude them from attention calculations.
"""

from torch import nn, Tensor
from ..utils import LayerFactory
from .utils import get_attn_pad_mask


@LayerFactory.register('bert_encoder')
class BERTEncoder(nn.Module):
    def __init__(self,
                 dim: int = 32,
                 hidden_dim: int = 32,
                 depth: int = 3,
                 embedding_layer: str = "bert_embedding",
                 embedding_layer_args: dict = {
                     "vocab_size": 1000, "dim": 32, "max_len": 8112, "n_segments": 0},
                 encoder_layer: str = "transformer_attention_block",
                 encoder_layer_args: dict = {"dim": 32, "num_heads": 8}) -> None:
        super(BERTEncoder, self).__init__()
        self.embedding = LayerFactory.create(
            embedding_layer, **embedding_layer_args)
        self.layers = nn.ModuleList([LayerFactory.create(
            encoder_layer, **encoder_layer_args) for _ in range(depth)])
        self.fc = nn.Linear(dim, hidden_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass for the BERT encoder.

        Args:
            input_ids (Tensor): Indices of input sequence tokens in the vocabulary.

        Returns:
            Tensor: The encoded output from the BERT encoder with shape (batch_size, seq_len, hidden_dim).
        """
        output = self.embedding(input_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        output = self.fc(output)
        return output
