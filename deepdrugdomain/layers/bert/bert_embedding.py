"""
This module provides an embedding layer class for BERT, which is a transformer-based model
for natural language processing pre-training introduced by Devlin et al. in their 2018 paper.
The embedding layer combines token embeddings with position embeddings and optional segment
embeddings, followed by layer normalization.

The `Embedding` class defined here is used as a part of the BERT model to generate input
representations which are further processed by the transformer encoder layers.

Example:
    >>> import torch
    >>> from embedding_module import Embedding  # Replace with actual module path
    >>> vocab_size, dim, max_len, n_segments = 30522, 768, 512, 2
    >>> embedding_layer = Embedding(vocab_size, dim, max_len, n_segments)
    >>> input_ids = torch.randint(0, vocab_size, (1, max_len))  # Example input tensor
    >>> segment_ids = torch.randint(0, n_segments, (1, max_len))  # Example segment tensor
    >>> embeddings = embedding_layer(input_ids, segment_ids)
    >>> print(embeddings.shape)  # Expected shape: (1, max_len, dim)

Citation:
    @article{devlin2018bert,
        title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
        author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
        journal={arXiv preprint arXiv:1810.04805},
        year={2018}
    }
"""

from typing import Optional
from torch import nn, Tensor
import torch
from ..utils import LayerFactory


@LayerFactory.register('bert_embedding')
class Embedding(nn.Module):
    """
    A module for BERT embeddings which combines token embeddings, position embeddings, 
    and optional segment embeddings with layer normalization.

    Attributes:
        tok_embed (nn.Embedding): Token embedding layer.
        pos_embed (nn.Embedding): Position embedding layer.
        seg_embed (Optional[nn.Embedding]): Segment embedding layer, used if `n_segments` is greater than 0.
        norm (nn.Module): Normalization layer, either a specific layer norm or an identity mapping.

    Args:
        vocab_size (int): Size of the vocabulary.
        dim (int): Dimensionality of the embeddings.
        max_len (int): Maximum sequence length for position embeddings.
        n_segments (int): Number of segments for segment embeddings. If 0, no segment embeddings are used.
        layer_norm (str, optional): The type of layer normalization to use. Defaults to "layer_norm".
            If None, an identity mapping is used instead.

    """

    def __init__(self, vocab_size: int, dim: int, max_len: int, n_segments: int, layer_norm: Optional[str] = "layer_norm") -> None:
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)  # Token embedding
        self.pos_embed = nn.Embedding(max_len, dim)  # Position embedding
        # Segment embedding (optional)
        self.seg_embed = nn.Embedding(
            n_segments, dim) if n_segments > 0 else None
        self.norm = LayerFactory.create(
            layer_norm, dim) if layer_norm else nn.Identity()  # Layer normalization

    def forward(self, x: Tensor, seg: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the Embedding module.

        Args:
            x (Tensor): Input token indices with shape (batch_size, seq_len).
            seg (Optional[Tensor]): Optional segment indices with shape (batch_size, seq_len). Used if `n_segments` > 0.

        Returns:
            Tensor: Combined embeddings with the same shape as input `x`, with added dimensionality for embedding.
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device,
                           dtype=torch.long)  # Position indices
        # Expand position indices to match the batch size of `x`
        pos = pos.unsqueeze(0).expand_as(x)

        # Compute segment embeddings if `seg` is provided; otherwise, use a zero tensor
        seg_embed = self.seg_embed(
            seg) if self.seg_embed is not None and seg is not None else 0

        # Sum token, position, and segment embeddings and apply normalization
        embedding = self.tok_embed(x) + self.pos_embed(pos) + seg_embed
        return self.norm(embedding)
