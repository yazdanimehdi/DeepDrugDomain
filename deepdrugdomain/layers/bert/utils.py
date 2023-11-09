"""
This function creates an attention pad mask for sequences with padding tokens. It is used in the attention mechanism
in models like BERT to prevent the model from attending to padding tokens. This mask is applied to the attention scores
before the softmax step to set the attention scores for padding tokens to a very large negative value, effectively zeroing
them in the softmax output.

Args:
    seq (Tensor): The input sequence tensor with shape (batch_size, seq_len) where padded positions are filled with 0.

Returns:
    Tensor: An attention pad mask with shape (batch_size, seq_len, seq_len) with 1s in positions corresponding to pads
    and 0s elsewhere.

Example:
    >>> import torch
    >>> seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])  # Example sequence tensor with padding
    >>> attn_pad_mask = get_attn_pad_mask(seq)
    >>> print(attn_pad_mask)
    tensor([[[0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1]],
            [[0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 0, 1, 1]]], dtype=torch.uint8)
"""
from torch import Tensor


def get_attn_pad_mask(seq: Tensor) -> Tensor:
    batch_size, seq_len = seq.size()
    # Create mask for padding tokens
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    # Expand mask for each sequence length (duplicated across the seq_len dimension)
    pad_attn_mask_expand = pad_attn_mask.expand(
        batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand
