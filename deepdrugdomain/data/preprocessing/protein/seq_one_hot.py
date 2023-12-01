from typing import Optional
import torch
import numpy as np
from ..base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
from torch.nn import functional as F


# Standard amino acids in the order they are commonly listed
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@PreprocessorFactory.register("sequence_to_one_hot", "protein_sequence", "encoding_tensor")
class OneHotEncoderPreprocessor(BasePreprocessor):
    def __init__(self, amino_acids: str = AMINO_ACIDS, max_sequence_length: Optional[int] = None, one_hot: bool = True, **kwargs) -> None:
        """
        Initializes the OneHotEncoderPreprocessor.
        parameters:
            max_sequence_length (Optional[int]): The desired fixed size for the sequence after padding.
            **kwargs: Additional keyword arguments to be passed to the BasePreprocessor class.
        """
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.one_hot = one_hot
        self.amino_acids = amino_acids

    def preprocess(self, sequence: str) -> Optional[torch.Tensor]:
        """
        Generates a one-hot encoded matrix for the given amino acid sequence.

        Parameters:
            sequence (str): A string representing the amino acid sequence of a protein.

        Returns:
            Optional[torch.Tensor]: A tensor representing the one-hot encoded sequence
                                    of the protein. Each row corresponds to an amino acid
                                    and each column corresponds to one of the 20 standard
                                    amino acids. Returns None if the sequence contains
                                    non-standard amino acids.

        Raises:
            ValueError: If the sequence contains non-standard amino acids.
        """
        sequence_length = len(sequence)
        if self.one_hot:
            one_hot_matrix = np.zeros(
                (sequence_length, len(self.amino_acids)), dtype=np.float32)
            for i, amino_acid in enumerate(sequence):
                if amino_acid not in self.amino_acids:
                    raise ValueError(
                        f"Non-standard amino acid found: {amino_acid}")
                position = self.amino_acids.index(amino_acid)
                one_hot_matrix[i, position] = 1
            one_hot_tensor = torch.from_numpy(one_hot_matrix)
            if self.max_sequence_length:
                # Calculate how much padding is needed
                padding_needed = self.max_sequence_length - sequence_length
                # Pad the tensor if needed, pad is a tuple (pad_left, pad_right, pad_top, pad_bottom)
                one_hot_tensor = F.pad(
                    one_hot_tensor, (0, 0, 0, padding_needed), 'constant', 0)
        else:
            one_hot_matrix = np.zeros(sequence_length, dtype=np.float32)
            if self.max_sequence_length:
                if sequence_length > self.max_sequence_length:
                    return None
                one_hot_matrix = np.zeros(
                    self.max_sequence_length, dtype=np.float32)
                # Calculate how much padding is needed
                padding_needed = self.max_sequence_length - sequence_length
                # Pad the tensor if needed, pad is a tuple (pad_left, pad_right, pad_top, pad_bottom)
                sequence = list(sequence) + ['<PAD>'] * padding_needed
            for i, amino_acid in enumerate(sequence):
                if amino_acid == '<PAD>':
                    position = 0
                elif amino_acid not in self.amino_acids:
                    return None
                else:
                    position = self.amino_acids.index(amino_acid) + 1

                one_hot_matrix[i] = position
            one_hot_tensor = torch.from_numpy(one_hot_matrix)
            one_hot_tensor = one_hot_tensor.long()

        return one_hot_tensor
