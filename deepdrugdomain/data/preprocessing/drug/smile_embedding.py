import re
import torch
import numpy as np
from typing import Any, Dict, List, Optional
from ..base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
from torch.nn import functional as F

# Define a dictionary for multi-character symbol replacements
REPLACEMENTS = {
    'Br': 'R',
    'Cl': 'L',
    'Si': 'A',
    'Se': 'Z'
}

# Define a tokenization pattern inspired by Olivecrona et al., 2017
TOKEN_REGEX = r'(\[[^\[\]]{1,6}\])'

CHARISOSMISET = {"<PAD>": 0, "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


@PreprocessorFactory.register('smiles_to_encoding', 'smile', 'encoding_tensor')
class SMILESToEncodingPreprocessor(BasePreprocessor):
    def __init__(self, one_hot: bool = False, embedding_dim: Optional[int] = None, max_sequence_length: Optional[int] = None, replacement_dict: Dict[str, str] = REPLACEMENTS, token_regex: Optional[str] = TOKEN_REGEX, from_set: Optional[Dict[str, int]] = CHARISOSMISET, **kwargs):
        """
        Initializes the SMILESToEmbeddingPreprocessor with an embedding dimension and optional max sequence length for padding.

        Parameters:
            embedding_dim (int): The dimensionality of the token embeddings.
            max_sequence_length (Optional[int]): The desired fixed size for the sequence after padding.
            replacement_dict (Dict[str, str]): A dictionary mapping multi-character symbols to single-character symbols.
            token_regex (str): A regular expression pattern for tokenizing SMILES strings.
            **kwargs: Additional keyword arguments to be passed to the BasePreprocessor class.
        """
        super().__init__(**kwargs)
        self.one_hot = one_hot
        self.embedding_dim = embedding_dim
        if one_hot:
            assert embedding_dim is not None, "Must specify embedding dimension if one-hot is True"

        self.max_sequence_length = max_sequence_length
        self.replacement_dict = replacement_dict
        self.token_regex = token_regex
        self.token_to_idx = None
        self.from_set = False
        if from_set is not None:
            self.from_set = True
            self.token_to_idx = from_set
        else:
            assert token_regex is not None, "Must specify token regex if all_chars is False"

    def tokenize_smiles(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string using a specified regular expression pattern.

        Parameters:
            smiles (str): The SMILES string to be tokenized.

        Returns:
            List[str]: A list of tokenized elements from the SMILES string.
        """
        for original, replacement in self.replacement_dict.items():
            smiles = smiles.replace(original, replacement)
        tokens = re.findall(self.token_regex, smiles)
        return tokens

    def data_preparations(self, data: Any) -> Any:
        """
        Prepare the data for the preprocessor by generating a dictionary mapping tokens to indices.
        parameters:
            data (Any): The data to be preprocessed.
        returns:
            Any: The preprocessed data.
        """
        if self.from_set:
            return data
        final_letters = []
        for s in data:
            tokens = self.tokenize_smiles(s)
            for token in tokens:
                if token not in final_letters:
                    final_letters.append(token)

        self.token_to_idx = {token: i + 1 for i,
                             token in enumerate(final_letters)}
        self.token_to_idx['<PAD>'] = 0
        return data

    def preprocess(self, smiles: str) -> Optional[torch.Tensor]:
        """
        Convert a SMILES string to a one-hot encoded 2D matrix of token embeddings.

        Parameters:
            smiles (str): The SMILES string of the drug molecule.

        Returns:
            Optional[torch.Tensor]: A 2D tensor representing the one-hot encoded token embeddings.
                                    Each row corresponds to a token, and each column corresponds to a
                                    potential token type. Returns None if the SMILES string contains
                                    tokens not covered by the embedding.
        """
        if not self.from_set:
            tokens = self.tokenize_smiles(smiles)
        else:
            tokens = list(smiles)

        sequence_length = len(tokens)
        if self.one_hot:
            one_hot_matrix = np.zeros(
                (sequence_length, self.embedding_dim), dtype=np.float32)

            for i, token in enumerate(tokens):
                token_idx = self.token_to_idx.get(token)
                if token_idx is None:
                    print(f"Unrecognized token in SMILES: {token}")
                    return None
                # Subtract 1 because indices start at 1
                one_hot_matrix[i, token_idx - 1] = 1

            one_hot_tensor = torch.from_numpy(one_hot_matrix)
            if self.max_sequence_length:
                # Calculate how much padding is needed
                padding_needed = self.max_sequence_length - sequence_length
                # Pad the tensor if needed, pad is a tuple (pad_left, pad_right, pad_top, pad_bottom)
                one_hot_tensor = F.pad(
                    one_hot_tensor, (0, 0, 0, padding_needed), 'constant', 0)
        else:
            if self.max_sequence_length:
                # Calculate how much padding is needed
                padding_needed = self.max_sequence_length - sequence_length
                if padding_needed < 0:
                    return None
                # Pad the tensor if needed, pad is a tuple (pad_left, pad_right, pad_top, pad_bottom)
                tokens = tokens + ['<PAD>'] * padding_needed

            max_length = self.max_sequence_length if self.max_sequence_length else sequence_length
            embedding_matrix = np.zeros(max_length, dtype=np.float32)
            for idx, item in enumerate(tokens):
                embedding_matrix[idx] = self.token_to_idx[item]

            one_hot_tensor = torch.from_numpy(embedding_matrix)

        return one_hot_tensor.to(torch.long)
