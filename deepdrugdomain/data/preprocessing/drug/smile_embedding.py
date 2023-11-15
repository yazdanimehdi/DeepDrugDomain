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


@PreprocessorFactory.register('smiles_to_embedding')
class SMILESToEmbeddingPreprocessor(BasePreprocessor):
    def __init__(self, embedding_dim: int = 128, max_sequence_length: Optional[int] = None, replacement_dict: Dict[str, str] = REPLACEMENTS, token_regex: str = TOKEN_REGEX, **kwargs):
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
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.replacement_dict = replacement_dict
        self.token_regex = token_regex
        self.token_to_idx = None

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

        tokens = self.tokenize_smiles(smiles)
        sequence_length = len(tokens)
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

        return one_hot_tensor
