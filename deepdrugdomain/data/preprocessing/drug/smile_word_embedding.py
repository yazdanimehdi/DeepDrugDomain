from collections import defaultdict
import os
import re
from typing import Dict, List, Optional
from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
import numpy as np
import torch
import deepsmiles


def split_sequence(sequence, ngram, stride=1):
    words = [sequence[i:i+ngram]
             for i in range(0, len(sequence)-ngram+1, stride)]
    return words


def tokenize_smiles(smiles: str, regex: str, replacement_dict: Dict) -> List[str]:
    for original, replacement in replacement_dict.items():
        smiles = smiles.replace(original, replacement)
    tokens = re.findall(regex, smiles)
    return tokens


@PreprocessorFactory.register("smiles_to_kword_embedding", "smile", "kword_encoding_tensor")
class SmilesWordEmbeddingPreprocessor(BasePreprocessor):
    def __init__(self, regex: str = None, replacement_dict: Dict = {},  window: int = 8, stride: int = 8, convert_deepsmiles: bool = False, one_hot: bool = False, max_length: Optional[int] = None, num_of_combinations: Optional[int] = None, word_dict: Optional[dict] = None, rings: Optional[bool] = True, branches: Optional[bool] = True, **kwargs):
        super().__init__(**kwargs)
        self.regex = regex
        self.replacement_dict = replacement_dict
        self.k = window
        self.word_dict = word_dict
        self.stride = stride
        self.deepsmiles = convert_deepsmiles
        self.one_hot = one_hot
        self.converter = deepsmiles.Converter(rings=rings, branches=branches)
        if one_hot:
            assert num_of_combinations , "Must specify num_of_combinations and max_length if one_hot is True"
        self.num_of_combinations = num_of_combinations
        self.max_length = max_length

    def preprocess(self, data: str) -> torch.Tensor:
        self.word_dict = self.word_dict if self.word_dict is not None else defaultdict(
            lambda: len(self.word_dict))

        if self.deepsmiles:
            data = self.converter.encode(data)

        if self.regex is None:
            words = split_sequence(data, self.k, self.stride)
        else:
            words = tokenize_smiles(data, self.regex, self.replacement_dict)

        sequence_length = len(words)
        if self.max_length is not None:
            if self.max_length - len(words) > 0:
                sequence_length = self.max_length
            else:
                return None

        if self.one_hot:
            encoded = np.zeros((sequence_length, self.num_of_combinations))
            for i, word in enumerate(words):
                if self.word_dict[word] >= self.num_of_combinations:
                    return None
                try:
                    encoded[i, self.word_dict[word]] = 1
                except KeyError:
                    return None
            encoded = torch.tensor(encoded, dtype=torch.float32)

        else:
            encoded = np.zeros((sequence_length))
            for i, word in enumerate(words):
                if self.num_of_combinations:
                    if self.word_dict[word] >= self.num_of_combinations:
                        return None
                try:
                    encoded[i] = self.word_dict[word]
                except KeyError:
                    return None

            encoded = torch.tensor(encoded, dtype=torch.long)

        return encoded
