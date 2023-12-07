from collections import defaultdict
import os
from typing import Optional
from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
import numpy as np
import torch


def split_sequence(sequence, ngram, stride=1):
    words = [sequence[i:i+ngram]
             for i in range(0, len(sequence)-ngram+1, stride)]
    return np.array(words)


@PreprocessorFactory.register("kmers", "protein_sequence", "kmers_encoded_tensor")
class KmersProteinPreprocessor(BasePreprocessor):
    def __init__(self, window: int, stride: int = 1, one_hot: bool = False, word_dict: Optional[dict] = None, number_of_combinations: Optional[int] = None, max_length: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.word_dict = word_dict
        self.ngram = window
        self.stride = stride
        self.max_length = max_length
        self.one_hot = one_hot
        self.number_of_combinations = number_of_combinations
        if one_hot:
            assert number_of_combinations is not None and max_length is not None, "If one_hot is True, number_of_combinations and max_length must be specified."

    def preprocess(self, data: str) -> torch.Tensor:
        self.word_dict = self.word_dict if self.word_dict is not None else defaultdict(
            lambda: len(self.word_dict))
        words = split_sequence(data, self.ngram, self.stride)
        sequence_length = len(words)
        if self.max_length is not None:
            if self.max_length - len(words) > 0:
                sequence_length = self.max_length
            else:
                return None

        if self.one_hot:
            encoded = np.zeros(
                (sequence_length, self.number_of_combinations), dtype=np.float32)
            for i, word in enumerate(words):
                if self.word_dict[word] >= self.number_of_combinations:
                    return None
                try:
                    encoded[i, self.word_dict[word]] = 1
                except KeyError:
                    return None

            encoded = torch.tensor(encoded, dtype=torch.float32)
        else:
            encoded = np.zeros((sequence_length), dtype=np.float32)
            for i, word in enumerate(words):
                if self.number_of_combinations:
                    if self.word_dict[word] >= self.number_of_combinations:
                        return None
                try:
                    encoded[i] = self.word_dict[word]
                except KeyError:
                    return None
            encoded = torch.tensor(encoded, dtype=torch.long)

        return encoded
