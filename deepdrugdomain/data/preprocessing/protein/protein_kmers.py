from collections import defaultdict
import os
from typing import Optional
from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
import numpy as np
import torch


def split_sequence(sequence, ngram, word_dict):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


@PreprocessorFactory.register("kmers")
class KmersProteinPreprocessor(BasePreprocessor):
    def __init__(self, ngram: int, word_dict: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.word_dict = word_dict
        self.ngram = ngram

    def preprocess(self, data: str) -> torch.Tensor:
        self.word_dict = self.word_dict if self.word_dict is not None else defaultdict(
            lambda: len(self.word_dict))
        words = split_sequence(data, self.ngram, self.word_dict)
        return torch.tensor(words, dtype=torch.long)
