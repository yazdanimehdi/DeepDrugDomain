from collections import defaultdict
import os
from typing import Optional
from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
import numpy as np
import torch


@PreprocessorFactory.register("interaction_to_binary", "binary", "binary_tensor")
class InteractionToBinary(BasePreprocessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, data: str) -> torch.Tensor:
        return torch.tensor([int(data)])


@PreprocessorFactory.register("value_to_binary", "float", "binary_tensor")
class ValueToBinary(BasePreprocessor):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def preprocess(self, data: str) -> torch.Tensor:
        return torch.tensor([int(float(data) < self.threshold)])
