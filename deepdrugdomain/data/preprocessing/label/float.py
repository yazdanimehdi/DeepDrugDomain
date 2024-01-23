from collections import defaultdict
import os
from typing import Optional
from deepdrugdomain.data.preprocessing.base_preprocessor import BasePreprocessor
from ..factory import PreprocessorFactory
import numpy as np
import torch


@PreprocessorFactory.register("log", "tensor", "log_tensor")
class InteractionToBinary(BasePreprocessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, data: str) -> torch.Tensor:
        return torch.tensor([np.log10(data/1e9)])
