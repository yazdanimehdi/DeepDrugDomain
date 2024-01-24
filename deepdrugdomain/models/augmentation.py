from typing import Any, Type, Union, Optional, List, Dict, Callable
import warnings
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from deepdrugdomain.data.preprocessing import PreprocessingObject
from deepdrugdomain.data.preprocessing.utils.preprocessing_data_struct import PreprocessingList
from deepdrugdomain.layers import LinearHead
from deepdrugdomain.schedulers import BaseScheduler
from .interaction_model import BaseInteractionModel
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from .factory import ModelFactory


class AugmentedModel(BaseInteractionModel):
    def __init__(self, original_model_class, added_dim, encoders, preprocessors, custom_collate: Optional[Callable] = None, **kwargs):
        kwargs_new = kwargs.copy()
        kwargs_new["remove_head"] = True
        original_model = original_model_class(**kwargs_new)
        head_kwargs = original_model.head_kwargs
        head_kwargs["input_size"] = original_model.head_kwargs["input_size"] + added_dim
        super().__init__(
            embedding_dim=None,
            encoders=encoders,
            head_kwargs=head_kwargs,
            aggregation_method='concat',
        )

        self.original_model = original_model
        self.preprocessors = preprocessors
        self.custom_collate = custom_collate

    def forward(self, augmented_inputs: List[torch.Tensor], *args):
        original_output = self.original_model.forward(*args)

        encoded_augmented_inputs = [self.encoders[i](
            augmented_inputs[i]) for i in range(len(augmented_inputs))]
        encoded = encoded_augmented_inputs + [original_output]
        concatenated_output = self.aggregate(encoded)
        return self.head(concatenated_output)

    def default_preprocess(self, *args):
        original_default_preprocess = self.original_model.default_preprocess(
            *args)
        return self.preprocessors + original_default_preprocess

    def collate(self, batch):
        if self.custom_collate is not None:
            return self.custom_collate(batch)

        # Unpacking the batch data
        *inputs, targets = list(zip(*batch))
        a = [[] for i in range(len(inputs))]
        for i in range(len(inputs[0])):
            for j in range(len(inputs)):
                a[j].append(inputs[j][i])

        targets = torch.stack(targets, 0)
        batch = a + [targets]
        return batch


# def extend_method_output(original_method, additional_elements):
#     def new_method(*args):
#         # Call the original method and get its output
#         original_output = original_method(*args)
#         # Add additional elements to the output
#         return additional_elements + original_output
#     return new_method


# class AugmentedInteractionModel:

#     def __init__(self, augmented_Dicts: List[Dict[str, Any]]) -> None:
#         self.encoders = [augmented_Dict['encoder']
#                          for augmented_Dict in augmented_Dicts]
#         self.preprocessors = [augmented_Dict['preprocessor']
#                               for augmented_Dict in augmented_Dicts]
#         self.added_dim = sum([augmented_Dict['output_dim']
#                              for augmented_Dict in augmented_Dicts])

#     def augment(self, original_model_class, **kwargs):
#         assert issubclass(original_model_class, BaseInteractionModel)
#         original_model_class = original_model_class(remove_head=True, **kwargs)
#         original_model_class.head_kwargs["input_size"] += self.added_dim
#         original_model_class.encoders = self.encoders + original_model_class.encoders
#         self.head = LinearHead(**original_model_class.head_kwargs)

#         original_model_class.default_preprocess = extend_method_output(
#             original_model_class.default_preprocess,  self.preprocessors)

#         return original_model_class


class AugmentedModelFactory(ModelFactory):
    def __init__(self, augmented_Dicts: List[Dict[str, Any]]) -> None:
        self.encoders = [augmented_Dict['encoder']
                         for augmented_Dict in augmented_Dicts]
        self.preprocessors = [augmented_Dict['preprocessor']
                              for augmented_Dict in augmented_Dicts]
        self.added_dim = sum([augmented_Dict['output_dim']
                              for augmented_Dict in augmented_Dicts])

    def create(cls, key: str, version: Optional[str] = None, **kwargs) -> BaseInteractionModel:
        if key not in cls._registry:
            raise ValueError(f"Model '{key}' not registered.")

        model_key = f"{key}_{version}" if version and f"{key}_{version}" in cls._registry else key
        if version and f"{key}_{version}" not in cls._registry:
            warnings.warn(
                f"version '{version}' configuration not found for model '{key}'. Using default configuration.", UserWarning)

        # Use default configuration as a base; override with version-specific if exists
        default_config = cls._default_args.get(key, {})
        version_config = cls._default_args.get(model_key, {})
        combined_config = {**default_config, **
                           version_config, **kwargs}  # Merge configurations

        model_class = AugmentedModel(
            cls._registry[model_key], cls.added_dim, cls.encoders, cls.preprocessors, **combined_config)
        # Use base model class registry for instantiation
        return model_class
