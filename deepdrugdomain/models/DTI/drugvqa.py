from functools import partial
from typing import Optional, Sequence

from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory

from ..factory import ModelFactory
import torch
from torch import nn
import torch.nn.functional as F
from deepdrugdomain.utils.weight_init import trunc_normal_


@ModelFactory.register('drugvqa')
class DrugVQA(nn.Module):
    def __init__(self,
                 head_dropout_rate: float,
                 head_activation_fn: Optional[str],
                 head_normalization: Optional[str],
                 head_dims: Sequence[int]):
        super().__init__()
        # Prediction layer
        self.head = nn.ModuleList()
        neuron_list = [self.embedding_dim] + list(head_dims)
        for item in range(len(neuron_list) - 1):
            self.head.append(nn.Dropout(head_dropout_rate))
            self.head.append(
                nn.Linear(neuron_list[item], neuron_list[item + 1]))
            self.head.append(ActivationFactory.create(
                head_activation_fn) if head_activation_fn else nn.Identity())

        trunc_normal_(self.latent_query, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.head(x)
