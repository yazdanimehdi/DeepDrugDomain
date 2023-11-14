import torch
from typing import List, Tuple, Any
from ..base_collate import BaseCollate
from ..factory import CollateFactory
import dgl


@CollateFactory.register("ammvf_collate")
class AMMVFCollate(BaseCollate):
    def __call__(self, batch: List[Tuple[Any, Any, torch.Tensor]]) -> Tuple[Tuple[List[Any], List[Any]], torch.Tensor]:

        # Unpacking the batch data
        protein_1, protein_2, drug_graphs, drug_fingerprint, targets = zip(
            *batch)
        return (protein_1, protein_2, drug_fingerprint, drug_graphs), targets
