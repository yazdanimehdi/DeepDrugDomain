"""
binding_smile_graph_collate.py - Provides collation for drug-target interaction (DTI)
and drug-target affinity (DTA) models.

The `BindingSiteSmilesGraphCollate` class provides a collate function to handle batches
of data points for models that require separate graph representations for proteins
and drugs. The collate function can be especially useful for models that deal with
protein binding sites, like FragXsiteDTI and AttentionSiteDTI, and need to process
the protein and drug graphs individually.

Example Usage:
>>> from deepdrugdomain.data.collate import CollateFactory
>>> collate_fn = CollateFactory.create("binding_graph_smile_graph")
>>> data_loader = DataLoader(dataset, collate_fn=collate_fn)
>>> for protein_graphs, drug_graphs, targets in data_loader:
...     # Model processing logic here
"""

import torch
from typing import List, Tuple, Any
from ..base_collate import BaseCollate
from ..factory import CollateFactory


@CollateFactory.register("binding_graph_smile_graph")
class BindingSiteSmilesGraphCollate(BaseCollate):
    def __call__(self, batch: List[Tuple[Any, Any, torch.Tensor]]) -> Tuple[Tuple[List[Any], List[Any]], torch.Tensor]:
        """
        Collate function for handling batches of protein and drug graph pairs.

        This method unpacks the input batch, separates the protein graphs, drug graphs,
        and the target values, and then repacks them into the desired format.

        Parameters:
        - batch (List[Tuple[Graph, Graph, Tensor]]): List of tuples, where each tuple
          contains a protein graph, a drug graph, and a target tensor.

        Returns:
        - Tuple[Tuple[List[Graph], List[Graph]], Tensor]: A tuple containing two
          elements: A tuple of protein graphs and drug graphs lists, and a tensor
          of targets.
        """

        # Unpacking the batch data
        protein_graphs, drug_graphs, targets = zip(*batch)

        # Stacking target tensors for batch processing
        batched_targets = torch.stack(targets, 0)

        return protein_graphs, drug_graphs, batched_targets
