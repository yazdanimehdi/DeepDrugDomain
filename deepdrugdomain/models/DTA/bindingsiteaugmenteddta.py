"""
Implementation of BindingSite-AugmentedDTA for drug-target affinity prediction.

Abstract:
BindingSite-AugmentedDTA is a deep learning (DL) based framework designed to improve 
drug-target affinity (DTA) predictions. This framework enhances the efficiency and 
accuracy of binding affinity predictions by reducing the search space for potential 
binding sites on proteins. A key feature of BindingSite-AugmentedDTA is its high 
generalizability, as it can be integrated with any DL-based regression model, 
significantly improving prediction performance. Additionally, the model stands out 
for its interpretability, thanks to its architecture and self-attention mechanism 
that maps attention weights back to protein-binding sites. Computational results 
confirm that BindingSite-AugmentedDTA enhances the performance of seven state-of-the-art 
DTA prediction algorithms across multiple evaluation metrics, including the concordance 
index, mean squared error, modified squared correlation coefficient, and the area under 
the precision curve. The model includes comprehensive data, such as 3D protein structures, 
from widely used datasets like Kiba and Davis, as well as the IDG-DREAM drug-kinase 
binding prediction challenge dataset. Furthermore, the practical potential of this 
framework is experimentally validated, showing high agreement between computational 
predictions and experimental observations, underscoring its utility in drug repurposing.

Citation:
Yousefi, N., Yazdani-Jahromi, M., Tayebi, A., Kolanthai, E., Neal, C. J., Banerjee, T., 
Gosai, A., Balasubramanian, G., Seal, S., & Ozmen Garibay, O. (2023). BindingSite-AugmentedDTA: 
enabling a next-generation pipeline for interpretable prediction models in drug repurposing. 
Briefings in Bioinformatics, 24(3), bbad136. https://doi.org/10.1093/bib/bbad136

GitHub Repository:
Source code available at: https://github.com/yazdanimehdi/BindingSite-AugmentedDTA.

Note:
Accurate preprocessing of drug and target data, including protein sequences and 3D 
structure information, is crucial for the effective use of BindingSite-AugmentedDTA. 
The model requires specific input formats for optimal performance in affinity prediction.

[Implementation details and methods go here.]
"""


from typing import Any, Dict
import torch
import torch.nn as nn
from deepdrugdomain.data.preprocessing import PreprocessingObject
from deepdrugdomain.layers import LinearHead
from deepdrugdomain.models import BaseInteractionModel
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from .deepdta import DeepDTA

from deepdrugdomain.models.factory import ModelFactory


class BindingSiteAugmentedDTA:
    """
    A class for augmenting drug-target interaction (DTI) models with additional graph encoders for ligand and protein inputs. This augmentation aims to improve the efficiency and accuracy of drug-target affinity (DTA) predictions by focusing on potential-binding sites of proteins.

    This class provides a method to enhance a given DTI model by incorporating separate graph encoders for ligand and protein inputs. The augmented model encodes these inputs and concatenates their outputs with the original model's output. This concatenated output is then passed through a modified linear head to produce the final result.

    Attributes:
    - trained_model_path (str): The path to the trained AttentionSiteDTI model.
    - graph_encoder_ligand (class): The graph encoder class for ligand inputs.
    - graph_encoder_ligand_kwargs (dict): Keyword arguments for initializing the ligand graph encoder.
    - graph_encoder_protein (class): The graph encoder class for protein inputs.
    - graph_encoder_protein_kwargs (dict): Keyword arguments for initializing the protein graph encoder.

    Methods:
    - augment(original_model_class): A method that takes a DTI model class as input and returns an augmented model class. The augmented model class includes additional encoders for ligand and protein, along with a modified forward method to handle the augmented inputs.
        default_preprocess(self, smile_attr, pdb_id_attr, *args): A method that takes the attribute names for ligand SMILES and protein PDB IDs as input and returns a list of PreprocessingObject instances for the default preprocessing pipeline. This method can be overridden to provide a custom preprocessing pipeline for the augmented model.

    Usage Example:
    >>> # Assuming LigandGraphEncoder and ProteinGraphEncoder are defined graph encoder classes
    >>> # and ligand_kwargs, protein_kwargs are dictionaries with appropriate arguments for these encoders.
    >>> binding_site_augmenter = BindingSiteAugmentedDTA(LigandGraphEncoder, ligand_kwargs, ProteinGraphEncoder, protein_kwargs)
    >>> # Define an original DTI model class that inherits from BaseInteractionModel
    >>> class MyOriginalDTIModel(BaseInteractionModel):
    ...     def __init__(self, ...):
    ...         super().__init__(...)
    ...         # Model initialization code
    ...
    ...     def forward(self, ...):
    ...         # Model forward pass code
    ...         pass
    >>> # Use the augmenter to decorate the original DTI model class
    >>> @binding_site_augmenter.augment
    ... class MyAugmentedModel(MyOriginalDTIModel):
    ...     pass
    >>> # Now, MyAugmentedModel is an augmented version of MyOriginalDTIModel
    >>> augmented_model = MyAugmentedModel(...)

    Note:
    - The original DTI model class must be a subclass of BaseInteractionModel.
    - The augmented model class automatically adjusts the input size of the linear head based on the concatenated outputs from the original and additional encoders.
    """
        
    def __init__(self, trained_model_path: str, graph_encoder_ligand: nn.Module, graph_encoder_ligand_kwargs: Dict[str, Any], graph_encoder_protein: nn.Module, graph_encoder_protein_kwargs: Dict[str, Any]) -> None:
        self.graph_encoder_ligand = graph_encoder_ligand
        self.graph_encoder_ligand_kwargs = graph_encoder_ligand_kwargs
        self.graph_encoder_protein = graph_encoder_protein
        self.graph_encoder_protein_kwargs = graph_encoder_protein_kwargs
        self.trained_model_path = trained_model_path

    def augment(original_model_class):
        assert issubclass(original_model_class, BaseInteractionModel)
        class AugmentedModel(original_model_class):
            def __init__(self, *args, **kwargs):
                super().__init__(remove_head=True, *args, **kwargs)
                self.encoder_ligand = self.graph_encoder_ligand(
                    **self.graph_encoder_ligand_kwargs)
                self.encoder_protein = self.graph_encoder_protein(
                    **self.graph_encoder_protein_kwargs)
                head_kwargs = self.head_kwargs

                head_kwargs['input_size'] = self.head_kwargs['input_size'] + \
                    self.encoder_ligand.get_output_size() + self.encoder_protein.get_output_size()

                self.head_new = LinearHead(**head_kwargs)

            def forward(self, ligand_graph, protein_graph, *args):
                # Process the original input
                original_output = super().forward(*args)

                # Process the additional inputs (this is just a placeholder)
                encoded_ligand = self.encoder_ligand(ligand_graph)
                encoded_protein = self.encoder_protein(protein_graph)

                # Concatenate the results
                concatenated_output = torch.cat(
                    (original_output, encoded_ligand, encoded_protein), dim=1)

                # Pass through the final linear layer(s)
                output = self.head_new(concatenated_output)
                return output
            
            def default_preprocess(self, smile_attr, pdb_id_attr, *args):
                original_default_preprocess = super().default_preprocess(*args)
                feat = CanonicalAtomFeaturizer()
                preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="graph", preprocessing_settings={
                    "fragment": False, "max_block": 6, "max_sr": 8, "min_frag_atom": 1, "node_featurizer": feat}, in_memory=True, online=False)
                preprocess_protein = PreprocessingObject(attribute=pdb_id_attr, from_dtype="pdb_id", to_dtype="binding_pocket_graph", preprocessing_settings={
                    "pdb_path": "data/pdb/", "protein_size_limit": 10000, "selection_model": self.trained_model_path}, in_memory=False, online=False)
                return [preprocess_drug, preprocess_protein] + original_default_preprocess

        return AugmentedModel

a = BindingSiteAugmentedDTA(trained_model_path="data/DeepDTA.pt", graph_encoder_ligand=DeepDTA, graph_encoder_ligand_kwargs={"input_size": 100, "hidden_sizes": [128, 128, 128], "output_size": 128, "dropout": 0.2, "normalization": True, "activation": "relu", "pooling": "mean", "pooling_kwargs": {}}, graph_encoder_protein=DeepDTA, graph_encoder_protein_kwargs={"input_size": 100, "hidden_sizes": [128, 128, 128], "output_size": 128, "dropout": 0.2, "normalization": True, "activation": "relu", "pooling": "mean", "pooling_kwargs": {}})

@ModelFactory.register('deepdta_augmented')
@a.augment
class AugmentedDeepDTA(DeepDTA):
    pass
