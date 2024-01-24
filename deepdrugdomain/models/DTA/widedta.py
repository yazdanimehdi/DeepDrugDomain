"""
Implementation of the WideDTA model for predicting drug-target binding affinity.

Abstract:
WideDTA is a deep-learning based model developed to predict the interaction affinity 
between proteins and compounds, a crucial aspect of the drug discovery process. The model 
utilizes four text-based information sources: protein sequence, ligand SMILES, protein 
domains and motifs, and maximum common substructure words. WideDTA has shown to outperform 
DeepDTA, a state-of-the-art model, on the KIBA dataset with statistical significance. This 
suggests the effectiveness of WideDTA's word-based sequence representation approach over 
character-based sequence representations in deep learning models. The study also reveals 
that while protein sequence and ligand SMILES are essential, the inclusion of protein 
domain, motif information, and ligand maximum common substructure words does not 
significantly enhance the model's performance. Intriguingly, representing proteins solely 
through domain and motif information achieves similar performance to using the full 
protein sequence, highlighting the importance of these elements in binding affinity.

Citation:
Öztürk, H., Ozkirimli, E., & Özgür, A. (2019). WideDTA: prediction of drug-target 
binding affinity. arXiv preprint arXiv:1902.04166.

GitHub Repository:
Source code available at: https://github.com/Sunitach10/MolPro.

Note:
To effectively use WideDTA, it is crucial to accurately preprocess protein sequences, 
ligand SMILES, and other molecular features. Proper input data formatting is necessary 
for optimal performance and accurate affinity prediction.
"""

from deepdrugdomain.layers import CNNEncoder
from ..factory import ModelFactory
import torch
from torch import nn
from ..interaction_model import BaseInteractionModel
from deepdrugdomain.data import PreprocessingObject


@ModelFactory.register('widedta')
class WideDTA(BaseInteractionModel):
    def __init__(self, motifs_input_dim, protein_input_dim, ligand_input_dim, encoder_ligand_kwargs, encoder_protein_kwargs, encoder_motif_kwargs, head_kwargs, *args, **kwargs):
        self.max_length_motifs = encoder_motif_kwargs['input_channels']
        self.number_of_combinations = motifs_input_dim
        self.smile_embedding_dim = ligand_input_dim
        self.protein_input_dim = protein_input_dim
        encoder_1 = CNNEncoder(**encoder_ligand_kwargs)
        encoder_2 = CNNEncoder(**encoder_protein_kwargs)
        encoder_3 = CNNEncoder(**encoder_motif_kwargs)
        head_kwargs['input_size'] = encoder_1.get_output_size(ligand_input_dim) + encoder_2.get_output_size(
            protein_input_dim) + encoder_3.get_output_size(motifs_input_dim)

        self.max_length_ligand = encoder_ligand_kwargs['input_channels']
        self.max_length_protein = encoder_protein_kwargs['input_channels']
        encoders = nn.ModuleList([encoder_1, encoder_2, encoder_3])

        super(WideDTA, self).__init__(
            encoders=encoders,
            head_kwargs=head_kwargs,
            aggregation_method='concat',
            **kwargs
        )

    def collate(self, batch):
        ligand, protein, motif, label = zip(*batch)
        ligand = torch.stack(ligand)
        protein = torch.stack(protein)
        motif = torch.stack(motif)
        label = torch.stack(label)
        return ligand, protein, motif, label

    def default_preprocess(self, smile_attr, target_seq_attr, label_attr):
        preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="kword_encoding_tensor", preprocessing_settings={
            "window": 8, "stride": 8, "convert_deepsmiles": True, "one_hot": True, "max_length": self.max_length_ligand, "num_of_combinations": self.smile_embedding_dim}, in_memory=True, online=False)
        preprocess_protein = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="kmers_encoded_tensor", preprocessing_settings={
            "window": 1, "stride": 1,
            "one_hot": True, "number_of_combinations": self.protein_input_dim, "max_length": self.max_length_protein}, in_memory=True, online=False)
        preprocess_protein2 = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="motif_tensor", preprocessing_settings={
            "one_hot": True, "number_of_combinations": self.number_of_combinations, "max_length": self.max_length_motifs, "ngram": 3}, in_memory=True, online=False)
        preprocess_label = PreprocessingObject(attribute=label_attr,  from_dtype="binary",
                                               to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)
        return [preprocess_drug, preprocess_protein, preprocess_protein2, preprocess_label]
