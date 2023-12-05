"""
Implementation of DeepDTA for drug-target binding affinity prediction.

Abstract:
DeepDTA is a deep learning-based model for predicting drug-target interaction 
binding affinities using sequence information of both targets and drugs. This 
approach differs from traditional binary classification methods, focusing 
instead on a continuum of binding affinity values. DeepDTA uses convolutional 
neural networks (CNNs) to model protein sequences and compound 1D representations. 
It demonstrates effectiveness in predicting drug-target binding affinities, 
outperforming existing methods like KronRLS and SimBoost on benchmark datasets.

Citation:
Öztürk, H., Özgür, A., & Ozkirimli, E. (2018). DeepDTA: deep drug-target binding 
affinity prediction. Bioinformatics, 34(17), i821-i829. doi: 10.1093/bioinformatics/bty593. 
PMID: 30423097; PMCID: PMC6129291.

Availability and implementation:
Source code available at: https://github.com/hkmztrk/DeepDTA.

Note:
Proper preprocessing of input sequences is essential. The model requires 
correctly formatted drug and target sequences for accurate affinity predictions.
"""
from deepdrugdomain.layers import CNNEncoder
from ..factory import ModelFactory
import torch
from ..interaction_model import BaseInteractionModel
from deepdrugdomain.data import PreprocessingObject


@ModelFactory.register('deepdta')
class DeepDTA(BaseInteractionModel):
    def __init__(self, embedding_dim, encoder_ligand_kwargs, encoder_protein_kwargs, head_kwargs, *args, **kwargs):
        encoder_ligand_kwargs['embedding_dim'] = embedding_dim
        encoder_protein_kwargs['embedding_dim'] = embedding_dim
        head_kwargs['input_size'] = encoder_ligand_kwargs["output_channels"] + \
            encoder_protein_kwargs["output_channels"]
        self.max_length_ligand = encoder_ligand_kwargs['input_channels']
        self.max_length_protein = encoder_protein_kwargs['input_channels']
        encoders = [CNNEncoder, CNNEncoder]
        encoder_kwargs = [encoder_ligand_kwargs, encoder_protein_kwargs]

        super(DeepDTA, self).__init__(
            embedding_dim,
            encoders=encoders,
            encoders_kwargs=encoder_kwargs,
            head_kwargs=head_kwargs,
            aggregation_method='concat',
            *args,
            **kwargs
        )

    def collate(self, batch):
        ligand, protein, label = zip(*batch)
        ligand = torch.stack(ligand)
        protein = torch.stack(protein)
        label = torch.stack(label)
        return ligand, protein, label

    def default_preprocess(self, smile_attr, target_seq_attr, label_attr):
        preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="encoding_tensor", preprocessing_settings={
                                              "all_chars": True, "max_sequence_length": self.max_length_ligand}, in_memory=True, online=False)
        preprocess_protein = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="encoding_tensor", preprocessing_settings={
                                                 "one_hot": False, "max_sequence_length": self.max_length_protein, "amino_acids": "ACBEDGFIHKMLONQPSRTWVYXZ"}, in_memory=True, online=False)
        preprocess_label = PreprocessingObject(attribute=label_attr,  from_dtype="binary",
                                               to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)
        return [preprocess_drug, preprocess_protein, preprocess_label]
