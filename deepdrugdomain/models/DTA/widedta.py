"""
"""
from deepdrugdomain.layers import CNNEncoder
from ..factory import ModelFactory
import torch
from ..interaction_model import BaseInteractionModel
from deepdrugdomain.data import PreprocessingObject


@ModelFactory.register('widedta')
class WideDTA(BaseInteractionModel):
    def __init__(self, motifs_input_dim, protein_input_dim, ligand_input_dim, encoder_ligand_kwargs, encoder_protein_kwargs, encoder_motif_kwargs, head_kwargs, *args, **kwargs):
        self.max_length_motifs = encoder_motif_kwargs['input_channels']
        self.number_of_combinations = motifs_input_dim
        self.smile_embedding_dim = ligand_input_dim
        encoder_1 = CNNEncoder(**encoder_ligand_kwargs)
        encoder_2 = CNNEncoder(**encoder_protein_kwargs)
        encoder_3 = CNNEncoder(**encoder_motif_kwargs)
        print(encoder_1.get_output_size(ligand_input_dim), encoder_2.get_output_size(protein_input_dim),
              encoder_3.get_output_size(motifs_input_dim))
        head_kwargs['input_size'] = encoder_1.get_output_size(ligand_input_dim) + encoder_2.get_output_size(
            protein_input_dim) + encoder_3.get_output_size(motifs_input_dim)
        del encoder_1, encoder_2, encoder_3
        self.max_length_ligand = encoder_ligand_kwargs['input_channels']
        self.max_length_protein = encoder_protein_kwargs['input_channels']
        encoders = [CNNEncoder, CNNEncoder, CNNEncoder]
        encoder_kwargs = [encoder_ligand_kwargs,
                          encoder_protein_kwargs, encoder_motif_kwargs]

        super(WideDTA, self).__init__(
            None,
            encoders=encoders,
            encoders_kwargs=encoder_kwargs,
            head_kwargs=head_kwargs,
            aggregation_method='concat',
            *args,
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
        preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="encoding_tensor", preprocessing_settings={
                                              "one_hot": True, "all_chars": True, "embedding_dim": self.smile_embedding_dim,  "max_sequence_length": self.max_length_ligand}, in_memory=True, online=False)
        preprocess_protein = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="encoding_tensor", preprocessing_settings={
                                                 "one_hot": True, "max_sequence_length": self.max_length_protein}, in_memory=True, online=False)
        preprocess_protein2 = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="motif_tensor", preprocessing_settings={
            "one_hot": True, "number_of_combinations": self.number_of_combinations, "max_length": self.max_length_motifs, "ngram": 3}, in_memory=True, online=False)
        preprocess_label = PreprocessingObject(attribute=label_attr,  from_dtype="binary",
                                               to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)
        return [preprocess_drug, preprocess_protein, preprocess_protein2, preprocess_label]
