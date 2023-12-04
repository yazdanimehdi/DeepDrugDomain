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
        super(DeepDTA, self).__init__(
            embedding_dim,
            encoder_1=CNNEncoder,
            encoder_1_kwargs=encoder_ligand_kwargs,
            encoder_2=CNNEncoder,
            encoder_2_kwargs=encoder_protein_kwargs,
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
