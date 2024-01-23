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
    def __init__(self, drug_config, protein_config, head_config, *args, **kwargs):
        head_config['input_size'] = drug_config["output_channels"] + \
            protein_config["output_channels"]
        self.max_length_ligand = drug_config['input_channels']
        self.max_length_protein = protein_config['input_channels']
        encoders = [CNNEncoder(**drug_config), CNNEncoder(**protein_config)]

        super(DeepDTA, self).__init__(
            None,
            encoders=encoders,
            head_kwargs=head_config,
            aggregation_method='concat',
        )

    def collate(self, batch):
        a = zip(*batch)
        a = [torch.stack(x) for x in a]

        return a

    def default_preprocess(self, smile_attr, target_seq_attr, label_attr):
        protein_dict = {x: i for i, x in enumerate("ACBEDGFIHKMLONQPSRTWVYXZ")}
        preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="kword_encoding_tensor",
                                              preprocessing_settings={"window": 1,
                                                                      "stride": 1,
                                                                      "convert_deepsmiles": False,
                                                                      "one_hot": False,
                                                                      "max_length": self.max_length_ligand}, in_memory=True, online=False)
        preprocess_protein = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="kmers_encoded_tensor", preprocessing_settings={
            "window": 1, "stride": 1,
            "one_hot": False, "word_dict": protein_dict, "max_length": self.max_length_protein}, in_memory=True, online=False)
        preprocess_label = PreprocessingObject(attribute=label_attr,  from_dtype="tensor",
                                               to_dtype="log_tensor", preprocessing_settings={}, in_memory=True, online=True)
        return [preprocess_drug, preprocess_protein, preprocess_label]
