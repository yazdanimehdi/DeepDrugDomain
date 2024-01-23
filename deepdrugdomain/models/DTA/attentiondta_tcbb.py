"""
Implementation of the AttentionDTA model for drug-target binding affinity prediction.

Abstract:
The AttentionDTA model addresses the challenge of predicting drug-target relations (DTRs) 
by treating them as a regression problem of drug-target affinities (DTAs). Unlike traditional 
methods that view DTRs as binary classification (drug-target interactions or DTIs), this model 
focuses on the quantitative aspects of DTRs, like dose dependence and binding affinities. 
AttentionDTA employs a deep learning-based approach, utilizing two separate one-dimensional 
Convolution Neural Networks (1D-CNNs) to process the semantic information of drug's SMILES strings 
and protein's amino acid sequences. A novel aspect of this model is the use of a two-sided 
multi-head attention mechanism to explore the relationship between drug and protein features, 
enhancing the biological interpretability of the deep learning model. The model's effectiveness 
has been demonstrated across multiple established DTA benchmark datasets (Davis, Metz, KIBA), 
outperforming state-of-the-art methods. Additionally, the model's capability to identify binding 
sites and its biological significance have been validated through visualization of attention weights.

Citation:
Zhao, Q., Duan, G., Yang, M., Cheng, Z., Li, Y., & Wang, J. (2023). AttentionDTA: Drug-Target Binding 
Affinity Prediction by Sequence-Based Deep Learning With Attention Mechanism. IEEE/ACM Transactions 
on Computational Biology and Bioinformatics, 20(2), 852-863. doi: 10.1109/TCBB.2022.3170365. Epub 2023 
Apr 3. PMID: 35471889.

Source Code:
The source code for AttentionDTA is available at: https://github.com/zhaoqichang/AttentionDTA_TCBB

Note:
Users of this implementation should ensure that they have the appropriate data preprocessing steps 
in place and understand the model's input and output formats for effective utilization.
"""

import torch.nn as nn
from deepdrugdomain.layers import GraphConvEncoder, LayerFactory, CNNEncoder, LinearHead
from ..base_model import BaseModel
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Type
from ..factory import ModelFactory
from ..base_model import BaseModel
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from deepdrugdomain.schedulers import BaseScheduler
import torch
import numpy as np
from tqdm import tqdm
from deepdrugdomain.data import PreprocessingObject


@LayerFactory.register('attentiondta_attention')
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head=8):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.d_a = nn.Linear(self.dim, self.dim * head)
        self.p_a = nn.Linear(self.dim, self.dim * head)
        self.scale = torch.sqrt(torch.FloatTensor([self.dim * 3]))

    def forward(self, drug, protein):
        bsz, d_ef, d_il = drug.shape
        bsz, p_ef, p_il = protein.shape
        drug_att = self.relu(self.d_a(drug.permute(0, 2, 1))).view(
            bsz, self.head, d_il, d_ef)
        protein_att = self.relu(self.p_a(protein.permute(0, 2, 1))).view(
            bsz, self.head, p_il, p_ef)
        interaction_map = torch.mean(self.tanh(torch.matmul(
            drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale), 1)
        compound_attention = self.tanh(
            torch.sum(interaction_map, 2)).unsqueeze(1)
        protein_attention = self.tanh(
            torch.sum(interaction_map, 1)).unsqueeze(1)
        drug = drug * compound_attention
        protein = protein * protein_attention

        return drug, protein


@ModelFactory.register('attentiondta_tcbb')
class AttentionDTA_TCBB(BaseModel):
    def __init__(self,
                 protein_config: Dict[str, Any],

                 drug_config: Dict[str, Any],

                 aggregation_config: Dict[str, Any],

                 head_config: Dict[str, Any],

                 ):
        super(AttentionDTA_TCBB, self).__init__()

        self.drug_max_length = drug_config["max_length"]
        self.protein_max_length = protein_config["max_length"]

        assert protein_config["cnn_out_channels"] == drug_config[
            "cnn_out_channels"], "The output channels of the drug and protein CNN encoders must be the same."
        self.cnn_out_channels = protein_config['cnn_out_channels']

        self.drug_cnn_encoder = CNNEncoder(
            input_channels=drug_config["dim"],
            hidden_channels=drug_config["hidden_channels"],
            output_channels=drug_config["cnn_out_channels"],
            kernel_sizes=drug_config["kernel"],
            strides=drug_config["strides"],
            pooling=drug_config["cnn_pooling"],
            pooling_kwargs=drug_config["cnn_pooling_kwargs"],
            paddings=drug_config["cnn_padding"],
            activations=drug_config["cnn_activation"],
            dropouts=drug_config["cnn_dropout"],
            normalization=drug_config["cnn_normalization"],
            input_embedding_dim=drug_config["input_embedding_dim"],
            permute_embedding_indices=drug_config["permute_embedding_indices"]
        )

        self.drug_pooling = nn.AdaptiveAvgPool1d(1)

        self.protein_cnn_encoder = CNNEncoder(
            input_channels=protein_config["dim"],
            hidden_channels=protein_config["hidden_channels"],
            output_channels=protein_config["cnn_out_channels"],
            kernel_sizes=protein_config["kernel"],
            strides=protein_config["strides"],
            pooling=protein_config["cnn_pooling"],
            pooling_kwargs=protein_config["cnn_pooling_kwargs"],
            paddings=protein_config["cnn_padding"],
            activations=protein_config["cnn_activation"],
            dropouts=protein_config["cnn_dropout"],
            normalization=protein_config["cnn_normalization"],
            input_embedding_dim=protein_config["input_embedding_dim"],
            permute_embedding_indices=protein_config["permute_embedding_indices"]
        )

        self.protein_pooling = nn.AdaptiveAvgPool1d(1)

        self.attention = LayerFactory.create(
            aggregation_config['attention_layer'], self.cnn_out_channels,  aggregation_config['head_num'])

        self.head_output_dim = head_config['head_output_dim']
        self.head_dims = head_config['head_dims']
        self.head_activations = head_config['head_activations']
        self.head_normalization = head_config['head_normalization']
        self.head_dropout_rate = head_config['head_dropout_rate']

        self.head = LinearHead(self.cnn_out_channels * 2, self.head_output_dim, self.head_dims,
                               self.head_activations, self.head_dropout_rate, self.head_normalization)
        
    def get_drug_encoder(self, smile_attr):
        return {
            "encoder": self.drug_cnn_encoder,
            "preprocessor": self.default_preprocess(smile_attr),
            "output_dim": self.cnn_out_channels
        }
    
    def get_protein_encoder(self, target_seq_attr):
        return {
            "encoder": self.protein_cnn_encoder,
            "preprocessor": self.default_preprocess(target_seq_attr),
            "output_dim": self.cnn_out_channels
        }

    def forward(self, drug, protein):

        drug_conv = self.drug_cnn_encoder(drug)
        protein_conv = self.protein_cnn_encoder(protein)

        drug_conv, protein_conv = self.attention(drug_conv, protein_conv)
        drug_conv = self.drug_pooling(drug_conv).squeeze(2)
        protein_conv = self.protein_pooling(protein_conv).squeeze(2)
        pair = torch.cat([drug_conv, protein_conv], dim=1)
        return self.head(pair)

    def predict(self, drug, protein):
        return self.forward(drug, protein)

    def train_one_epoch(self, dataloader: DataLoader, device: torch.device, criterion: Callable, optimizer: Optimizer, num_epochs: int, scheduler: Optional[Type[BaseScheduler]] = None, evaluator: Optional[Type[Evaluator]] = None, grad_accum_steps: int = 1, clip_grad: Optional[str] = None, logger: Optional[Any] = None) -> Any:
        """
            Train the model for one epoch.
        """
        accum_steps = grad_accum_steps
        last_accum_steps = len(dataloader) % accum_steps
        updates_per_epoch = (len(dataloader) + accum_steps - 1) // accum_steps
        num_updates = num_epochs * updates_per_epoch
        last_batch_idx = len(dataloader) - 1
        last_batch_idx_to_accum = len(dataloader) - last_accum_steps

        losses = []
        predictions = []
        targets = []
        self.train()
        with tqdm(dataloader) as t:
            t.set_description('Training')
            for batch_idx, (drug, protein, target) in enumerate(t):
                last_batch = batch_idx == last_batch_idx
                need_update = last_batch or (batch_idx + 1) % accum_steps == 0
                update_idx = batch_idx // accum_steps

                if batch_idx >= last_batch_idx_to_accum:
                    accum_steps = last_accum_steps

                drug = drug.to(device)
                protein = protein.to(device)
                out = self.forward(drug, protein)

                target = target.to(
                    device).view(-1, 1).to(torch.float)

                loss = criterion(out, target)
                loss /= accum_steps

                loss.backward()
                losses.append(loss.detach().cpu().item())
                predictions.append(out.detach().cpu())
                targets.append(target.detach().cpu())
                metrics = evaluator(predictions, targets) if evaluator else {}

                metrics["loss"] = np.mean(losses) * accum_steps
                lrl = [param_group['lr']
                       for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                metrics["lr"] = lr

                t.set_postfix(**metrics)

                if logger is not None:
                    logger.log(metrics)

                optimizer.step()

                num_updates += 1
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step_update(
                        num_updates=num_updates, metric=metrics["loss"])

    def evaluate(self, dataloader: DataLoader, device: torch.device, criterion: Callable, evaluator: Optional[Type[Evaluator]] = None, logger: Optional[Any] = None) -> Any:
        losses = []
        predictions = []
        targets = []
        self.eval()
        with tqdm(dataloader) as t:
            t.set_description('Testing')
            for batch_idx, (drug, protein, target) in enumerate(t):
                drug = drug.to(device)
                protein = protein.to(device)

                with torch.no_grad():
                    out = self.forward(drug, protein)

                target = target.to(
                    device).view(-1, 1).to(torch.float)

                loss = criterion(out, target)
                losses.append(loss.detach().cpu().item())
                predictions.append(out.detach().cpu())
                targets.append(target.detach().cpu())
                metrics = evaluator(predictions, targets) if evaluator else {}
                metrics["loss"] = np.mean(losses)
                t.set_postfix(**metrics)

        metrics = evaluator(predictions, targets) if evaluator else {}
        metrics["loss"] = np.mean(losses)

        metrics = {"val_" + str(key): val for key, val in metrics.items()}

        if logger is not None:
            logger.log(metrics)
        return metrics

    def collate(self, batch: List[Tuple[Any, Any, torch.Tensor]]) -> Tuple[Tuple[List[Any], List[Any]], torch.Tensor]:
        """
            Collate function for the AMMVF model.
        """
        # Unpacking the batch data
        drug, protein, targets = zip(*batch)
        targets = torch.stack(targets, 0)
        drug = torch.stack(drug, 0)
        protein = torch.stack(protein, 0)

        return drug, protein, targets

    def reset_head(self) -> None:
        self.head = LinearHead(self.cnn_out_channels * 2, self.head_output_dim, self.head_dims,
                               self.head_activations, self.head_dropout_rate, self.head_normalization)

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)

    def default_preprocess(self, smile_attr=None, target_seq_attr=None, label_attr=None):
        CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                         "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                         "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                         "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                         "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                         "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                         "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                         "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

        protein_dict = {x: i for i, x in enumerate("ACBEDGFIHKMLONQPSRTWVYXZ")}
        assert smile_attr is not None or target_seq_attr is not None, "At least one of smile_attr or target_seq_attr must be specified."
        if target_seq_attr is None and label_attr is None:
            preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="kword_encoding_tensor",
                                              preprocessing_settings={"window": 1,
                                                                      "stride": 1,
                                                                      "word_dict": CHARISOSMISET,
                                                                      "convert_deepsmiles": False,
                                                                      "one_hot": False,
                                                                      "max_length": self.drug_max_length}, in_memory=True, online=False)
            return preprocess_drug
        
        if smile_attr is None and label_attr is None:
            preprocess_protein = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="kmers_encoded_tensor", preprocessing_settings={
                "window": 1, "stride": 1,
                "one_hot": False, "word_dict": protein_dict, "max_length": self.protein_max_length}, in_memory=True, online=False)
            return preprocess_protein
        
        else:
            preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="kword_encoding_tensor",
                                              preprocessing_settings={"window": 1,
                                                                      "stride": 1,
                                                                      "word_dict": CHARISOSMISET,
                                                                      "convert_deepsmiles": False,
                                                                      "one_hot": False,
                                                                      "max_length": self.drug_max_length}, in_memory=True, online=False)
            
            preprocess_protein = PreprocessingObject(attribute=target_seq_attr, from_dtype="protein_sequence", to_dtype="kmers_encoded_tensor", preprocessing_settings={
                "window": 1, "stride": 1,
                "one_hot": False, "word_dict": protein_dict, "max_length": self.protein_max_length}, in_memory=True, online=False)
            preprocess_label = PreprocessingObject(attribute=label_attr,  from_dtype="binary",
                                                to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)
            return [preprocess_drug, preprocess_protein, preprocess_label]
