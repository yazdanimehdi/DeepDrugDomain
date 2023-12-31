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
                 protein_max_length: int,

                 protein_kernel: List[int],
                 protein_strides: List[int],
                 protein_cnn_activation: List[str],
                 protein_cnn_dropout: List[float],
                 protein_cnn_normalization: List[str],
                 protein_hidden_channels: List[int],

                 drug_max_length: int,
                 drug_kernel: List[int],
                 drug_strides: List[int],
                 drug_cnn_activation: List[str],
                 drug_cnn_dropout: List[float],
                 drug_cnn_normalization: List[str],
                 drug_hidden_channels: List[int],

                 cnn_out_channels,

                 attention_layer: str,
                 head_num: int,

                 char_dim: int,

                 head_output_dim: int,
                 head_dims: List[int],
                 head_activations: List[str],
                 head_normalization: List[str],
                 head_dropout_rate: List[float],
                 ):
        super(AttentionDTA_TCBB, self).__init__()
        self.dim = char_dim
        self.head_num = head_num
        self.drug_max_length = drug_max_length
        self.drug_kernel = drug_kernel
        self.protein_max_length = protein_max_length
        self.protein_kernel = protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)

        self.drug_cnn_encoder = CNNEncoder(
            input_channels=self.dim,
            hidden_channels=protein_hidden_channels,
            output_channels=cnn_out_channels,
            kernel_sizes=protein_kernel,
            strides=protein_strides,
            pooling=None,
            pooling_kwargs=None,
            paddings=0,
            activations=protein_cnn_activation,
            dropouts=protein_cnn_dropout,
            normalization=protein_cnn_normalization,
        )

        self.drug_pooling = nn.AdaptiveAvgPool1d(1)

        self.protein_cnn_encoder = CNNEncoder(
            input_channels=self.dim,
            hidden_channels=drug_hidden_channels,
            output_channels=cnn_out_channels,
            kernel_sizes=drug_kernel,
            strides=drug_strides,
            pooling=None,
            pooling_kwargs=None,
            paddings=0,
            activations=drug_cnn_activation,
            dropouts=drug_cnn_dropout,
            normalization=drug_cnn_normalization,
        )

        self.protein_pooling = nn.AdaptiveAvgPool1d(1)

        self.attention = LayerFactory.create(
            attention_layer, cnn_out_channels, head_num)

        self.head_output_dim = head_output_dim
        self.head_dims = head_dims
        self.head_activations = head_activations
        self.head_normalization = head_normalization
        self.head_dropout_rate = head_dropout_rate
        self.cnn_out_channels = cnn_out_channels

        self.head = LinearHead(self.cnn_out_channels * 2, self.head_output_dim, self.head_dims,
                               self.head_activations, self.head_dropout_rate, self.head_normalization)

    def forward(self, drug, protein):
        drug_embed = self.drug_embed(drug)
        protein_embed = self.protein_embed(protein)
        drug_embed = drug_embed.permute(0, 2, 1)
        protein_embed = protein_embed.permute(0, 2, 1)

        drug_conv = self.drug_cnn_encoder(drug_embed)
        protein_conv = self.protein_cnn_encoder(protein_embed)

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

    def default_preprocess(self, smile_attr, target_seq_attr, label_attr):
        CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                         "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                         "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                         "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                         "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                         "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                         "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                         "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
        
        protein_dict = {x: i for i, x in enumerate("ACBEDGFIHKMLONQPSRTWVYXZ")}
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
