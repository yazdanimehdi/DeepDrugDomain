"""
Implementation of DrugVQA for predicting drug-protein interactions.

Abstract:
DrugVQA introduces a novel approach to predict drug-protein interactions by 
representing proteins with two-dimensional distance maps (Image) from monomer 
structures and drugs with molecular linear notation (String), akin to visual 
question answering systems. This end-to-end deep learning framework addresses 
the limitations of one-dimensional protein sequences and inefficient three-dimensional 
structure inputs. It employs a dynamic attentive convolutional neural network to 
learn from variable-length distance maps and a self-attentional sequential model for 
extracting features from linear notations. DrugVQA has demonstrated competitive 
performance on DUD-E, human, and BindingDB benchmark datasets, and provides 
biological interpretation through attention visualization, highlighting key regions 
in both protein and drug molecules.

Citation:
Zheng, S., Li, Y., Chen, S., et al. (2020). Predicting drug-protein interaction using 
quasi-visual question answering system. Nature Machine Intelligence, 2, 134-140. 
https://doi.org/10.1038/s42256-020-0152-y

GitHub Repository:
Source code available at: https://github.com/prokia/drugVQA.

Note:
To effectively use DrugVQA, users must ensure appropriate preprocessing of protein 
distance maps and drug linear notations. The model demands precise input formats 
for accurate drug-protein interaction prediction and analysis.
"""


from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Dict, Union
import numpy as np

from tqdm import tqdm

from deepdrugdomain.layers.utils import LayerFactory, ActivationFactory
from deepdrugdomain.layers import LSTMEncoder, LinearHead

from ..factory import ModelFactory
import torch
from torch import nn
import torch.nn.functional as F
from deepdrugdomain.utils.weight_init import trunc_normal_
from torch.autograd import Variable
from ..interaction_model import BaseInteractionModel
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from deepdrugdomain.schedulers import BaseScheduler
from deepdrugdomain.data import PreprocessingList, PreprocessingObject

# Weird attention layer, but I guess it works!


@LayerFactory.register('drugvqa_seq_attention')
class DrugVQASequentialAttention(nn.Module):
    def __init__(self, num_heads: int, dim_input: int, dim_hidden: int, activation_fn: str = "tanh", ) -> None:
        super().__init__()
        self.first_linear = nn.Linear(dim_input, dim_hidden)
        self.activation_fn = ActivationFactory.create(activation_fn)
        self.second_linear = nn.Linear(dim_hidden, num_heads)
        self.num_heads = num_heads

    def forward(self, x):
        att = self.activation_fn(self.first_linear(x))
        att = self.second_linear(att)
        att = torch.softmax(att, axis=1)
        att = att.transpose(1, 2)
        embed = att @ x
        embed = torch.sum(embed, 1) / self.num_heads
        return embed


class DrugVQADrugEncoder(nn.Module):
    def __init__(self, drug_config: Dict[str, Any]) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            drug_config["vocab_size_smiles"], drug_config["embedding_dim"])
        self.lstm = LSTMEncoder(drug_config["embedding_dim"], drug_config["lstm_hid_dim"],
                                drug_config["lstm_layers"], drug_config["lstm_dropout"], drug_config["lstm_bidirectional"])
        self.smile_attention = LayerFactory.create(
            drug_config["attention_layers_smile"], **drug_config["attention_layers_smile_kwargs"])

    def forward(self, x):
        smile_embed = self.embeddings(x)
        outputs = self.lstm(smile_embed)
        avg_sentence_embed = self.smile_attention(outputs)

        return avg_sentence_embed


class DrugVQAProteinEncoder(nn.Module):
    def __init__(self, protein_config: Dict[str, Any]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(protein_config["contact_map_in_channels"], protein_config["contact_map_out_channels"],
                              kernel_size=protein_config["contact_map_kernel_size"], stride=protein_config["contact_map_stride"])
        self.bn = LayerFactory.create(
            protein_config["contact_map_normalization_layer"], protein_config["contact_map_out_channels"])
        self.contact_map_act = ActivationFactory.create(
            protein_config["contact_map_activation_fn"])

        self.res_block1 = nn.Sequential(*[LayerFactory.create(
            protein_config["resnet_block1"], inplanes=protein_config["contact_map_out_channels"], planes=protein_config["contact_map_out_channels"], **protein_config["resnet_block1_kwargs"]) for _ in range(protein_config["resnet_block1_layers"])])

        self.res_block2 = nn.Sequential(*[LayerFactory.create(
            protein_config["resnet_block2"], inplanes=protein_config["contact_map_out_channels"], planes=protein_config["contact_map_out_channels"], **protein_config["resnet_block2_kwargs"]) for _ in range(protein_config["resnet_block2_layers"])])

        self.seq_attention = LayerFactory.create(
            protein_config["attention_layers_seq"], **protein_config["attention_layers_seq_kwargs"])

    def forward(self, x):
        pic = self.conv(x)
        pic = self.bn(pic)

        pic = self.contact_map_act(pic)

        pic = self.res_block1(pic)
        pic = self.res_block2(pic)

        pic_emb = torch.mean(pic, 2)

        pic_emb = pic_emb.permute(0, 2, 1)
        avg_seq_embed = self.seq_attention(pic_emb)

        return avg_seq_embed


@ModelFactory.register('drugvqa')
class DrugVQA(BaseInteractionModel):
    def __init__(self,
                 num_batches: int,
                 drug_config: Dict[str, Any],
                 protein_config: Dict[str, Any],
                 head_config: Dict[str, Any]):
        drug_encoder = DrugVQADrugEncoder(drug_config)
        protein_encoder = DrugVQAProteinEncoder(protein_config)
        encoders = [drug_encoder, protein_encoder]
        head_config['input_size'] = drug_config["lstm_hid_dim"] * 2 + \
            protein_config["contact_map_out_channels"]

        super().__init__(
            embedding_dim=None,
            encoders=encoders,
            head_kwargs=head_config,
            aggregation_method='concat',
        )
        self.drug_encoder = drug_encoder
        self.protein_encoder = protein_encoder
        self.vocab_size_smiles = drug_config["vocab_size_smiles"]
        self.drug_config = drug_config
        self.protein_config = protein_config
        self.num_batches = num_batches

    def get_drug_encoder(self, smile_attr):
        return {
            "encoder": self.drug_encoder,
            "preprocessor": self.default_preprocess(smile_attr, None, None),
            "output_dim": self.drug_config["lstm_hid_dim"] * 2
        }

    def get_protein_encoder(self, target_seq_attr):
        return {
            "encoder": self.protein_encoder,
            "preprocessor": self.default_preprocess(None, target_seq_attr, None),
            "output_dim": self.protein_config["contact_map_out_channels"]
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def predict(self, *args, **kwargs) -> Any:
        return super().predict(*args, **kwargs)

    def collate(self, batch: List[Tuple[Any, Any, torch.Tensor]]) -> Tuple[Tuple[List[Any], List[Any]], torch.Tensor]:
        """
            Collate function for the DrugVQA model.
        """
        # Unpacking the batch data
        drug, protein, targets = zip(*batch)
        targets = torch.stack(targets, 0)

        return drug, protein, targets

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

                outs = []
                for item in range(len(drug)):
                    d = drug[item].unsqueeze(0).to(device)
                    p = protein[item].unsqueeze(0).to(device)
                    out = self.forward(d, p)

                    if isinstance(out, tuple):
                        out = out[0]

                    outs.append(out)

                out = torch.stack(outs, dim=0).squeeze(1)
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
        """
            Evaluate the model on the given dataset.
        """
        losses = []
        predictions = []
        targets = []
        self.eval()
        with tqdm(dataloader) as t:
            t.set_description('Testing')
            for batch_idx, (drug, protein, target) in enumerate(t):
                outs = []
                with torch.no_grad():
                    for item in range(len(drug)):
                        d = drug[item].unsqueeze(0).to(device)
                        p = protein[item].unsqueeze(0).to(device)
                        out = self.forward(d, p)
                        if isinstance(out, tuple):
                            out = out[0]

                        outs.append(out)

                out = torch.stack(outs, dim=0).squeeze(1)
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

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)

    def default_preprocess(self, smile_attr: Optional[str] = None, pdb_id_attr: Optional[str] = None, label_attr: Optional[str] = None) -> Union[PreprocessingObject, List[PreprocessingObject]]:

        assert smile_attr is not None or pdb_id_attr is not None, "At least one of smile_attr or target_seq_attr must be specified."

        if pdb_id_attr is None:
            preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="kword_encoding_tensor",
                                                  preprocessing_settings={"window": 1,
                                                                          "stride": 1,
                                                                          "convert_deepsmiles": False,
                                                                          "one_hot": False,
                                                                          "max_length": None,
                                                                          "num_of_combinations": self.vocab_size_smiles}, in_memory=True, online=False)
            return preprocess_drug
        elif smile_attr is None:
            preprocess_protein = PreprocessingObject(
                attribute=pdb_id_attr, from_dtype="pdb_id", to_dtype="contact_map", preprocessing_settings={"pdb_path": "data/pdb/"}, in_memory=False, online=False)
            return preprocess_protein

        else:
            preprocess_drug = PreprocessingObject(attribute=smile_attr, from_dtype="smile", to_dtype="kword_encoding_tensor",
                                                  preprocessing_settings={"window": 1,
                                                                          "stride": 1,
                                                                          "convert_deepsmiles": False,
                                                                          "one_hot": False,
                                                                          "max_length": None,
                                                                          "num_of_combinations": self.vocab_size_smiles}, in_memory=True, online=False)
            preprocess_protein = PreprocessingObject(
                attribute=pdb_id_attr, from_dtype="pdb_id", to_dtype="contact_map", preprocessing_settings={"pdb_path": "data/pdb/"}, in_memory=False, online=False)
            preprocess_label = PreprocessingObject(
                attribute=label_attr, from_dtype="binary", to_dtype="binary_tensor", preprocessing_settings={}, in_memory=True, online=True)

            return [preprocess_drug, preprocess_protein, preprocess_label]
