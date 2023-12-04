from .base_model import BaseModel
from deepdrugdomain.layers import LayerFactory, LinearHead
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Optional, Type, Union
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from deepdrugdomain.schedulers import BaseScheduler
import numpy as np
from tqdm import tqdm


class BaseInteractionModel(BaseModel):
    def __init__(self, embedding_dim: int, encoder_1: nn.Module, encoder_1_kwargs: Dict[str, Any], encoder_2: nn.Module, encoder_2_kwargs: Dict[str, Any], head_kwargs: Dict[str, Any], aggregation_method: str, aggregation_module: Optional[Union[nn.Module, str]] = None, *args, **kwargs):
        super(BaseInteractionModel, self).__init__()
        self.encoder_1 = encoder_1(**encoder_1_kwargs)
        self.encoder_2 = encoder_2(**encoder_2_kwargs)
        self.head_kwargs = head_kwargs
        self.head = LinearHead(**head_kwargs)
        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method

        assert self.aggregation_method in ['concat', 'sum', 'product', 'average', 'max_pooling', 'dot_product',
                                           'weighted_sum', 'self_attention', 'cross_attention', 'cosine_similarity', 'custom'], 'Aggregation method not supported'
        if self.aggregation_method == 'custom':
            assert aggregation_module is not None, 'Aggregation module not provided'
            self.aggregation_module = LayerFactory.create(
                aggregation_module, **kwargs['aggregation_module']) if isinstance(aggregation_module, str) else aggregation_module

        if self.aggregation_method == 'cross_attention':
            assert 'cross_attention' in kwargs, 'Cross attention parameters are not provided'

            if kwargs['transformer_cross_attention']['num_layers'] > 1:
                self.cross_attention = LayerFactory.create(
                    'transformer_cross_attention_block', **kwargs['cross_attention'])
            else:
                self.cross_attention = LayerFactory.create(
                    'transformer_cross_attention', **kwargs['cross_attention'])

        elif self.aggregation_method == 'self_attention':
            assert 'self_attention' in kwargs, 'Self attention parameters are not provided'
            assert 'cls_token' in kwargs, 'CLS token parameters are not provided'

            if kwargs['transformer_self_attention']['num_layers'] > 1:
                self.self_attention = LayerFactory.create(
                    'transformer_self_attention_block', **kwargs['self_attention'])

            else:
                self.self_attention = LayerFactory.create(
                    'transformer_self_attention', **kwargs['self_attention'])

            if kwargs['cls_token'] == True:
                self.cls_token = True
                self.cls_token_embedding = nn.Embedding(1, 1, embedding_dim)
            else:
                self.cls_token = False
        elif self.aggregation_method == 'weighted_sum':
            self.weight_matrix_ligand = torch.nn.Parameter(
                torch.randn(embedding_dim))
            self.weight_matrix_protein = torch.nn.Parameter(
                torch.randn(embedding_dim))

    def forward(self,  x1, x2):
        batch_size = x1.shape[0]

        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)

        if self.aggregation_method == 'concat':
            x1 = x1.view(batch_size, -1)
            x2 = x2.view(batch_size, -1)
            x = torch.cat([x1, x2], dim=1)

        elif self.aggregation_method == 'sum':
            x = x1 + x2

        elif self.aggregation_method == 'product':
            x = x1 * x2

        elif self.aggregation_method == 'average':
            x = (x1 + x2) / 2

        elif self.aggregation_method == 'max_pooling':
            x = torch.max(torch.stack([x1, x2]), dim=0)[0]

        elif self.aggregation_method == 'dot_product':
            x = torch.dot(x1.flatten(), x2.flatten())

        elif self.aggregation_method == 'weighted_sum':
            x = (x1 * self.weight_matrix_ligand) + \
                (x2 * self.weight_matrix_protein)

        elif self.aggregation_method == 'self_attention':
            if self.cls_token:
                cls_token = self.cls_token_embedding
                x1 = torch.cat([cls_token, x1, x2], dim=1)
                x = self.self_attention(x1, x2)
                x = x[:, 0, :]
            else:
                x = self.self_attention(x1, x2)
                x = torch.mean(x, dim=1)

        elif self.aggregation_method == 'cross_attention':
            x = self.cross_attention(x1, x2)
            x = torch.mean(x, dim=1)

        elif self.aggregation_method == 'cosine_similarity':
            x = torch.nn.functional.cosine_similarity(
                x1.unsqueeze(0), x2.unsqueeze(0))

        elif self.aggregation_method == 'custom':
            x = self.aggregation_module(x1, x2)

        x = self.head(x)
        return x

    def collate(self, *args, **kwargs) -> Any:
        return super().collate(*args, **kwargs)

    def preprocesses(self, *args, **kwargs) -> Any:
        return super().preprocesses(*args, **kwargs)

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

    def reset_head(self) -> None:
        self.head = LinearHead(**self.head_kwargs)

    def predict(self, *args, **kwargs) -> Any:
        return super().predict(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs) -> None:
        return super().save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs) -> None:
        return super().load_checkpoint(*args, **kwargs)
