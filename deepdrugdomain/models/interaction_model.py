from .base_model import BaseModel
from deepdrugdomain.layers import LayerFactory, LinearHead
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Optional, Type, Union, List
from deepdrugdomain.metrics import Evaluator
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from deepdrugdomain.schedulers import BaseScheduler
import numpy as np
from tqdm import tqdm


class BaseInteractionModel(BaseModel):
    """
    A base model for drug-target interaction prediction with various aggregation methods.

    This class implements a flexible and modular architecture for drug-target interaction 
    prediction, leveraging different types of neural network encoders and a variety of 
    aggregation methods to combine the encoded representations.

    Parameters:
    embedding_dim (int): Dimension of the embedding space.
    encoders (List[nn.Module]): List of neural network modules used as encoders.
    encoders_kwargs (List[Dict[str, Any]]): List of dictionaries containing keyword 
        arguments for each encoder in 'encoders'.
    head_kwargs (Dict[str, Any]): Keyword arguments for the final linear head of the model.
    aggregation_method (str): Method for aggregating the outputs of the encoders. Supported 
        methods include 'concat', 'sum', 'product', 'average', 'max_pooling', 'dot_product', 
        'weighted_sum', 'self_attention', 'cross_attention', and 'custom'.
    aggregation_module (Optional[Union[nn.Module, str]]): Module or string identifier for 
        the aggregation module, required if 'aggregation_method' is set to 'custom'.

    Attributes:
    encoders (List[nn.Module]): Instantiated encoder modules.
    head (LinearHead): The linear head module for final prediction.
    embedding_dim (int): The dimension of embeddings.
    aggregation_method (str): The chosen method for aggregating encoder outputs.
    aggregation_module (Optional[nn.Module]): The aggregation module (if any).

    Raises:
    AssertionError: If the specified aggregation method is not supported or required parameters 
        for certain aggregation methods are not provided.

    Methods:
    forward(*args): Forward pass through the model.
    collate(*args, **kwargs): Collates a batch of data.
    preprocesses(*args, **kwargs): Preprocesses data before feeding into the model.
    train_one_epoch(dataloader, device, criterion, optimizer, num_epochs, scheduler, evaluator, 
        grad_accum_steps, clip_grad, logger): Trains the model for one epoch.
    evaluate(dataloader, device, criterion, evaluator, logger): Evaluates the model.
    reset_head(): Resets the linear head of the model.
    predict(*args, **kwargs): Makes predictions using the model.
    save_checkpoint(*args, **kwargs): Saves the model checkpoint.
    load_checkpoint(*args, **kwargs): Loads a model checkpoint.
    """

    def __init__(self, embedding_dim: Optional[int], encoders: nn.ModuleList, head_kwargs: Dict[str, Any], aggregation_method: str, aggregation_module: Optional[Union[nn.Module, str]] = None, remove_head: bool = False, return_encoded: bool = False, *args, **kwargs):
        super(BaseInteractionModel, self).__init__()
        self.encoders = encoders
        self.head_kwargs = head_kwargs
        self.head = LinearHead(**head_kwargs)

        if remove_head:
            self.head = None

        self.return_encoded = return_encoded

        self.embedding_dim = embedding_dim
        self.aggregation_method = aggregation_method

        assert self.aggregation_method in ['concat', 'sum', 'product', 'average', 'max_pooling', 'dot_product',
                                           'weighted_sum', 'self_attention', 'cross_attention', 'custom'], 'Aggregation method not supported'
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
                assert 'embedding_dim' in kwargs, 'Embedding dimension is not provided'
                self.cls_token = True
                self.cls_token_embedding = nn.Embedding(1, 1, embedding_dim)
            else:
                self.cls_token = False
        elif self.aggregation_method == 'weighted_sum':
            assert embedding_dim is not None, 'Embedding dimension is not provided'
            self.weight_matrix = [torch.nn.Parameter(
                torch.randn(embedding_dim)) for _ in range(len(encoders))]
        

    def aggregate(self, encoded_inputs: List[torch.Tensor]) -> torch.Tensor:

        batch_size = encoded_inputs[0].shape[0]

        if self.aggregation_method == 'concat':
            encoded_inputs = [x.view(batch_size, -1) for x in encoded_inputs]
            x = torch.cat(encoded_inputs, dim=1)

        elif self.aggregation_method == 'sum':
            encoded_inputs = [x.view(batch_size, -1) for x in encoded_inputs]
            x = torch.sum(torch.stack(encoded_inputs),
                          dim=0).view(batch_size, -1)

        elif self.aggregation_method == 'product':
            encoded_inputs = [x.view(batch_size, -1) for x in encoded_inputs]
            x = torch.prod(torch.stack(encoded_inputs),
                           dim=0).view(batch_size, -1)

        elif self.aggregation_method == 'average':
            encoded_inputs = [x.view(batch_size, -1) for x in encoded_inputs]
            x = torch.mean(torch.stack(encoded_inputs),
                           dim=0).view(batch_size, -1)

        elif self.aggregation_method == 'max_pooling':
            encoded_inputs = [x.view(batch_size, -1) for x in encoded_inputs]
            x = torch.max(torch.stack(encoded_inputs),
                          dim=0).view(batch_size, -1)

        elif self.aggregation_method == 'dot_product':
            x1 = encoded_inputs[0].view(batch_size, -1)
            x2 = encoded_inputs[1].view(batch_size, -1)
            x = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)
                          ).view(batch_size, -1)

        elif self.aggregation_method == 'weighted_sum':
            encoded_inputs = [x.view(batch_size, -1) for x in encoded_inputs]
            x = torch.stack(encoded_inputs)
            x = torch.stack([torch.matmul(x[i], self.weight_matrix[i])
                             for i in range(len(x))])
            x = torch.sum(x, dim=0)

        elif self.aggregation_method == 'self_attention':
            x = torch.cat(encoded_inputs, dim=1)
            if self.cls_token:
                cls_token = self.cls_token_embedding
                x = torch.cat([cls_token, x], dim=1)
                x = self.self_attention(x)
                x = x[:, 0, :]
            else:
                x = self.self_attention(encoded_inputs)
                x = torch.mean(x, dim=1)

        elif self.aggregation_method == 'cross_attention':
            x = self.cross_attention(x1, x2)
            x = torch.mean(x, dim=1)

        elif self.aggregation_method == 'custom':
            x = self.aggregation_module(encoded_inputs)

        return x

    def forward(self,  *args):
        assert len(args) == len(
            self.encoders), 'Number of inputs must be the same as the number of encoders'

        x = [self.encoders[i](args[i]) for i in range(len(args))]

        if self.return_encoded and self.head is None:
            return x

        elif not self.return_encoded and self.head is None:
            x = self.aggregate(x)

        if self.head is not None:
            x = self.aggregate(x)
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
            for batch_idx, x in enumerate(t):
                last_batch = batch_idx == last_batch_idx
                need_update = last_batch or (batch_idx + 1) % accum_steps == 0
                update_idx = batch_idx // accum_steps

                if batch_idx >= last_batch_idx_to_accum:
                    accum_steps = last_accum_steps

                outs = []
                for item in range(len(x[0])):
                    inp = [x[i][item].unsqueeze(0).to(device)
                           for i in range(len(x) - 1)]
                    out = self.forward(*inp)

                    if isinstance(out, tuple):
                        out = out[0]

                    outs.append(out)

                out = torch.stack(outs, dim=0).squeeze(1)
                target = x[-1].to(
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
        return metrics

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
            for batch_idx, x in enumerate(t):
                outs = []
                with torch.no_grad():
                    for item in range(len(x[1])):
                        inp = [x[i][item].unsqueeze(0).to(device)
                               for i in range(len(x) - 1)]
                        out = self.forward(*inp)
                        if isinstance(out, tuple):
                            out = out[0]

                        outs.append(out)

                out = torch.stack(outs, dim=0).squeeze(1)
                target = x[-1].to(
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
