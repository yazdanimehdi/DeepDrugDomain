import abc
from abc import ABC
from typing import Any, Dict, Optional

import torch


class BaseScheduler(ABC):
    """
    A scheduler base class in PyTorch for updating optimizer parameters at the end of each epoch or optimizer update.

    This abstract base class provides a structured way to implement custom learning rate schedules or other optimizer
    parameter updates. It ensures that all derived schedulers perform updates consistently either at the end of each epoch
    or after each optimizer step, based on the provided schedule.

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer for which parameters are scheduled.
        param_group_field (str): The name of the parameter within the optimizer to update.
        t_in_epochs (bool): If True, updates are based on epoch progression; otherwise, on optimizer updates.
        noise_range_t (Optional[tuple or int]): Epoch/update number range or threshold to start applying noise.
        noise_type (str): Type of noise to apply ('normal' or other types assumed to be uniform noise).
        noise_pct (float): The percentage limit of the noise magnitude relative to the parameter's value.
        noise_std (float): The standard deviation of the normal noise to apply.
        noise_seed (Optional[int]): Random seed for noise generation; default is 42 if not specified.
        initialize (bool): If True, initializes the param_group_field for each param group on instantiation.

    Examples:
        >>> class MyScheduler(BaseScheduler):
        ...     def _get_lr(self, t):
        ...         # Define how the learning rate should be calculated here
        ...         return super()._get_lr(t)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = MyScheduler(optimizer, param_group_field='lr', t_in_epochs=True)
        >>> scheduler.step(epoch=1)  # Update at the end of an epoch
        >>> scheduler.step_update(num_updates=1)  # Update after an optimizer step

    Note:
        - The '_get_lr' method must be implemented by all subclasses.
        - The 'step' method should be called at the end of each epoch.
        - The 'step_update' method should be called after each optimizer step.
        - Noise can be added to the parameter updates based on the configured noise parameters.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            param_group_field: str = 'lr',
            t_in_epochs: bool = False,
            noise_range_t=None,
            noise_type='normal',
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=None,
            initialize: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(
                        f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field,
                                 group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(
                        f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field]
                            for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    @abc.abstractmethod
    def _get_lr(self, t: int) -> float:
        pass

    def _get_values(self, t: int, on_epoch: bool = True) -> Optional[float]:
        proceed = (on_epoch and self.t_in_epochs) or (
            not on_epoch and not self.t_in_epochs)
        if not proceed:
            return None
        return self._get_lr(t)

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self._get_values(epoch, on_epoch=True)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self._get_values(num_updates, on_epoch=False)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            if 'lr_scale' in param_group:
                param_group[self.param_group_field] = value * \
                    param_group['lr_scale']
            else:
                param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lrs = [v + v * noise for v in lrs]
        return lrs

    def _is_apply_noise(self, t) -> bool:
        """Return True if scheduler in noise range."""
        apply_noise = False
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
        return apply_noise

    def _calculate_noise(self, t) -> float:
        g = torch.Generator()
        g.manual_seed(self.noise_seed + t)
        if self.noise_type == 'normal':
            while True:
                # resample if noise out of percent limit, brute force but shouldn't spin much
                noise = torch.randn(1, generator=g).item()
                if abs(noise) < self.noise_pct:
                    return noise
        else:
            noise = 2 * (torch.rand(1, generator=g).item() -
                         0.5) * self.noise_pct
        return noise
