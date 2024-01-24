"""
This module provides a factory pattern implementation for creating scheduler objects for machine learning training.
The SchedulerFactory class, enables the registration and instantiation of scheduler types.
Schedulers are derived from a BaseScheduler class, ensuring compatibility and consistent behavior across different types
of schedulers used for training optimization algorithms.
The SchedulerFactory allows for the dynamic association between string keys and scheduler subclasses, facilitating
easy configuration and usage in different training contexts.
Example:
    >>> # Assuming you have a custom scheduler that inherits from BaseScheduler:
    >>> class MyCustomScheduler(BaseScheduler):
    ...     def __init__(self, optimizer, last_epoch=-1):
    ...         super().__init__(optimizer, last_epoch)
    ...
    >>> # Register your custom scheduler with the factory:
    >>> SchedulerFactory.register('my_custom')(MyCustomScheduler)
    >>> # Now you can create an instance of your custom scheduler:
    >>> my_scheduler = SchedulerFactory.create('my_custom', optimizer=my_optimizer)
This example demonstrates how to define a new scheduler class that inherits from BaseScheduler, register it with
the factory, and then create an instance of it using the factory's create method.
"""


from typing import Dict, Type, TypeVar, List, Union
from deepdrugdomain.utils import BaseFactory
from .base_scheduler import BaseScheduler
from torch.optim import Optimizer


T = TypeVar('T', bound=BaseScheduler)


class SchedulerFactory(BaseFactory):
    """
    A factory class for creating instances of different types of schedulers.
    This class provides a mechanism for registering new scheduler classes and creating instances of those classes
    based on a provided key.
    Attributes:
        _registry (Dict[str, Type[BaseScheduler]]): A dictionary mapping keys to registered scheduler classes.
    """

    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, key: str):
        """
        A decorator function for registering a new scheduler class.
        Args:
            key (str): A unique key to associate with the registered scheduler class.
        Returns:
            A decorator function that takes a subclass of BaseScheduler as an argument and registers it with the factory.
        """
        def decorator(subclass: Type[BaseScheduler]) -> Type[BaseScheduler]:
            if not issubclass(subclass, BaseScheduler):
                raise TypeError(
                    f"Class {subclass.__name__} is not a subclass of BaseScheduler.")
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, key: str,
               optimizer: Optimizer,
               num_epochs: int = 300,
               decay_epochs: int = 90,
               decay_milestones: List[int] = (90, 180, 270),
               cooldown_epochs: int = 0,
               patience_epochs: int = 10,
               decay_rate: float = 0.1,
               min_lr: float = 0,
               warmup_lr: float = 1e-5,
               warmup_epochs: int = 0,
               warmup_prefix: bool = False,
               noise: Union[float, List[float]] = None,
               noise_pct: float = 0.67,
               noise_std: float = 1.,
               noise_seed: int = 42,
               cycle_mul: float = 1.,
               cycle_decay: float = 0.1,
               cycle_limit: int = 1,
               k_decay: float = 1.0,
               plateau_mode: str = 'max',
               step_on_epochs: bool = True,
               updates_per_epoch: int = 0) -> Type[T]:
        """
        Create a new instance of a scheduler based on the given key.
        Args:
            key (str): The key of the scheduler to create.
            *args: Additional positional arguments to pass to the scheduler constructor.
            **kwargs: Additional keyword arguments to pass to the scheduler constructor.
        Returns:
            Type[T]: An instance of the scheduler with the given key.
        Raises:
            ValueError: If the given key is not registered.
        """
        if key not in cls._registry:
            raise ValueError(f"Key '{key}' not registered.")
        t_initial = num_epochs
        warmup_t = warmup_epochs
        decay_t = decay_epochs
        cooldown_t = cooldown_epochs

        if not step_on_epochs:
            assert updates_per_epoch > 0, 'updates_per_epoch must be set to number of dataloader batches'
            t_initial = t_initial * updates_per_epoch
            warmup_t = warmup_t * updates_per_epoch
            decay_t = decay_t * updates_per_epoch
            decay_milestones = [
                d * updates_per_epoch for d in decay_milestones]
            cooldown_t = cooldown_t * updates_per_epoch

        warmup_lr_init = warmup_lr
        warmup_t = warmup_t
        warmup_prefix = warmup_prefix

        # setup noise args for supporting schedulers
        if noise is not None:
            if isinstance(noise, (list, tuple)):
                noise_range = [n * t_initial for n in noise]
                if len(noise_range) == 1:
                    noise_range = noise_range[0]
            else:
                noise_range = noise * t_initial
        else:
            noise_range = None

        noise_range_t = noise_range,
        noise_pct = noise_pct,
        noise_std = noise_std,
        noise_seed = noise_seed,

        # setup cycle args for supporting schedulers

        kwargs = {
            'num_epochs': num_epochs,
            'decay_epochs': decay_epochs,
            'decay_milestones': decay_milestones,
            'cooldown_epochs': cooldown_epochs,
            'patience_epochs': patience_epochs,
            'decay_rate': decay_rate,
            'min_lr': min_lr,
            'warmup_lr': warmup_lr,
            'warmup_epochs': warmup_epochs,
            'warmup_prefix': warmup_prefix,
            'noise': noise,
            'noise_pct': noise_pct,
            'noise_std': noise_std,
            'noise_seed': noise_seed,
            'cycle_mul': cycle_mul,
            'cycle_decay': cycle_decay,
            'cycle_limit': cycle_limit,
            'k_decay': k_decay,
            'plateau_mode': plateau_mode,
            'step_on_epochs': step_on_epochs,
            'updates_per_epoch': updates_per_epoch,
            't_initial': t_initial,
            'warmup_t': warmup_t,
            'decay_t': decay_t,
            'cooldown_t': cooldown_t,
            'warmup_lr_init': warmup_lr_init,
            'noise_range_t': None,
        }
        scheduler_instance = cls._registry[key](optimizer, **kwargs)

        return scheduler_instance
