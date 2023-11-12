from torch.optim import Optimizer
import torch.optim as optim
from .factory import OptimizerFactory


def register_all_optimizers():
    """
    Registers all optimizer classes available in the torch.optim module to the OptimizerFactory.

    This function iterates over all attributes in the torch.optim module, checks for classes that are 
    a subclass of torch.optim.Optimizer, and adds them to the factory's registry. Each optimizer is 
    registered with a key corresponding to its lowercase class name.

    This registration process allows for dynamic instantiation of optimizers based on string keys, 
    which is particularly useful for configurations in machine learning experiments that require 
    flexible optimizer usage.

    Example:
        >>> # After calling this, optimizers can be created via the factory using string keys:
        >>> from my_optimizer_factory_module import OptimizerFactory
        >>> sgd_optimizer = OptimizerFactory.create('sgd', lr=0.01, momentum=0.9)
        >>> adam_optimizer = OptimizerFactory.create('adam', lr=0.001)
    """
    for optimizer_name in dir(optim):
        # Ensure we're only getting classes defined in torch.optim, which are Optimizer subclasses
        optimizer_cls = getattr(optim, optimizer_name)
        if isinstance(optimizer_cls, type) and issubclass(optimizer_cls, Optimizer):
            OptimizerFactory.register(optimizer_name.lower())(optimizer_cls)
