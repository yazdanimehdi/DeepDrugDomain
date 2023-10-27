from functools import partial
import warnings
from .utils import LayerFactory, drop_path
from torch import nn


@LayerFactory.register('drop_path')
class DropPath(nn.Module):
    """
    Implements Drop paths (Stochastic Depth) per sample. Applied in the main path of residual blocks.

    Args:
        drop_prob (float, optional): Probability of dropping a path. Defaults to 0.
        scale_by_keep (bool, optional): If set to True, scales by the keep probability. Defaults to True.
    """

    def __init__(self, drop_prob: float, **kwargs):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        defaults = {
            "scale_by_keep": True,
        }
        # Fill in missing parameters with defaults
        for key, default_val in defaults.items():
            kwargs.setdefault(key, default_val)
            if key not in kwargs:
                warnings.warn(
                    f"'{key}' parameter is missing. Using default value '{default_val}' for the '{self.__class__.__name__}' layer.")

        self.scale_by_keep = kwargs["scale_by_keep"]

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
