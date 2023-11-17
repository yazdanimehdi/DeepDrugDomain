"""
This module provides wrapper classes for PyTorch normalization layers.
Each of these classes is registered in the `LayerFactory` for unified layer management.
"""

import torch.nn as nn
from ..utils import LayerFactory


@LayerFactory.register('batch_norm1d')
class BatchNorm1D(nn.BatchNorm1d):
    """
    A wrapper class for PyTorch's BatchNorm1d.
    It provides batch normalization operation specifically designed for 1D input data.
    """
    pass


@LayerFactory.register('batch_norm2d')
class BatchNorm2D(nn.BatchNorm2d):
    """
    A wrapper class for PyTorch's BatchNorm2d.
    It provides batch normalization operation specifically designed for 2D input data (typically used in CNNs).
    """
    pass


@LayerFactory.register('batch_norm3d')
class BatchNorm3D(nn.BatchNorm3d):
    """
    A wrapper class for PyTorch's BatchNorm3d.
    It provides batch normalization operation specifically designed for 3D input data (typically used in 3D CNNs).
    """
    pass


@LayerFactory.register('instance_norm1d')
class InstanceNorm1D(nn.InstanceNorm1d):
    """
    A wrapper class for PyTorch's InstanceNorm1d.
    It provides instance normalization operation for 1D input data.
    """
    pass


@LayerFactory.register('instance_norm2d')
class InstanceNorm2D(nn.InstanceNorm2d):
    """
    A wrapper class for PyTorch's InstanceNorm2d.
    It provides instance normalization operation for 2D input data (often used in style transfer tasks).
    """
    pass


@LayerFactory.register('instance_norm3d')
class InstanceNorm3D(nn.InstanceNorm3d):
    """
    A wrapper class for PyTorch's InstanceNorm3d.
    It provides instance normalization operation for 3D input data.
    """
    pass


@LayerFactory.register('layer_norm')
class LayerNorm(nn.LayerNorm):
    """
    A wrapper class for PyTorch's LayerNorm.
    It normalizes the activations of the layer for each given example in a batch independently, rather than across a batch.
    """
    pass


@LayerFactory.register('group_norm')
class GroupNorm(nn.GroupNorm):
    """
    A wrapper class for PyTorch's GroupNorm.
    It divides channels into groups and computes within each group the mean and variance for normalization.
    """
    pass


@LayerFactory.register('local_response_norm')
class LocalResponseNorm(nn.LocalResponseNorm):
    """
    A wrapper class for PyTorch's LocalResponseNorm.
    It implements the Local Response Normalization, a form of normalization often used in early CNN architectures.
    """
    pass
