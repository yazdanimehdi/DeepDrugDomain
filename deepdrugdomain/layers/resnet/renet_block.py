"""
PyTorch ResNet Basic Block

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool. 
This block was taken from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py by Ross Wightman. Thanks Ross!
"""

import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepdrugdomain.layers import LayerFactory, ActivationFactory
from .resnet_layers import create_aa


@LayerFactory.register('resnet_basic_block')
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size_1: int = 3,
            kernel_size_2: int = 3,
            stride_1: int = 1,
            stride_2: int = 1,
            stride: int = 1,
            downsample: Optional[str] = None,
            downsample_kwargs: Optional[Dict[str, Any]] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: str = "relu",
            norm_layer: str = "batchnorm2d",
            attn_layer: Optional[str] = None,
            attn_layer_kwargs: Optional[Dict[str, Any]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[str] = None,
            drop_path: float = 0.0,
    ):
        """
        Args:
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (
            stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=kernel_size_1, stride=stride_1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = ActivationFactory.create(act_layer, inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes,
                            stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=kernel_size_2, stride=stride_2, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = LayerFactory.create(norm_layer, outplanes)

        self.se = LayerFactory.create(
            attn_layer, outplanes, **attn_layer_kwargs) if attn_layer else None

        self.act2 = ActivationFactory.create(act_layer, inplace=True)
        self.downsample = LayerFactory.create(
            downsample, inplanes, outplanes, stride, **downsample_kwargs)

        self.stride = stride
        self.dilation = dilation
        self.drop_path = LayerFactory.create(
            "drop_path", drop_path) if drop_path > 0. else nn.Identity()

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x
