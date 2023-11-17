from torch import nn
from torch.nn import functional as F
from typing import Optional, Type, Union, Tuple, List
from ..utils import to_2tuple, pad_same, LayerFactory


@LayerFactory.register('avgpool2d_same')
class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: int, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size,
                                            stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)


@LayerFactory.register('resnet_downsample_conv')
class DownsampleConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 first_dilation: Optional[int] = None,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 ) -> None:
        norm_layer = norm_layer or nn.BatchNorm2d
        kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
        first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
        p = get_padding(kernel_size, stride, first_dilation)

        self.layer = nn.Sequential(*[nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
            norm_layer(out_channels)
        ])

    def forward(self, x):
        return self.layer(x)


@LayerFactory.register('resnet_downsample_avg')
class DownsampleAvg(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 first_dilation: Optional[int] = None,
                 norm_layer: Optional[Type[nn.Module]] = None,
                 ) -> None:
        norm_layer = norm_layer or nn.BatchNorm2d
        avg_stride = stride if dilation == 1 else 1
        if stride == 1 and dilation == 1:
            pool = nn.Identity()
        else:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            pool = avg_pool_fn(2, avg_stride, ceil_mode=True,
                               count_include_pad=False)

        self.layer = nn.Sequential(*[
            pool,
            nn.Conv2d(in_channels, out_channels, 1,
                      stride=1, padding=0, bias=False),
            norm_layer(out_channels)
        ])

    def forward(self, x):
        return self.layer(x)


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)
