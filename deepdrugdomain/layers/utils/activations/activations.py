"""
This module provides a collection of activation functions implemented as classes.
Each class offers two methods:
    - `activation_layer`: which returns the corresponding PyTorch layer/module.
    - `activation_function`: which returns the corresponding PyTorch function.
    
Classes are registered with the `ActivationFactory` using a decorator syntax,
allowing for easy extension and retrieval.
"""

from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from .activation_factory import ActivationFactory
from .base_activation import BaseActivation


@ActivationFactory.register('relu')
class Relu(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.ReLU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.relu, *args, **kwargs)


@ActivationFactory.register('leaky_relu')
class LeakyRelu(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.LeakyReLU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.leaky_relu, *args, **kwargs)


@ActivationFactory.register('sigmoid')
class Sigmoid(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Sigmoid(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.sigmoid, *args, **kwargs)


@ActivationFactory.register('tanh')
class Tanh(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Tanh(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.tanh, *args, **kwargs)


@ActivationFactory.register('softmax')
class Softmax(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Softmax(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.softmax, *args, **kwargs)


@ActivationFactory.register('log_softmax')
class LogSoftmax(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.LogSoftmax(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.log_softmax, *args, **kwargs)


@ActivationFactory.register('elu')
class ELU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.ELU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.elu, *args, **kwargs)


@ActivationFactory.register('selu')
class SELU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.SELU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.selu, *args, **kwargs)


@ActivationFactory.register('softplus')
class Softplus(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Softplus(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.softplus, *args, **kwargs)


@ActivationFactory.register('softsign')
class Softsign(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Softsign(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.softsign, *args, **kwargs)


@ActivationFactory.register('hardtanh')
class Hardtanh(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Hardtanh(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.hardtanh, *args, **kwargs)


@ActivationFactory.register('celu')
class CELU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.CELU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.celu, *args, **kwargs)


@ActivationFactory.register('glu')
class GLU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.GLU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.glu, *args, **kwargs)


@ActivationFactory.register('gelu')
class GELU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.GELU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.gelu, *args, **kwargs)


@ActivationFactory.register('hardshrink')
class Hardshrink(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Hardshrink(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.hardshrink, *args, **kwargs)


@ActivationFactory.register('hardsigmoid')
class Hardsigmoid(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Hardsigmoid(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.hardsigmoid, *args, **kwargs)


@ActivationFactory.register('hardswish')
class Hardswish(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Hardswish(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.hardswish, *args, **kwargs)


@ActivationFactory.register('logsigmoid')
class LogSigmoid(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.LogSigmoid(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.logsigmoid, *args, **kwargs)


@ActivationFactory.register('softmin')
class Softmin(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Softmin(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.softmin, *args, **kwargs)


@ActivationFactory.register('softshrink')
class Softshrink(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Softshrink(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.softshrink, *args, **kwargs)


@ActivationFactory.register('threshold')
class Threshold(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Threshold(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.threshold, *args, **kwargs)


@ActivationFactory.register('rrelu')
class RReLU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.RReLU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.rrelu, *args, **kwargs)


@ActivationFactory.register('prelu')
class PRelu(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.PReLU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.prelu, *args, **kwargs)


@ActivationFactory.register('mish')
class Mish(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Mish(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.mish, *args, **kwargs)


@ActivationFactory.register('tanhshrink')
class Tanhshrink(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Tanhshrink(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.tanhshrink, *args, **kwargs)


@ActivationFactory.register('softmax2d')
class Softmax2d(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Softmax2d(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.softmax2d, *args, **kwargs)


@ActivationFactory.register('log_softmax2d')
class LogSoftmax2d(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.LogSoftmax2d(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.log_softmax2d, *args, **kwargs)


@ActivationFactory.register('adaptive_log_softmax_with_loss')
class AdaptiveLogSoftmaxWithLoss(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.AdaptiveLogSoftmaxWithLoss(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.adaptive_log_softmax, *args, **kwargs)


@ActivationFactory.register('Relu6')
class Relu6(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.ReLU6(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.relu6, *args, **kwargs)


@ActivationFactory.register('silu')
class SiLU(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.SiLU(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.silu, *args, **kwargs)


@ActivationFactory.register('hard_mish')
class HardMish(BaseActivation):

    @staticmethod
    def activation_layer(*args, **kwargs) -> nn.Module:
        return nn.Hardswish(*args, **kwargs)

    @staticmethod
    def activation_function(*args, **kwargs) -> Callable:
        return partial(torch.hardswish, *args, **kwargs)
