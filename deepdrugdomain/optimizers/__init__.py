"""
This module custom optimizers was lifted from https://github.com/huggingface/pytorch-image-models/tree/main/timm/optim
Originally licensed Apache-2.0 license, Ross Wightman. Thanks Ross!
"""
from .adabelief import *
from .adafactor import *
from .adahessian import *
from .adamp import *
from .adan import *
from .lamb import *
from .lars import *
from .lion import *
from .lookahead import *
from .madgrad import *
from .nadam import *
from .nadamw import *
from .nvnovograd import *
from .radam import *
from .rmsprop_tf import *
from .sgdp import *
from .torch_optimizers import register_all_optimizers

register_all_optimizers()
