"""
This module was lifted from https://github.com/huggingface/pytorch-image-models/tree/main/timm/scheduler
Originally licensed Apache-2.0 license, Ross Wightman. Thanks Ross!
"""
from .factory import SchedulerFactory
from .base_scheduler import BaseScheduler
from .cosine_lr import *
from .plateau_lr import *
from .step_lr import *
from .tanh_lr import *
from .poly_lr import *
from .multistep_lr import *
