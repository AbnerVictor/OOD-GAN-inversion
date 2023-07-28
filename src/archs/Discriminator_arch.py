from basicsr.utils.registry import ARCH_REGISTRY

import math
import numpy as np
import random
import functools
import operator
import itertools
import easydict
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from basicsr.archs.arch_util import trunc_normal_
from src.ops.StyleGAN.modules import EqualLinear, PixelNorm
from src.ops.StyleGAN.stylegan2_arch import StyleGAN2Discriminator

@ARCH_REGISTRY.register()
class StyleGAN2Discriminator_mod(StyleGAN2Discriminator):
    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator_mod, self).__init__(out_size, channel_multiplier, resample_kernel, stddev_group, narrow)


@ARCH_REGISTRY.register()
class LatentDiscrinimator(nn.Module):
    def __init__(self, chn=18, dim=512, n_mlp=8, hidden_chn=1):
        super(LatentDiscrinimator, self).__init__()
        self.first_linear = EqualLinear(chn, hidden_chn, bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu')

        layers = [EqualLinear(hidden_chn * dim, dim, bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu')]

        for i in range(n_mlp):
            layers.append(EqualLinear(dim, dim, bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'))

        self.layers = nn.Sequential(*layers)

        self.final_linear = EqualLinear(dim, 1, bias=True, bias_init_val=0, lr_mul=1, activation=None)

    def forward(self, x):
        b, c, n = x.shape
        x = self.first_linear(x.permute(0, 2, 1).reshape(-1, c)).reshape(b, n, -1).permute(0, 2, 1).reshape(b, -1)
        x = self.layers(x)
        return self.final_linear(x), None

