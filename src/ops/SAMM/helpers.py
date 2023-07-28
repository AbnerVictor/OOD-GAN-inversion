from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
from src.ops.StyleGAN.modules import Upsample, Blur
from src.ops.e4e.encoders.helpers import _upsample_add, BN, bottleneck_IR

from src.ops.StyleGAN.model import ModulatedConv2d, FusedLeakyReLU, StyledConv, Blur
from src.ops.StyleGAN.model import NoiseInjection

def pad_square(x, value=0):
    h, w = x.shape[-2:]
    length = h if h > w else w
    pad_h = (length - h) // 2
    pad_w = (length - w) // 2
    x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='constant', value=value)
    return x


class style_bottleneck_IR(Module):
    def __init__(self, in_channel, depth, style_dim, stride=1,
                 upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1],
                 demodulate=True, bn=False):
        super(style_bottleneck_IR, self).__init__()
        self.btn = nn.Sequential(bottleneck_IR(in_channel=in_channel, depth=in_channel, stride=stride, bn=bn),
                                 bottleneck_IR(in_channel=in_channel, depth=depth, stride=stride, bn=bn))
        self.final_conv = ModulatedConv2d(depth, depth, 3, style_dim, demodulate=demodulate,
                                          upsample=upsample, downsample=downsample, blur_kernel=blur_kernel)
        self.act = FusedLeakyReLU(depth)

    def forward(self, x, style):
        x = self.btn(x)
        res = self.final_conv(x, style)
        res = self.act(res)
        return res


class styleBlock(Module):
    def __init__(self, in_channel, depth, style_dim,
                 upsample=False, blur_kernel=[1, 3, 3, 1],
                 demodulate=True, noiseInjection=True, activation=False):
        super(styleBlock, self).__init__()
        self.conv1 = StyledConv(in_channel, depth, 3, style_dim,
                                demodulate=demodulate, upsample=upsample, blur_kernel=blur_kernel, noiseInjection=False,
                                activation=True)
        self.conv2 = StyledConv(depth, depth, 3, style_dim,
                                demodulate=demodulate, upsample=upsample, blur_kernel=blur_kernel,
                                noiseInjection=noiseInjection, activation=activation)

    def forward(self, x, style):
        res = self.conv1(x, style)
        res = self.conv2(res, style)
        return res


def scaleNshiftBlock(in_chn, out_chn, norm_type=False, bias=False):
    return nn.Sequential(bottleneck_IR(in_channel=in_chn, depth=in_chn, stride=1, bn=norm_type, bias=bias),
                         bottleneck_IR(in_channel=in_chn, depth=out_chn, stride=1, bn=norm_type, bias=bias))

def new_PRM(x, y, **kwargs):
    _, _, H, W = y.size()
    _, _, h, w = x.size()
    
    g = x
    
    if h != H or w != W:
        up_x = F.interpolate(x, size=(H, W), mode='bicubic', align_corners=True)
        up_g = F.interpolate(g, size=(H, W), mode='bicubic', align_corners=True)
    else:
        up_x = x
        up_g = g

    y_fused = (y * up_g) + (up_x * (1 - up_g))
    
    return y_fused

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class AlignNet(nn.Module):
    def __init__(self, in_chn, out_chn=3, scale=1., blur_kernel=[1, 3, 3, 1], **kwargs):
        super(AlignNet, self).__init__()
        self.norm = nn.InstanceNorm2d(in_chn)
        self.body = scaleNshiftBlock(in_chn * 2, out_chn, 'InstanceNorm', kwargs.get('bias', False))
        self.tanh = nn.Tanh()
        # self.relu6 = nn.ReLU6()
        self.sigmoid = nn.Sigmoid()
        self.scale = scale
        self.diff_fAndg = kwargs.get('diff_fAndg', True)

    def forward(self, source, target, **kwargs):
        source, target = self.norm(source), self.norm(target)
        if self.diff_fAndg:
            align = self.body(torch.cat([source - target, target], dim=1))
        else:
            align = self.body(torch.cat([source, target], dim=1))

        delta_x, delta_y, alpha = self.tanh(align[:, 0:1, ...]) * self.scale, \
                                  self.tanh(align[:, 1:2, ...]) * self.scale, \
                                  self.sigmoid(align[:, 2:, ...])

        align = torch.cat([delta_x, delta_y, alpha], dim=1)

        return align

class SPM_Warp(nn.Module):
    def __init__(self, in_chn, scale=0.1, style_dim=512,
                 blur_kernel=[1, 3, 3, 1], cycle_align=1, **kwargs):
        super(SPM_Warp, self).__init__()
        self.body = AlignNet(in_chn, 3, scale=scale, style_dim=style_dim, 
                            blur_kernel=blur_kernel, **kwargs)
        self.body.apply(weight_init)
        self.scale = scale
        self.cycle_align = cycle_align
        self.weight_init()

        self.blur = Blur(kernel=blur_kernel, pad=(2, 1))

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, (Conv2d, ModulatedConv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
    
    def add(self, aligned, align):
        delta_x, delta_y, alpha = aligned[:, 0:1, ...], aligned[:, 1:2, ...], aligned[:, 2:, ...]
        d_delta_x, d_delta_y, d_alpha = align[:, 0:1, ...], align[:, 1:2, ...], align[:, 2:, ...]

        delta_x = torch.clip((delta_x + d_delta_x), -self.scale, self.scale)
        delta_y = torch.clip((delta_y + d_delta_y), -self.scale, self.scale)
        alpha = torch.clip(new_PRM(x=alpha, y=d_alpha), 0., 1.)
        
        return torch.cat([delta_x, delta_y, alpha], dim=1)
    
    def upsample_add(self, aligned, align):
        delta_x, delta_y, alpha = aligned[:, 0:1, ...], aligned[:, 1:2, ...], aligned[:, 2:, ...]
        d_delta_x, d_delta_y, d_alpha = align[:, 0:1, ...], align[:, 1:2, ...], align[:, 2:, ...]

        delta_x = d_delta_x
        delta_y = d_delta_y
        alpha = torch.clip(new_PRM(x=alpha, y=d_alpha), 0., 1.)

        return torch.cat([delta_x, delta_y, alpha], dim=1)

    def forward(self, source, target, style=None, aligned=None):
        aligned_target = target
        aligned_ = None
        
        for k in range(self.cycle_align):
            align = self.blur(self.body(aligned_target, source))  # flow的输入改成source - target?              
            
            if aligned_ is not None:
                aligned_ = self.add(aligned_, align)
            else:
                aligned_ = align
            
            if k == self.cycle_align - 1:
                if aligned is not None:
                    aligned_ = self.upsample_add(aligned, aligned_)
                else:
                    aligned_ = aligned_
                    
            # transform
            delta_x, delta_y, alpha = aligned_[:, 0, ...], aligned_[:, 1, ...], aligned_[:, 2:, ...]

            ind_y, ind_x = torch.meshgrid(torch.linspace(-1, 1, source.shape[2], device=delta_x.device),
                                        torch.linspace(-1, 1, source.shape[3], device=delta_y.device))
        
            grid = torch.stack([ind_x.unsqueeze(0) + delta_x, ind_y.unsqueeze(0) + delta_y], dim=-1)
        
            aligned_target = F.grid_sample(target, grid)  
                  
            aligned_target = aligned_target * alpha + (target * (1 - alpha))
                    
        return aligned_target, aligned_


class StyledscaleNshfitBlock(nn.Module):
    def __init__(self, in_chn, out_chn, style_dim, alignment=True,
                 btn='style_bottleneck_IR', **kwargs):
        super(StyledscaleNshfitBlock, self).__init__()
        if btn == 'style_bottleneck_IR':
            self.btn1 = style_bottleneck_IR(in_channel=in_chn, depth=out_chn, style_dim=style_dim, bn=False)
        elif btn == 'styleBlock':
            self.btn1 = styleBlock(in_channel=in_chn, depth=out_chn, style_dim=style_dim, noiseInjection=False,
                                   activation=False)
        else:
            self.btn1 = lambda x, y: x
            out_chn = in_chn

        if alignment:
            self.alignment = SPM_Warp(out_chn, **kwargs)
        else:
            self.alignment = lambda x, y, z: (x, None)

        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)

        self.noiseInj = NoiseInjection()

    def forward(self, x, styles, **kwargs):
        res = self.btn1(x, styles)  # CNN modulated feature extraction

        transform = kwargs.get('transform', lambda x: x)
        misalign_res = transform(res)

        gen_feat = kwargs.get('image', None)
        assert gen_feat is not None

        aligned = kwargs.get('aligned', None)
        aligned_res, align = self.alignment(misalign_res, gen_feat, styles, aligned)

        return aligned_res, align