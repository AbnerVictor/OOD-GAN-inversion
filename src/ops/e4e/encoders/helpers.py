from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
from src.ops.StyleGAN.modules import Upsample, Blur

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# class BN(Module):
#     def __init__(self, depth, bn=True):
#         super(BN, self).__init__()
#         if bn == 'InstanceNorm':
#             self.bn = nn.InstanceNorm2d(depth)
#         if bn == 'BatchNorm' or bn is True:
#             self.bn = nn.BatchNorm2d(depth)
#         else:
#             self.bn = nn.Identity()

#     def forward(self, x):
#         res = self.bn(x)
#         return res

def BN(depth, bn=True):
    if bn == 'InstanceNorm':
        return nn.InstanceNorm2d(depth, affine=True)
    if bn == 'BatchNorm' or bn is True:
        return nn.BatchNorm2d(depth)
    else:
        return nn.Identity()


from src.ops.StyleGAN.model import ModulatedConv2d, FusedLeakyReLU, StyledConv, Blur


def scaleNshiftBlock(in_chn, out_chn, norm_type=False, deform=False, equal=False, bias=False):
    if deform:
        return nn.Sequential(bottleneck_IR_Deform(in_channel=in_chn, depth=in_chn, stride=1, bn=norm_type),
                             bottleneck_IR_Deform(in_channel=in_chn, depth=out_chn, stride=1, bn=norm_type))
    elif not equal:
        return nn.Sequential(bottleneck_IR(in_channel=in_chn, depth=in_chn, stride=1, bn=norm_type),
                             bottleneck_IR(in_channel=in_chn, depth=out_chn, stride=1, bn=norm_type))
    elif equal:
        return nn.Sequential(bottleneck_IR_Equal(in_channel=in_chn, depth=in_chn, stride=1, bn=norm_type, bias=bias),
                             bottleneck_IR_Equal(in_channel=in_chn, depth=out_chn, stride=1, bn=norm_type, bias=bias))


class AlignNet(nn.Module):
    def __init__(self, in_chn, out_chn=3, scale=1., blur_kernel=[1, 3, 3, 1], **kwargs):
        super(AlignNet, self).__init__()
        self.norm = nn.InstanceNorm2d(in_chn)
        self.body = scaleNshiftBlock(in_chn * 2, out_chn, 'InstanceNorm')
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

        # strength = torch.norm(torch.stack([delta_x, delta_y], dim=1), dim=1) / np.sqrt(2 * (self.scale**2))

        align = torch.cat([delta_x, delta_y, alpha], dim=1)

        return align


def align_upsample_add(aligned, align, scale=1., RPM_th=(0.0001, 0.1, 0.2), RPM_up=None, indFlow=True):
    delta_x, delta_y, alpha = aligned[:, 0:1, ...], aligned[:, 1:2, ...], aligned[:, 2:, ...]
    d_delta_x, d_delta_y, d_alpha = align[:, 0:1, ...], align[:, 1:2, ...], align[:, 2:, ...]

    delta_x = d_delta_x
    delta_y = d_delta_y
    alpha = torch.clip(new_PRM(x=alpha, y=d_alpha), 0., 1.)
    # alpha = torch.clip(_RPM(x=alpha, y=d_alpha, th=RPM_th[0], th2=RPM_th[1], th3=RPM_th[2], up=RPM_up), 0., 1.)
    # delta_x = torch.clip(_upsample_add(delta_x, d_delta_x), -scale, scale)
    # delta_y = torch.clip(_upsample_add(delta_y, d_delta_y), -scale, scale)
    # alpha = d_alpha

    return torch.cat([delta_x, delta_y, alpha], dim=1)


class SPM_Warp(nn.Module):
    def __init__(self, in_chn, scale=0.1, style_dim=512,
                 blur_kernel=[1, 3, 3, 1], **kwargs):
        super(SPM_Warp, self).__init__()
        self.body = AlignNet(in_chn, 3, scale=scale, style_dim=style_dim, blur_kernel=blur_kernel, diff_fAndg=kwargs.get('diff_fAndg', True))
        self.scale = scale
        self.cycle_align = kwargs.get('cycle_align', 3)
        self.RPM_th = kwargs.get('RPM_th', (0.0001, 0.1, 0.2))
        self.weight_init()

        # borrowed from StyleGAN2
        if kwargs.get('RPM_up', False):
            self.upsample = Upsample(kernel=blur_kernel, factor=2)
        else:
            self.upsample = None
        
        self.blur = Blur(kernel=blur_kernel, pad=(2, 1))

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, (Conv2d, ModulatedConv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)

    # def forward(self, source, target, style=None, aligned=None):
    #     align = self.blur(self.body(target, source))  # flow的输入改成source - target

    #     if aligned is not None:
    #         aligned_ = align_upsample_add(aligned, align, self.scale, self.RPM_th, self.upsample, self.indFlow)
    #     else:
    #         aligned_ = align

    #     # transform
    #     delta_x, delta_y, alpha = align[:, 0, ...], align[:, 1, ...], align[:, 2:, ...]

    #     ind_y, ind_x = torch.meshgrid(torch.linspace(-1, 1, source.shape[2], device=delta_x.device),
    #                                   torch.linspace(-1, 1, source.shape[3], device=delta_y.device))
        
    #     grid = torch.stack([ind_x.unsqueeze(0) + delta_x, ind_y.unsqueeze(0) + delta_y], dim=-1)
    #     warped_target = F.grid_sample(target, grid)
    #     aligned_target = warped_target * alpha + (target * (1 - alpha))
        
    #     # cycle alignment
    #     cycle_align = self.blur(self.body(aligned_target, source))

    #     return aligned_target, cycle_align
    
    def add(self, aligned, align):
        delta_x, delta_y, alpha = aligned[:, 0:1, ...], aligned[:, 1:2, ...], aligned[:, 2:, ...]
        d_delta_x, d_delta_y, d_alpha = align[:, 0:1, ...], align[:, 1:2, ...], align[:, 2:, ...]

        delta_x = torch.clip((delta_x + d_delta_x), -self.scale, self.scale)
        delta_y = torch.clip((delta_y + d_delta_y), -self.scale, self.scale)
        # alpha = torch.clip(d_alpha * alpha + alpha * (1 - alpha), 0., 1.)
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


class SPM_Slim(nn.Module):
    def __init__(self, in_chn, scale=0.1, style_dim=512, **kwargs):
        super(SPM_Slim, self).__init__()
        self.body = AlignNet(in_chn, 3, style_dim=style_dim)
        self.refine = scaleNshiftBlock(in_chn, in_chn)
        self.scale = scale
        self.tanh = nn.Tanh()
        self.relu6 = nn.ReLU6()
        self.weight_init()

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, (Conv2d, ModulatedConv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)

    def forward(self, source, target, style=None, aligned=None):
        align = self.body(source, target, style=style)  # flow的输入改成source - target

        if aligned is not None:
            aligned = align_upsample_add(aligned, align)
        else:
            aligned = align

        # transform
        delta_x, delta_y, alpha = self.tanh(aligned[:, 0, ...]) * self.scale, \
                                  self.tanh(aligned[:, 1, ...]) * self.scale, \
                                  self.relu6(aligned[:, 2:, ...]) / 6

        ind_y, ind_x = torch.meshgrid(torch.linspace(-1, 1, source.shape[2], device=delta_x.device),
                                      torch.linspace(-1, 1, source.shape[3], device=delta_y.device))

        grid = torch.stack([ind_x.unsqueeze(0) + delta_x, ind_y.unsqueeze(0) + delta_y], dim=-1)

        warped_source = F.grid_sample(source, grid)

        aligned_source = self.refine(warped_source - target) * alpha + ((warped_source - target) * (1 - alpha))

        return aligned_source, aligned


class SPM(nn.Module):
    def __init__(self, in_chn, scale=0.1, style_dim=512, **kwargs):
        super(SPM, self).__init__()
        self.body = AlignNet(in_chn, style_dim=style_dim)
        self.refine = scaleNshiftBlock(in_chn * 2, in_chn)
        self.scale = scale
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, source, target, style=None, aligned=None):
        align = self.body(source, target)  # flow的输入改成source - target, delta w

        if aligned is not None:
            aligned = align_upsample_add(aligned, align)
        else:
            aligned = align

        # transform
        delta_x, delta_y, alpha = aligned[:, 0, ...], aligned[:, 1, ...], aligned[:, 2:, ...]
        delta_x, delta_y = self.tanh(delta_x) * self.scale, self.tanh(delta_y) * self.scale
        alpha = self.sigmoid(alpha)
        ind_y, ind_x = torch.meshgrid(torch.linspace(-1, 1, source.shape[2], device=delta_x.device),
                                      torch.linspace(-1, 1, source.shape[3], device=delta_y.device))
        grid = torch.stack([ind_x.unsqueeze(0) + delta_x, ind_y.unsqueeze(0) + delta_y], dim=-1)

        warped_source = F.grid_sample(source, grid)
        aligned_source = self.refine(torch.cat([warped_source, target], dim=1)) * alpha \
                         + (warped_source * (1 - alpha))

        return aligned_source, aligned


class SPM_Legacy(nn.Module):
    def __init__(self, in_chn, scale=0.1, **kwargs):
        super(SPM_Legacy, self).__init__()
        self.body = scaleNshiftBlock(in_chn * 2, 3, 'InstanceNorm')  # our = delta x, y, alpha
        self.refine = scaleNshiftBlock(in_chn * 2, in_chn)
        self.scale = scale
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, source, target, aligned=None):
        align = self.body(torch.cat([source, target], dim=1))  # flow的输入改成source - target, delta w

        if aligned is not None:
            aligned = align_upsample_add(aligned, align)
        else:
            aligned = align

        # transform
        delta_x, delta_y, alpha = aligned[:, 0, ...], aligned[:, 1, ...], aligned[:, 2:, ...]
        delta_x, delta_y = self.tanh(delta_x) * self.scale, self.tanh(delta_y) * self.scale
        alpha = self.sigmoid(alpha)
        ind_y, ind_x = torch.meshgrid(torch.linspace(-1, 1, source.shape[2], device=delta_x.device),
                                      torch.linspace(-1, 1, source.shape[3], device=delta_y.device))
        grid = torch.stack([ind_x.unsqueeze(0) + delta_x, ind_y.unsqueeze(0) + delta_y], dim=-1)

        warped_source = F.grid_sample(source, grid)
        aligned_source = self.refine(torch.cat([warped_source, target], dim=1)) * alpha + (warped_source * (1 - alpha))

        return aligned_source, aligned

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


from src.ops.dcn import DeformableConv2d


class bottleneck_IR_Deform(Module):
    def __init__(self, in_channel, depth, stride, bn=True):
        super(bottleneck_IR_Deform, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                DeformableConv2d(in_channel, depth, 1, stride, 0, bias=False),
                BN(depth, bn=bn)
            )
        self.res_layer = Sequential(
            # BatchNorm2d(in_channel),
            BN(in_channel, bn=bn),
            DeformableConv2d(in_channel, depth, 3, 1, 1, bias=False), PReLU(depth),
            DeformableConv2d(depth, depth, 3, stride, 1, bias=False), BN(depth, bn=bn)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride, bn=True, bias=False):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=bias),
                BN(depth, bn=bn)
            )
        self.res_layer = Sequential(
            # BatchNorm2d(in_channel),
            BN(in_channel, bn=bn),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=bias),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=bias),
            BN(depth, bn=bn)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


from src.ops.StyleGAN.modules import EqualConv2d


class bottleneck_IR_Equal(Module):
    def __init__(self, in_channel, depth, stride, bn=True, bias=False):
        super(bottleneck_IR_Equal, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                EqualConv2d(in_channel, depth, 1, stride, bias=bias),
                BN(depth, bn=bn)
            )
        self.res_layer = Sequential(
            # BatchNorm2d(in_channel),
            BN(in_channel, bn=bn),
            EqualConv2d(in_channel, depth, 3, 1, 1, bias=bias),
            PReLU(depth),
            EqualConv2d(depth, depth, 3, stride, 1, bias=bias),
            BN(depth, bn=bn)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride, bn=True):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BN(depth, bn=bn)
            )
        self.res_layer = Sequential(
            BN(in_channel, bn=bn),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BN(depth, bn=bn),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x) + shortcut
        return res


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bicubic upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bicubic', align_corners=True) + y


def _RPM(x, y, th=0.0001, th2=0.1, th3=0.2, up=None):
    _, _, H, W = y.size()
    # g = torch.where(torch.gt(x, th) & torch.lt(x, 1 - th), torch.ones_like(x), torch.zeros_like(x))
    g = torch.clip(torch.min((1 - th - x), x - th) / th2, th3, 1.0)
    if up is None:
        up_x = F.interpolate(x, size=(H, W), mode='bicubic', align_corners=True)
        up_g = F.interpolate(g, size=(H, W), mode='bicubic', align_corners=True)
    else:
        up_x = up(x)
        up_g = up(g)

    y_fused = (y * up_g) + (up_x * (1 - up_g))
    return y_fused

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
    # y_fused = (y * up_g) + up_x
    
    return y_fused