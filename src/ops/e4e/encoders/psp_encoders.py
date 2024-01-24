from enum import Enum
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from src.ops.e4e.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, \
    _upsample_add, BN, get_block
from src.ops.StyleGAN.modules import EqualLinear


class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class Encoder4Editing(Module):
    def __init__(self, num_layers, mode='ir', opts=None, bn=True):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BN(64, bn=bn),
                                      PReLU(64))
        self.channels = [64]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride,
                                           bn=bn))
            self.channels.append(block[-1].depth)

        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x, **kwargs):
        x = self.input_layer(x)

        feats = [x]

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                feats.append(x)
            if i == 6:
                c1 = x
                feats.append(x)
            elif i == 20:
                c2 = x
                feats.append(x)
            elif i == 23:
                c3 = x
                feats.append(x)

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        if kwargs.get('return_feats', False):
            return w, feats
        else:
            return w


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2)


from src.ops.StyleGAN.modules import *


# Consultation encoder
class ResidualEncoder(Module):
    def __init__(self, opts=None):
        super(ResidualEncoder, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))

        self.conv_layer2 = Sequential(*[bottleneck_IR(32, 48, 2), bottleneck_IR(48, 48, 1), bottleneck_IR(48, 48, 1)])

        self.conv_layer3 = Sequential(*[bottleneck_IR(48, 64, 2), bottleneck_IR(64, 64, 1), bottleneck_IR(64, 64, 1)])

        self.condition_scale3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

        self.condition_shift3 = nn.Sequential(
            EqualConv2d(64, 512, 3, stride=1, padding=1, bias=True),
            ScaledLeakyReLU(0.2),
            EqualConv2d(512, 512, 3, stride=1, padding=1, bias=True))

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def forward(self, x):
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)

        scale = self.condition_scale3(feat3)
        scale = F.interpolate(scale, size=(64, 64), mode='bilinear')
        conditions.append(scale.clone())
        shift = self.condition_shift3(feat3)
        shift = F.interpolate(shift, size=(64, 64), mode='bilinear')
        conditions.append(shift.clone())
        return conditions


# ADA
class ResidualAligner(Module):
    def __init__(self, opts=None):
        super(ResidualAligner, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(6, 16, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(16),
                                      PReLU(16))

        self.conv_layer2 = Sequential(*[bottleneck_IR(16, 32, 2), bottleneck_IR(32, 32, 1), bottleneck_IR(32, 32, 1)])
        self.conv_layer3 = Sequential(*[bottleneck_IR(32, 48, 2), bottleneck_IR(48, 48, 1), bottleneck_IR(48, 48, 1)])
        self.conv_layer4 = Sequential(*[bottleneck_IR(48, 64, 2), bottleneck_IR(64, 64, 1), bottleneck_IR(64, 64, 1)])

        self.dconv_layer1 = Sequential(*[bottleneck_IR(112, 64, 1), bottleneck_IR(64, 32, 1), bottleneck_IR(32, 32, 1)])
        self.dconv_layer2 = Sequential(*[bottleneck_IR(64, 32, 1), bottleneck_IR(32, 16, 1), bottleneck_IR(16, 16, 1)])
        self.dconv_layer3 = Sequential(*[bottleneck_IR(32, 16, 1), bottleneck_IR(16, 3, 1), bottleneck_IR(3, 3, 1)])

    def forward(self, x):
        feat1 = self.conv_layer1(x)
        feat2 = self.conv_layer2(feat1)
        feat3 = self.conv_layer3(feat2)
        feat4 = self.conv_layer4(feat3)

        feat4 = F.interpolate(feat4, size=(64, 64), mode='bilinear')
        dfea1 = self.dconv_layer1(torch.cat((feat4, feat3), 1))
        dfea1 = F.interpolate(dfea1, size=(128, 128), mode='bilinear')
        dfea2 = self.dconv_layer2(torch.cat((dfea1, feat2), 1))
        dfea2 = F.interpolate(dfea2, size=(256, 256), mode='bilinear')
        dfea3 = self.dconv_layer3(torch.cat((dfea2, feat1), 1))

        res_aligned = dfea3

        return res_aligned


class pSp_like_Encoder(nn.Module):
    def __init__(self,
                 out_size=256,
                 style_dim=512,
                 channel_multiplier=1,
                 narrow=1,
                 style_cnt=None,
                 min_feat_size=8,
                 norm_type='InstanceNorm',
                 mode='ir'):
        super(pSp_like_Encoder, self).__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.out_size = out_size
        self.log_outsize = int(math.log(out_size, 2))

        # Down layers
        self.conv_first = nn.Sequential(nn.Conv2d(3, channels[out_size], (3, 3), 1, 1, bias=False),
                                        BN(channels[out_size], bn=norm_type),
                                        nn.PReLU(channels[out_size]))

        self.down_layers = ['down%d' % i for i in range(self.log_outsize - 1)]
        in_channel = channels[out_size]

        for i in range(self.log_outsize, 4, -1):
            out_channel = channels[2 ** (i - 1)]
            conv_block = [bottleneck_IR(in_channel=in_channel, depth=out_channel,
                                        stride=2, bn=norm_type),
                          bottleneck_IR(in_channel=out_channel, depth=out_channel,
                                        stride=1, bn=norm_type),
                          bottleneck_IR(in_channel=out_channel, depth=out_channel,
                                        stride=1, bn=norm_type)]
            setattr(self, self.down_layers[self.log_outsize - i + 1], nn.Sequential(*conv_block))
            in_channel = out_channel

        self.style_dim = style_dim
        if style_cnt is None:
            self.style_cnt = self.log_outsize * 2 - 2
        else:
            self.style_cnt = style_cnt
        self.coarse_ind = 3
        self.middle_ind = 7
        self.styles = nn.ModuleList()

        for i in range(self.style_cnt):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer2 = nn.Conv2d(channels[32], 512, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(channels[64], 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def forward(self, x, **kwargs):
        x = self.conv_first(x)

        feats = [x]

        for i in range(self.log_outsize, 4, -1):
            conv_layer = getattr(self, self.down_layers[self.log_outsize - i + 1])
            x = conv_layer(x)
            feats.append(x)

        c3 = feats[-1]
        c2 = _upsample_add(c3, self.latlayer2(feats[-2]))
        c1 = _upsample_add(c2, self.latlayer1(feats[-3]))

        lat0 = self.styles[0](c3)
        styles = lat0.reshape(-1, 1, self.style_dim).repeat(1, self.style_cnt, 1)
        stage = self.progressive_stage.value
        for i in range(1, min(stage + 1, self.style_cnt)):
            if i < self.coarse_ind:
                delta = self.styles[i](c3)
            elif i < self.middle_ind:
                delta = self.styles[i](c2)
            else:
                delta = self.styles[i](c1)
            styles[:, i] += delta

        return styles, feats


class Feat_Encoder(nn.Module):
    def __init__(self,
                 out_size=256,
                 style_dim=512,
                 channel_multiplier=1,
                 narrow=1,
                 min_feat_size=8,
                 norm_type='InstanceNorm',
                 mode='ir'):
        super(Feat_Encoder, self).__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.out_size = out_size
        self.log_outsize = int(math.log(out_size, 2))

        # Down layers
        self.conv_first = nn.Sequential(nn.Conv2d(3, channels[out_size], (3, 3), 1, 1, bias=False),
                                        BN(channels[out_size], bn=norm_type),
                                        nn.PReLU(channels[out_size]))

        self.down_layers = ['down%d' % i for i in range(self.log_outsize - 1)]
        in_channel = channels[out_size]

        for i in range(self.log_outsize, 4, -1):
            out_channel = channels[2 ** (i - 1)]
            conv_block = [bottleneck_IR(in_channel=in_channel, depth=out_channel,
                                        stride=2, bn=norm_type),
                          bottleneck_IR(in_channel=out_channel, depth=out_channel,
                                        stride=1, bn=norm_type),
                          bottleneck_IR(in_channel=out_channel, depth=out_channel,
                                        stride=1, bn=norm_type)]
            setattr(self, self.down_layers[self.log_outsize - i + 1], nn.Sequential(*conv_block))
            in_channel = out_channel

        self.style_dim = style_dim
        self.style_cnt = self.log_outsize * 2 - 2
        self.progressive_stage = ProgressiveStage.Inference

    def forward(self, x):
        x = self.conv_first(x)

        feats = [x]

        for i in range(self.log_outsize, 4, -1):
            conv_layer = getattr(self, self.down_layers[self.log_outsize - i + 1])
            x = conv_layer(x)
            feats.append(x)

        return feats


class pSp_like_Encoder_v2(nn.Module):
    def __init__(self,
                 out_size=256,
                 style_dim=512,
                 channel_multiplier=1,
                 narrow=1.0,
                 style_cnt=None,
                 norm_type=False,
                 mode='ir_se',
                 num_layers='ours_v2'):
        super(pSp_like_Encoder_v2, self).__init__()

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow),
        }

        self.out_size = out_size
        self.log_outsize = int(math.log(out_size, 2))

        # Down layers
        self.conv_first = nn.Sequential(nn.Conv2d(3, self.channels[out_size], (3, 3), 1, 1, bias=False),
                                        BN(self.channels[out_size], bn=norm_type),
                                        nn.PReLU(self.channels[out_size]))

        self.down_layers = ['down%d' % i for i in range(4)]

        blocks = [
            get_block(in_channel=self.channels[out_size], depth=self.channels[out_size // 2], num_units=3),
            get_block(in_channel=self.channels[out_size // 2], depth=self.channels[out_size // 4], num_units=4),
            get_block(in_channel=self.channels[out_size // 4], depth=self.channels[out_size // 8], num_units=14),
            get_block(in_channel=self.channels[out_size // 8], depth=self.channels[out_size // 16], num_units=3)
        ]

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        i = 0
        for block in blocks:
            conv_block = []
            for bottleneck in block:
                conv_block.append(unit_module(bottleneck.in_channel,
                                              bottleneck.depth,
                                              bottleneck.stride,
                                              bn=norm_type))
            setattr(self, self.down_layers[i], nn.Sequential(*conv_block))
            i += 1

        self.style_dim = style_dim
        if style_cnt is None:
            self.style_cnt = self.log_outsize * 2 - 2
        else:
            self.style_cnt = style_cnt
        self.coarse_ind = 3
        self.middle_ind = 7
        self.styles = nn.ModuleList()

        for i in range(self.style_cnt):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer2 = nn.Conv2d(self.channels[32], 512, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(self.channels[64], 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def forward(self, x, **kwargs):
        x = self.conv_first(x)

        feats = [x]

        for i in range(0, 4):
            conv_layer = getattr(self, self.down_layers[i])
            x = conv_layer(x)
            feats.append(x)

        c3 = feats[-1]
        c2 = _upsample_add(c3, self.latlayer2(feats[-2]))
        c1 = _upsample_add(c2, self.latlayer1(feats[-3]))

        lat0 = self.styles[0](c3)
        styles = lat0.reshape(-1, 1, self.style_dim).repeat(1, self.style_cnt, 1)
        stage = self.progressive_stage.value
        for i in range(1, min(stage + 1, self.style_cnt)):
            if i < self.coarse_ind:
                delta = self.styles[i](c3)
            elif i < self.middle_ind:
                delta = self.styles[i](c2)
            else:
                delta = self.styles[i](c1)
            styles[:, i] += delta

        return styles, feats


if __name__ == '__main__':
    encoder = pSp_like_Encoder_v2(out_size=256)
    fake_input = torch.randn((1,3,256,256))
    encoder(fake_input)