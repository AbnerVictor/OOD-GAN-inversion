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
import torchvision.transforms as transforms

from basicsr.archs.arch_util import trunc_normal_
from src.ops.e4e.encoders.psp_encoders import ProgressiveStage
from src.ops.StyleGAN.model import Generator, EqualLinear, NoiseInjection
from src.ops.StyleGAN.modules import Upsample

from src.ops.FeatureStyle.feature_style_encoder import fs_encoder_v2
from src.ops.SAMM.helpers import StyledscaleNshfitBlock


@ARCH_REGISTRY.register()
class ood_faceGAN_FeatureStyle(nn.Module):
    def __init__(self, 
                 # generator opts
                 out_size=1024, style_dim=512, n_mlp=8, channel_multiplier=2, narrow=1, merge='', 
                 StyleGAN_pth=None, StyleGAN_pth_key='g_ema', 
                 # augmentation
                 aug_alignment=False, aug_inputcolor=False,
                 # encoder opts 
                 encoder='FeatureStyle', FeatureStyle_pth=None, arcface_model_path=None, avg_latent_pth=None,
                 optim_delta_latent=False, delta_latent_pth=None, 
                 # modulation opts
                 enable_modulation=True, modulation_type='NOISE', warp_scale=0.02,
                 blend_with_gen=True, ModSize=None,
                 # training opts
                 progressiveModSize=[16, 32, 64, 128, 256], progressiveStart=20000, 
                 progressiveStep=2000, progressiveStageSteps=[999999999], eval_path_length=None,
                 **kwargs):
        super(ood_faceGAN_FeatureStyle, self).__init__()
        self.encoder_type = encoder
        log_outsize = int(math.log(out_size, 2))
        self.style_cnt = log_outsize * 2 - 2
        self.style_dim = style_dim
        
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

        self.avg_latent = nn.Parameter(torch.zeros((self.style_cnt, style_dim)), requires_grad=False)
        if avg_latent_pth is not None:
            self.avg_latent.data = torch.load(avg_latent_pth, map_location='cpu')


        if encoder == 'FeatureStyle':
            assert FeatureStyle_pth is not None
            enc_ckpt = torch.load(FeatureStyle_pth, map_location='cpu')
            
            self.encoder = fs_encoder_v2(n_styles=self.style_cnt, opts=easydict.EasyDict({'arcface_model_path': arcface_model_path}), 
                                         residual=False, use_coeff=False, resnet_layer=[4, 5, 6],  stride=(2, 2))

            self.encoder.load_state_dict(enc_ckpt, strict=True)
            
            self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

            if enable_modulation:
                enc_channels = [64, 64, 128, 256]
                self.feats_conv = nn.ModuleList()
                featsize = 256
                for i in range(4):
                    conv = nn.Conv2d(enc_channels[i], self.channels[featsize], kernel_size=1, stride=1, padding=0)
                    self.feats_conv.append(conv)
                    featsize /= 2

        self.generator = Generator(size=out_size, n_mlp=n_mlp, style_dim=style_dim,
                                    channel_multiplier=channel_multiplier)

        if StyleGAN_pth is not None:
            gen_ckpt = torch.load(StyleGAN_pth, map_location='cpu')[StyleGAN_pth_key]
            self.generator.load_state_dict(gen_ckpt, strict=False)
        
        # self.conditions = {}
        self.aligns = {}
        self.log_outsize = int(math.log(256, 2))
        if enable_modulation:
            self.modulation = nn.ModuleList()
            self.progressiveModSize = progressiveModSize
            self.modulation_type = modulation_type
            self.blend_with_gen = blend_with_gen
            self.blend_cnt = kwargs.get('blend_cnt', 1)
            self.skip_SA = kwargs.get('skip_SA', False)

            if aug_alignment:
                self.randomTransform = transforms.RandomPerspective(aug_scale_and_p[0], aug_scale_and_p[1])
            else:
                self.randomTransform = None

            if aug_inputcolor:
                self.colorTransform = nn.Sequential(
                    transforms.Normalize([-1, -1, -1], [2, 2, 2]),  # [-1, 1] to [0, 1]
                    transforms.ColorJitter(brightness=aug_colorjitter[0], contrast=aug_colorjitter[1],
                                           saturation=aug_colorjitter[2], hue=aug_colorjitter[3]),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                )
            else:
                self.colorTransform = None

            if ModSize is None:
                self.ModSize = self.progressiveModSize.pop(0)
            else:
                self.ModSize = ModSize

            for i in range(self.log_outsize, 4, -1):
                chn = self.channels[2 ** i]
                chn_mul = 2 if modulation_type == 'SFT' else 1
                scaleNshift = StyledscaleNshfitBlock(chn, chn * chn_mul, style_dim,
                                                     scale=warp_scale,
                                                     btn=kwargs.get('mod_btn', None),
                                                     cycle_align=kwargs.get('cycle_align', 1),
                                                     diff_fAndg=kwargs.get('diff_fAndg', True),
                                                     bias=kwargs.get('mod_bias', False))
                self.modulation.append(scaleNshift)
        else:
            self.modulation = None
            self.ModSize = 0
        
        self.delta_latent = nn.Parameter(torch.zeros((1, self.style_cnt, style_dim)), requires_grad=False) 
        
        self.progressiveStageSteps = progressiveStageSteps
        if self.progressiveStageSteps is None:
            self.progressiveStageSteps = [progressiveStart + progressiveStep * i for i in
                                          range(self.style_cnt)]

    def fix_module_list(self):
        # Fix modulation layers
        fix_list = []
        return fix_list

    def update_stage(self, step, logger=None):
        if len(self.progressiveStageSteps) > 0:
            milestone = self.progressiveStageSteps[0]
            while step > milestone:
                self.progressiveStageSteps.pop(0)

                # progressive Mod training
                if self.modulation is not None and len(self.progressiveModSize) > 0 and self.ModSize < \
                        self.progressiveModSize[0]:
                    self.ModSize = self.progressiveModSize.pop(0)
                    if logger is not None:
                        logger.info(f'modulation size: {self.ModSize}')
                        logger.info(f'fix layers: {self.fix_module_list()}')

                if len(self.progressiveStageSteps) > 0:
                    milestone = self.progressiveStageSteps[0]
                else:
                    milestone = step

    def get_style_mlp(self, x):
        if isinstance(self.generator, Generator):
            return self.generator.style(x)
        else:
            return None

    def random_gen(self, batch_size=1, gen=True):
        with torch.no_grad():
            style = torch.randn((batch_size, self.style_dim), device=self.avg_latent.device)
            lats = self.get_style_mlp(style).unsqueeze(1).repeat(1, self.style_cnt, 1)
            if gen:
                out, _ = self.generator(lats, input_is_tensor=True, input_is_latent=True)
            else:
                out = None
        return out, lats

    def random_gen_center(self, scale=0.1, gen=True):
        with torch.no_grad():
            lats = self.avg_latent + (torch.randn_like(self.avg_latent) * scale)
            lats = lats.unsqueeze(1).repeat(1, self.style_cnt, 1)
            if gen:
                out, _ = self.generator(lats, input_is_tensor=True, input_is_latent=True)
            else:
                out = None
        return out, lats

    def feats2condition(self, feats, **kwargs):
        conditions = []
        if self.ModSize > 0:
            max_size = int(np.floor(math.log(self.ModSize, 2)))
            min_size = int(np.floor(math.log(feats[-1].shape[-1], 2)))
            cond_len = min(max((1 + max_size - min_size), 0), len(feats))
            for i in range(cond_len):
                conditions.append([None, None])
        return conditions

    def feats2condition_callback(self, image, **kwargs):
        ind = kwargs.get('index') + 1
        feat = self.feats[-ind]
        mod = self.modulation[-ind]
        style = kwargs.get('style')
        # delta = style - self.ori_lats[:, 2 * (ind + 1) - 1, ...]

        if kwargs.get('align_aug', False):
            kwargs.update({'transform': self.randomTransform})

        noise = kwargs.get('noise', torch.randn_like(image))
        noise_weight = kwargs.get('noise_weight', 1)

        aligned = self.aligns[ind - 1] if ind > 1 else None
        condition, align = mod(feat, style, image=image, aligned=aligned, **kwargs)
        condition = condition - image + noise * noise_weight

        self.aligns[ind] = align
        return condition / noise_weight

    def generate(self, lats, feats=None, x=None, contents=None, skip_modulation=False, skip_blending=False):
        if self.modulation is not None and not skip_modulation:
            try:
                del self.feats
                torch.cuda.empty_cache()
            except:
                pass
            self.feats = []
            for i in range(4):
                feat = self.feats_conv[i](feats[i])
                self.feats.append(feat)
            
            try:
                del self.lats
                torch.cuda.empty_cache()
            except:
                pass
            
            self.lats = lats
            
            conditions = self.feats2condition(self.feats)
            cond_ind = [(2 * (k + 2)) + 1 for k in range(len(conditions))]
            out, _ = self.generator(lats, input_is_tensor=True, input_is_latent=True,
                                    conditions=conditions, cond_layers=cond_ind, cond_type=self.modulation_type,
                                    callback=self.feats2condition_callback,
                                    align_aug=True if self.randomTransform is not None else False,
                                    features_in=contents, feature_scale=1.0)

            if self.blend_with_gen and not skip_blending:
                if self.skip_SA:
                    with torch.no_grad():
                        gen, _ = self.generator(lats, input_is_tensor=True, input_is_latent=True)
                        out = gen.detach()

                alpha_scale = self.blending_mask()
                for i in range(self.blend_cnt):
                    out = self.blend(x.detach(), out, detach=False, alpha_scale=alpha_scale)
        else:
            out, _ = self.generator(lats, input_is_tensor=True, input_is_latent=True,
                                    features_in=contents, feature_scale=1.0)
        
        return out

    def forward(self, x, **kwargs):
        random_gen = kwargs.get('random_gen', False)
        if random_gen:
            return self.random_gen(batch_size=kwargs.get('batch_size', 1),
                                   gen=kwargs.get('gen', True))

        # update progressive stage
        step = kwargs.get('step', None)
        if step is not None:
            self.update_stage(step, kwargs.get('logger', None))

        with torch.no_grad():
            # initial predction of latent codes
            self.encoder.eval()
            lats, contents, feats = self.encoder(self.face_pool(x), return_feats=True)
            contents = [None]*5 + [contents] + [None]*(17-5)    
            lats = lats + self.avg_latent.unsqueeze(0)
        
        # add delta_latent
        lats += self.delta_latent

        # truncation
        truncation = kwargs.get('truncation', 1.0)
        if truncation < 1.0:
            lats = self.avg_latent.unsqueeze(0) * (1. - truncation) + (lats * truncation)

        self.ori_lats = lats

        out = self.generate(lats, feats, x)

        return out, lats
    
    def blending_mask(self):
        try:
            item = self.aligns.pop(1024)
            del item
            torch.cuda.empty_cache()
        except:
            pass

        keys = sorted(self.aligns.keys(), reverse=False)

        alpha_scale = None
        for k in range(len(keys)):
            align = self.aligns[keys[k]]
            
            if alpha_scale is None:
                alpha_scale = F.interpolate(align[:, 2:, ...], size=(self.generator.size, self.generator.size), mode='bilinear')
            else:
                alpha_scale_ = F.interpolate(align[:, 2:, ...], size=(self.generator.size, self.generator.size), mode='bilinear')
                alpha_scale = (alpha_scale_ * alpha_scale) + (alpha_scale * (1 - alpha_scale))
                # alpha_scale += alpha_scale_

        if alpha_scale is not None:
            alpha_scale = torch.clip(alpha_scale, 0.0, 1.0)
            self.aligns[1024] = alpha_scale.repeat(1, 3, 1, 1)
        return alpha_scale
    
    def blend(self, target, output, detach=True, alpha_scale=None):
        out = None
        if alpha_scale is not None:
            if detach:
                alpha_scale = alpha_scale.detach()
            out = alpha_scale * target + output * (1 - alpha_scale)
        return out