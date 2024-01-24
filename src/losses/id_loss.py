import os
import torch
from torch import nn
import torch.nn.functional as F
from src.ops.face_id.model_irse import Backbone
from src.ops.face_id.arcface_arch import ResNetArcFace
from scipy.special import comb

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import *
from basicsr import get_root_logger


@LOSS_REGISTRY.register()
class IDLoss(nn.Module):
    def __init__(self, ckpt='experiments/pretrained_models/ir_se/model_ir_se50.pth',
                 loss_weight=1.0, ref_loss_weight=1.0,
                 device='cpu', ckpt_dict=None, reduce='mean'):
        super(IDLoss, self).__init__()
        logger = get_root_logger()
        logger.info(f'Loading ResNet ArcFace from {ckpt}')
        self.reduce = reduce
        self.loss_weight = loss_weight
        self.ref_loss_weight = ref_loss_weight
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').to(device)
        if ckpt_dict is None:
            self.facenet.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        else:
            self.facenet.load_state_dict(ckpt_dict)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h==w
        ss = h//256
        if ss >= 1:
            x = x[:, :, 35*ss:-33*ss, 32*ss:-36*ss]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x, mimo_id=False, **kwargs):
        # mimo_id = kwargs.get('mimo_id', False)

        if not mimo_id:
            n_samples = x.shape[0]
            x_feats = self.extract_feats(x)
            y_feats = self.extract_feats(y)  # Otherwise use the feature from there
            y_hat_feats = self.extract_feats(y_hat)
        else:
            b, k, c, h, w = y_hat.shape
            x_feats = self.extract_feats(x.reshape(-1, c, h, w)).reshape(b, k, -1)
            y_feats = self.extract_feats(y.reshape(-1, c, h, w)).reshape(b, k, -1)  # Otherwise use the feature from there
            y_hat_feats = self.extract_feats(y_hat.reshape(-1, c, h, w)).reshape(b, k, -1)

        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0

        # minimize id distance between outputs
        if mimo_id:
            score = kwargs.get('score', None) # b, k
            b, k = score.shape

            # loss term 1: output and gt
            diff_target = 1 - torch.sum(y_hat_feats * y_feats, dim=-1)

            indexes = torch.stack([score.argmax(dim=-1).to(diff_target.device), diff_target.argmin(dim=-1)], dim=1)
            
            # remove id loss for highest score
            # diff_target[indexes] = 0.

            # loss term 2: output and best output
            diff_outputs = 1 - y_hat_feats @ y_hat_feats.detach().transpose(-1, -2)
            ref_id_loss = 0
            for i in range(b):
                diff = diff_outputs[i, ...] - torch.diag(diff_outputs[i, ...])
                ref_id_loss += torch.mean(diff[:, indexes[i, :].unique()])
            ref_id_loss /= b

            loss = self.loss_weight * reduce_loss(diff_target, self.reduce)
            sim_improvement = self.ref_loss_weight * ref_id_loss
            id_logs.append({'indexs': indexes,
                            'diff_outputs': diff_outputs})

        else:
            for i in range(n_samples):
                diff_target = y_hat_feats[i].dot(y_feats[i])
                diff_input = y_hat_feats[i].dot(x_feats[i])
                diff_views = y_feats[i].dot(x_feats[i])
                id_logs.append({'diff_target': float(diff_target),
                                'diff_input': float(diff_input),
                                'diff_views': float(diff_views)})
                loss += 1 - diff_target
                id_diff = float(diff_target) - float(diff_views)
                sim_improvement += id_diff
                count += 1

            loss = self.loss_weight * loss / count
            sim_improvement = self.loss_weight * sim_improvement / count
        return loss, sim_improvement, id_logs


@LOSS_REGISTRY.register()
class ArcFaceLoss(nn.Module):
    def __init__(self, ckpt='experiments/pretrained_models/arcface/arcface_resnet18.pth',
                 block='IRBlock', layers=[2, 2, 2, 2], use_se=False,
                 loss_weight=1.0, device='cpu', ckpt_dict=None):
        super(ArcFaceLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.loss_weight = loss_weight
        self.facenet = ResNetArcFace(block=block, layers=layers, use_se=use_se).to(device)
        if ckpt_dict is None:
            self.facenet.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        else:
            self.facenet.load_state_dict(ckpt_dict)

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def foward(self, y_hat, y, x, **kwargs):
        y_hat_gray = self.gray_resize_for_identity(y_hat)
        y_gray = self.gray_resize_for_identity(y)

        id_y_hat = self.facenet(y_hat_gray)
        id_y = self.facenet(y_gray).detach()

        l_id = F.l1_loss(id_y_hat, id_y) * self.loss_weight

        return l_id