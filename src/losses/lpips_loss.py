import lpips
import torch
from torch import nn as nn
import torch.fft as fft
import torch.functional as F
import torch.nn.functional as FF

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss, reduce_loss


@LOSS_REGISTRY.register()
class LPIPS_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, min_max=(0, 1), net='alex', model_path=None, reduction='mean', device='cpu'):
        super(LPIPS_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_fn = lpips.LPIPS(net=net, model_path=model_path)  # best forward scores
        self.loss_fn.to(device)
        self.min_max = min_max
        self.reduction = reduction

    def forward(self, pred, target, normalize=True):
        if pred.device != 'cpu':
            self.loss_fn.to(pred.device)

        # norm to 0 ~ 1
        if normalize:
            pred = (pred - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
            target = (target - self.min_max[0]) / (self.min_max[1] - self.min_max[0])

        l = self.loss_fn(pred, target, normalize=normalize)

        l = reduce_loss(l, self.reduction) * self.loss_weight
        return l, None
