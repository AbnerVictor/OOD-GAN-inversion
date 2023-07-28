
import torch
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']
@LOSS_REGISTRY.register()
class MaskLoss(nn.Module):
    def __init__(self, loss_weight=1.0, loss_func={}, **kwargs):
        super(MaskLoss, self).__init__()
        self.loss_weight = loss_weight
        self.binary = loss_func.get('binary', [64])  #
        self.area = loss_func.get('area', {'64': 0.35,
                                           '128': 0.01,
                                           '256': 0.01})
        self.target = loss_func.get('target', 0)  # reverse mask if target is 0
        self.binary_weight = loss_func.get('binary_weight', 0.5)

    def forward(self, aligns):
        total_bin_loss = torch.tensor(0.)
        total_area_loss = torch.tensor(0.)
        for _, align in aligns.items():
            if total_bin_loss.device != align.device:
                total_bin_loss = total_bin_loss.to(align.device)
                total_area_loss = total_area_loss.to(align.device)

            mask = align[:, 2:, ...]
            b, c, size, _ = mask.shape
            
            # binary loss
            if size in self.binary:
                bin_loss = torch.mean(torch.min(mask, 1 - mask))
                total_bin_loss += bin_loss

            # area loss
            if str(size) in self.area.keys():
                
                if self.target == 0:
                    mask = 1 - mask

                area = self.area[str(size)]
                length = b * c * size * size
                avg_area = torch.sum(mask) / length
                # print(size, self.target, avg_area)
                # input('stop')
                area_loss = max(torch.tensor(0., device=mask.device), avg_area - area)
                # area_loss = max(area - avg_area, avg_area - area)
                total_area_loss += area_loss


        return total_bin_loss * self.binary_weight * self.loss_weight, total_area_loss * self.loss_weight
