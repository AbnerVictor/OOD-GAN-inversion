from src.archs.OOD_faceGAN_e4e_arch import ood_faceGAN_e4e
from basicsr.utils import tensor2img, img2tensor
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os.path as osp
import os
import matplotlib.pyplot as plt
import yaml
import logging
from collections import OrderedDict
from tqdm import tqdm
import time

flow_scale = 0.08
root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

# 
exp_name = 'CelebA-Eval-1000-E4E-N3'
data_root = '/dataset/xinyang/newdata_backup/datasets/CelebA/CelebAMask-HQ/CelebA-Eval-1000'

# ood faceGAN pretrained models
pth_ = torch.load('/dataset/xinyang/newdata_backup/workspace/OOD-GAN-inversion/experiments/OOD_faceGAN_e4e/models/net_g_410000.pth')['params_ema']

# pretrained models
StyleGAN_pth=f'{root}/checkpoints/pretrained_models/StyleGAN2/stylegan2-ffhq-config-f.pth'
avg_latent_pth=f'{root}/checkpoints/pretrained_models/StyleGAN2/stylegan2-ffhq-config-f_avg_latent.pth'
E4E_pth=f'{root}/checkpoints/pretrained_models/StyleGAN2/e4e_ffhq_encode.pt'
directions = f'{root}/checkpoints/pretrained_models/directions'
delta_latent_pth = None

save_root = osp.join(root, 'results')
save_dir = osp.join(save_root, exp_name)

def save_img_to(tensor, name='vis', root=save_dir, ten2img=True):
    if not osp.isdir(root):
        os.makedirs(root, exist_ok=True)
    if ten2img:
        img = tensor2img(tensor, rgb2bgr=True, min_max=(-1, 1))
    else:
        img = tensor
    cv2.imwrite(osp.join(root, name + '.jpg'), img)
    
def extract_masks(aligns):
    try:
        keys = aligns.keys()
        keys = sorted(keys)
        masks = []
        for key_ in keys:
            mask = aligns[key_][:, 2:, ...]
            mask = F.interpolate(mask, size=(1024, 1024))
            masks.append(mask)
        masks = torch.cat(masks, dim=3)
        masks = tensor2img(masks[0, ...], min_max=(0, 1))
    except Exception as e:
        masks = None
    return masks

# Ours
model = ood_faceGAN_e4e(StyleGAN_pth=StyleGAN_pth, StyleGAN_pth_key='g_ema', 
                        stage='Inference', avg_latent_pth=avg_latent_pth, E4E_pth=E4E_pth,
                        enable_modulation=True, warp_scale=flow_scale,
                        blend_with_gen=True, cycle_align=1, blend_cnt=1, skip_SA=False,
                        ModSize=256, optim_delta_latent=False, delta_latent_pth=None)

pth__ = OrderedDict()

for k in pth_.keys():
    if 'delta_latent' not in k:
        pth__[k] = pth_[k]
    elif len(pth_[k].shape) >= 3:
        pth__[k] = pth_[k]

model.load_state_dict(pth__, strict=False)
model.delta_latent.data = torch.zeros_like(model.delta_latent)

im_list = os.listdir(data_root)
im_list = sorted(im_list, key=lambda x: (x[:-4]))

pbar = tqdm(total=len(im_list))

times = []

for im in im_list[:100]:
    cv2im = cv2.imread(osp.join(data_root, im)) / 255.0
    input_im = (torch.stack(img2tensor([cv2im], bgr2rgb=True), dim=0) - 0.5) * 2
    if input_im.shape[-1] != 1024:
        input_im = F.interpolate(input_im, size=(1024, 1024), mode='bilinear')

    # to cuda
    input_im = input_im.cuda()
    model = model.cuda()

    with torch.no_grad():
        start = time.time()
        inversion_im, _ = model(input_im)
        torch.cuda.synchronize()
        end = time.time()
        times.append(end-start)

        save_img_to(inversion_im, name=im[:-4], root=f'{save_dir}/inversion')

        masks = extract_masks(model.aligns)
        save_img_to(masks, name=im[:-4], root=f'{save_dir}/masks', ten2img=False)

    pbar.update(1)

avg_time = np.mean(np.array(times[1:]))
print(avg_time)