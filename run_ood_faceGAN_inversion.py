from src.archs.OOD_faceGAN_e4e_arch import ood_faceGAN_e4e
from src.archs.OOD_faceGAN_featureStyle_arch import ood_faceGAN_FeatureStyle
from src.archs.OOD_faceGAN_restyle_arch import ood_faceGAN_restyle

from basicsr.metrics import calculate_psnr, calculate_ssim
from src.metrics import calculate_lpips, calculate_identity

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
import argparse

model_dict = {
    'ood_faceGAN_e4e': ood_faceGAN_e4e,
    'ood_faceGAN_restyle': ood_faceGAN_restyle,
    'ood_faceGAN_FeatureStyle': ood_faceGAN_FeatureStyle
}

def load_model(opts):
    opt = opts['network_g']
    pretrained_opt = opts['path']

    model_type = opt.pop('type')
    model = model_dict[model_type](**opt)
    pth_ = torch.load(pretrained_opt['pretrain_network_g'])[pretrained_opt['param_key_g']]
    pth__ = OrderedDict()

    for k in pth_.keys():
        if 'delta_latent' not in k:
            pth__[k] = pth_[k]
        elif len(pth_[k].shape) >= 3:
            pth__[k] = pth_[k]

    model.load_state_dict(pth__, strict=pretrained_opt['strict_load_g'])
    model.delta_latent.data = torch.zeros_like(model.delta_latent)

    return model

def load_files_from_path(opt, directions_dir=None):
    data_root = opt['dataroot']
    im_list = os.listdir(data_root)
    im_list = sorted(im_list, key=lambda x: (x[:-4]))
    im_list = [os.path.join(data_root, im) for im in im_list]

    editing = opt.get('editing', None)
    if editing is not None:
        direction = np.load(osp.join(directions_dir, editing['direction']+'.npy'))
        direction = torch.tensor(direction, dtype=torch.float32).unsqueeze(0) * editing['intensity']
    else:
        direction = torch.tensor(0.)
    
    return im_list, direction

def save_img_to(tensor, name='vis', root=None, ten2img=True):
    if not osp.isdir(root):
        os.makedirs(root, exist_ok=True)
    if ten2img:
        img = tensor2img(tensor, rgb2bgr=True, min_max=(-1, 1))
    else:
        img = tensor
    cv2.imwrite(osp.join(root, name), img)
    return img
    
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

def eval(cv2gt, cv2res, metrics=None, opt=None):
    if metrics is None:
        metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'identity': []
        }
    
    lpips = opt.get('lpips', None)
    ssim = opt.get('ssim', None)
    psnr = opt.get('psnr', None)
    identity = opt.get('identity', None)

    # cal lpips
    if lpips:
        metrics['lpips'].append(calculate_lpips(cv2gt, cv2res, crop_border=lpips['crop_border'], test_y_channel=lpips['test_y_channel']))
    else:
        metrics['identity'].append(0)

    if ssim:
        metrics['ssim'].append(calculate_ssim(cv2gt, cv2res, crop_border=ssim['crop_border'], test_y_channel=ssim['test_y_channel']))
    else:
        metrics['identity'].append(0)

    if psnr:
        metrics['psnr'].append(calculate_psnr(cv2gt, cv2res, crop_border=psnr['crop_border'], test_y_channel=psnr['test_y_channel']))
    else:
        metrics['identity'].append(0)

    if identity:
        metrics['identity'].append(calculate_identity(cv2gt, cv2res, crop_border=identity['crop_border'], test_y_channel=identity['test_y_channel'], net=identity['model_path']))
    else:
        metrics['identity'].append(0)
    
    return metrics

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', default=None, required=True, help='the testing option file path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    with open(args.opt) as f:
        opts = yaml.load(f, Loader=yaml.FullLoader)
    
    # load pretrained model
    model = load_model(opts)

    # load predefined editing directions
    directions_dir = opts.get('directions_dir', './directions')

    save_root = opts.get('save_dir', './results')
    save_root = os.path.join(save_root, opts['name'])

    for dataset_name, dataset_opt in opts['datasets'].items():
        files, edit_direction = load_files_from_path(dataset_opt, directions_dir)
        save_dir = os.path.join(save_root, dataset_name)
        
        model = model.cuda()
        model.delta_latent.data += edit_direction.cuda()

        pbar = tqdm(total=len(files))
        
        times = []
        metrics = None

        for file in files:
            cv2im = cv2.imread(file) / 255.0
            input_im = (torch.stack(img2tensor([cv2im], bgr2rgb=True), dim=0) - 0.5) * 2
            if input_im.shape[-1] != 1024:
                input_im = F.interpolate(input_im, size=(1024, 1024), mode='bilinear')

            # to cuda
            input_im = input_im.cuda()

            with torch.no_grad():
                start = time.time()
                inversion_im, _ = model(input_im)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end-start)

                result = save_img_to(inversion_im, name=os.path.split(file)[-1], root=f'{save_dir}/inversion')

                # evaluations
                metrics = eval(cv2im * 255.0, result, metrics, opts['metrics'])

                masks = extract_masks(model.aligns)
                save_img_to(masks, name=os.path.split(file)[-1], root=f'{save_dir}/masks', ten2img=False)

            pbar.update(1)

        model.delta_latent.data -= edit_direction.cuda()

        # print summerizes
        logger.info('Average process time of %s: %f', dataset_name, np.mean(np.array(times)))
        logger.info('Average PSNR of %s: %f', dataset_name, np.mean(np.array(metrics['psnr'])))
        logger.info('Average SSIM of %s: %f', dataset_name, np.mean(np.array(metrics['ssim'])))
        logger.info('Average LPIPS of %s: %f', dataset_name, np.mean(np.array(metrics['lpips'])))
        logger.info('Average ID_SCORE of %s: %f', dataset_name, np.mean(np.array(metrics['identity'])))

