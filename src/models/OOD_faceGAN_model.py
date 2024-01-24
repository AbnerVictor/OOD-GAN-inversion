import torch
import torch.nn.functional as F
import copy
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import cv2
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.transforms as transforms

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.losses.losses import g_path_regularize  # , r1_penalty
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.stylegan2_model import StyleGAN2Model
from src.ops.optim.ranger import Ranger
from src.ops.e4e.encoders.psp_encoders import ProgressiveStage
import torch.cuda.amp as amp
from basicsr.models import lr_scheduler as lr_scheduler

from torch import autograd as autograd


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def batch_tensor_to_list_tensor(x):
    if len(x.shape) < 4:
        return x
    out = []
    for i in range(x.shape[0]):
        out.append(x[i, ...])
    return out


@MODEL_REGISTRY.register()
class ood_faceGAN_Model(StyleGAN2Model):
    def __init__(self, opt):
        super(ood_faceGAN_Model, self).__init__(opt)
        self.is_dist = True if self.opt['num_gpu'] > 1 else False
        self.save_checkpoint = self.opt['logger'].get('save_checkpoint', True)
        self.save_delta_latent = self.opt['logger'].get('save_delta_latent', False)

        # torch.autograd.set_detect_anomaly(True)
        if not self.is_train:
            self.init_test_settings()

    def init_test_settings(self):
        self.logger = get_root_logger()

        test_opt = self.opt['test']

        # def mimo
        self.is_mimo = self.opt.get('is_mimo', False)

        self.autocast = test_opt.get('autocast', False)
        if self.autocast:
            self.logger.info('Autocast Enabled')

        self.net_d = None

        # define network net_g with Exponential Moving Average (EMA)
        # net_g_ema only used for testing on one GPU and saving, do not need to
        # wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g_ema.eval()

        # Startup-iter
        self.startup_iter = test_opt.get('startup_iter', None)
        self.enable_startup_noise_skipping = test_opt.get('enable_startup_noise_skipping', False)
        self.fix_both_enc_and_gen = test_opt.get('fix_both_enc_and_gen', False)

        if self.startup_iter:
            self.logger.info(f'startup_iter: {self.startup_iter}')
            if self.fix_both_enc_and_gen:
                self.logger.info(f'fix both encoder and generator in startup iters')

    def init_training_settings(self):
        self.logger = get_root_logger()

        train_opt = self.opt['train']

        # def mimo
        self.is_mimo = self.opt.get('is_mimo', True)

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        # define network net_g with Exponential Moving Average (EMA)
        # net_g_ema only used for testing on one GPU and saving, do not need to
        # wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        # define avg_latent
        self.avg_latent = self.get_bare_model(self.net_g).avg_latent.data.clone().detach()

        # define network net_d2
        self.net_d2 = build_network(self.opt['network_d2'])
        self.net_d2 = self.model_to_device(self.net_d2)
        self.print_network(self.net_d2)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d2', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d2', 'params')
            self.load_network(self.net_d2, load_path, self.opt['path'].get('strict_load_d2', True), param_key)

        logger = get_root_logger()
        logger.info(f'Progressive steps: {self.get_bare_model(self.net_g).progressiveStageSteps}')
        try:
            stage = self.get_bare_model(self.net_g).encoder.progressive_stage
            logger.info(f'Initial encoder stage: {stage}')
        except Exception:
            pass
        logger.info(f'Initial mod size: {self.get_bare_model(self.net_g).ModSize}')

        self.net_g.train()
        self.net_g_ema.eval()
        self.net_d.train()
        self.net_d2.train()

        # define losses
        if train_opt.get('gan_opt', None):
            # gan loss (wgan)
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
            # regularization weights
            self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
            self.path_reg_weight = train_opt['path_reg_weight']  # for generator

            self.net_g_reg_every = train_opt['net_g_reg_every']
            self.net_d_reg_every = train_opt['net_d_reg_every']
            self.mixing_prob = train_opt['mixing_prob']

            self.mean_path_length = 0
        else:
            self.cri_gan = None

        # Identity loss
        if train_opt.get('id_opt', None) is not None:
            self.cri_id = build_loss(train_opt['id_opt']).to(self.device)
        else:
            self.cri_id = None

        # Landmark loss
        if train_opt.get('ldm_opt', None) is not None:
            self.cri_ldm = build_loss(train_opt['ldm_opt']).to(self.device)
        else:
            self.cri_ldm = None

        if train_opt.get('latent_opt', None) is not None:
            self.cri_latent = build_loss(train_opt['latent_opt']).to(self.device)
        else:
            self.cri_latent = None

        if train_opt.get('latent_reg_opt', None) is not None:
            self.cri_latent_reg = build_loss(train_opt['latent_reg_opt']).to(self.device)
        else:
            self.cri_latent_reg = None

        # Pixel loss
        if train_opt.get('pix_opt', None) is not None:
            self.cri_pix = build_loss(train_opt['pix_opt']).to(self.device)
        else:
            self.cri_pix = None

        # Peceptual loss
        if train_opt.get('perceptual_opt', None) is not None:
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # augmentation loss
        if train_opt.get('aug_opt', None) is not None:
            self.cri_aug = build_loss(train_opt['aug_opt']).to(self.device)
        else:
            self.cri_aug = None

        # mask loss
        if train_opt.get('mask_opt', None) is not None:
            self.cri_mask = build_loss(train_opt['mask_opt']).to(self.device)
        else:
            self.cri_mask = None
            
        # clip loss
        if train_opt.get('clip_opt', None) is not None:
            self.cri_clip = build_loss(train_opt['clip_opt']).to(self.device)
        else:
            self.cri_clip = None

        if train_opt.get('clip_direct_opt', None) is not None:
            self.cri_clip_direct = build_loss(train_opt['clip_direct_opt']).to(self.device)
            self.src_image = None
        else:
            self.cri_clip_direct = None
            self.src_image = None
        
        if train_opt.get('contextual_opt', None) is not None:
            self.cri_contextual = build_loss(train_opt['contextual_opt']).to(self.device)
        else:
            self.cri_contextual = None

        # which_gt
        self.which_gt = train_opt.get('which_gt', 'gt_inv')
        self.skip_latent_g = train_opt.get('skip_latent_g', False)
        self.skip_gen_g = train_opt.get('skip_gen_g', False)
        self.grad_clip_norm = train_opt.get('grad_clip_norm', 1.0)

        # refinement loss
        self.refinement_loss = train_opt.get('refinement_loss', None)

        # Startup-iter
        self.startup_iter = train_opt.get('startup_iter', None)
        self.fix_and_grad = train_opt.get('fix_and_grad', None)

        if self.startup_iter:
            self.logger.info(f'startup_iter: {self.startup_iter}')
            if self.fix_and_grad:
                self.logger.info(f'fix and grad {self.fix_and_grad} in startup iters')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            if train_opt['scheduler'].get('milestones', None):
                if isinstance(train_opt['scheduler']['milestones'], list):
                    pass
                else:
                    step_size = train_opt['scheduler']['milestones']
                    milestones = [step_size * i for i in range(1, train_opt.get('total_iter') // step_size)]
                    train_opt['scheduler']['milestones'] = milestones
                    logger = get_root_logger()
                    logger.info(f'{scheduler_type} milestones: {milestones}')
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'ReduceLROnPlateau':
            for optimizer in self.optimizers:
                self.schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        if self.cri_gan:
            net_g_reg_ratio = self.net_g_reg_every / (self.net_g_reg_every + 1)
        else:
            net_g_reg_ratio = 1

        encoder_params = []
        generator_params = []
        overfit_params = []

        fix_layers = self.fix_and_grad.get('fix', [])
        grad_layers = self.fix_and_grad.get('grad', [])

        for name, param in self.net_g.named_parameters():
            fix_flag = False
            for n_ in fix_layers:
                fix_flag = True if n_ in name else fix_flag
            for n_ in grad_layers:
                fix_flag = False if n_ in name else fix_flag

            if not fix_flag:
                if 'generator' in name:
                    generator_params.append(param)
                elif 'delta_latent' in name:
                    overfit_params.append(param)
                else:
                    encoder_params.append(param)

        optim_params_g = [
            {  # add encoder params first
                'params': encoder_params,
                'lr': train_opt['optim_g']['lr']
            },
            {  # add generator params
                'params': generator_params,
                'lr': train_opt['optim_g']['lr'] * train_opt['optim_g'].get('generator_lr_decay', 0.1)
            },
            {  # add overfit params
                'params': overfit_params,
                'lr': train_opt['optim_g']['lr'] * train_opt['optim_g'].get('overfit_lr_decay', 1.0)
            }
        ]

        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0 ** net_g_reg_ratio, 0.99 ** net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        if self.cri_gan:
            net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        else:
            net_d_reg_ratio = 1

        normal_params = []
        for name, param in self.net_d.named_parameters():
            normal_params.append(param)
        optim_params_d = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_d']['lr']
        }]

        optim_type = train_opt['optim_d'].pop('type')
        lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
        betas = (0 ** net_d_reg_ratio, 0.99 ** net_d_reg_ratio)
        self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
        self.optimizers.append(self.optimizer_d)

        # optimizer d2
        normal_params = []
        for name, param in self.net_d2.named_parameters():
            normal_params.append(param)
        optim_params_d2 = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_d']['lr']
        }]

        optim_type = train_opt['optim_d2'].pop('type')
        lr = train_opt['optim_d2']['lr'] * net_d_reg_ratio
        betas = (0 ** net_d_reg_ratio, 0.99 ** net_d_reg_ratio)
        self.optimizer_d2 = self.get_optimizer(optim_type, optim_params_d2, lr, betas=betas)
        self.optimizers.append(self.optimizer_d2)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'Ranger':
            optimizer = Ranger(params=params, lr=lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    def feed_data(self, data):
        # self.real_img = data['gt'].to(self.device)
        self.gt = data['gt'].to(self.device)  # (B, K, C, H, W)
        self.lr = data['lr'].to(self.device)  # (B, K, C, H, W)
        self.lq_size = data['lq_size'].to(self.device)  # (B, K)
        self.deg_type = data['deg_type'].to(self.device)  # (B, K)
        if len(self.gt.shape) < 5:
            self.gt = self.gt.unsqueeze(1)
            self.lr = self.lr.unsqueeze(1)
            self.lq_size = self.lq_size.unsqueeze(1)


    def clean_cache(self, gt_inv=True, latent_dis=True, dis=True):
        del self.fake_hr
        del self.latents
        # del self.conditions
        try:
            del self.aligns 
            del self.get_bare_model(self.net_g).aligns
            self.get_bare_model(self.net_g).aligns = {}
        except:
            pass
        try:
            del self.src_image
        except:
            pass
        try:
            del self.blend
        except:
            pass
        try:
            del self.delta
        except:
            pass
        if gt_inv:
            del self.gt_inv
            del self.gt_latents
        if latent_dis:
            del self.fake_latent_pred
            del self.real_latent_pred
        if dis:
            del self.fake_pred
            del self.real_pred

        torch.cuda.empty_cache()

    def infer(self, gt_inv=True, latent_dis=True, dis=True, query_id=True, **kwargs):
        if self.is_mimo:
            b, k, c, h, w = self.gt.shape
        else:
            b, c, h, w = self.gt.shape

        self.fake_hr, self.latents = self.net_g(self.lr.reshape(-1, c, h, w), input_size=self.lq_size,
                                                query_id=query_id,
                                                skip_refinement=False, refinement_only=False,
                                                step=kwargs.get('step', None), logger=get_root_logger())
        
        if self.is_mimo:
            self.fake_hr = self.fake_hr.reshape(b, k, c, h, w)

        try:
            if self.src_image is None:
                self.src_image = self.fake_hr.clone().detach()
        except:
            pass
        
        try:
            self.aligns = self.get_bare_model(self.net_g).aligns
        except Exception as e:
            raise e

        try:
            self.delta = self.get_bare_model(self.net_g).delta_latent
        except Exception as e:
            pass

        if gt_inv:
            self.gt_inv, self.gt_latents = self.get_bare_model(self.net_g).random_gen(
                batch_size=b * k if self.is_mimo else b, gen=False)

        if self.cri_gan:
            if dis:
                self.fake_pred, _ = self.net_d(self.fake_hr.reshape(-1, c, h, w).detach())
                self.real_pred, _ = self.net_d(self.gt.reshape(-1, c, h, w))
            if latent_dis:
                self.fake_latent_pred, _ = self.net_d2(self.latents.detach())
                self.real_latent_pred, _ = self.net_d2(self.gt_latents.detach())

    def test(self):
        if self.is_mimo:
            b, k, c, h, w = self.gt.shape
        else:
            b, c, h, w = self.gt.shape

        with torch.no_grad():
            self.fake_hr, _ = self.net_g(self.lr.reshape(-1, c, h, w), input_size=self.lq_size,
                                                    refinement_only=False, dropout=False)
            
        try:
            self.aligns = copy.deepcopy(self.get_bare_model(self.net_g).aligns)
            del self.get_bare_model(self.net_g).aligns
            self.get_bare_model(self.net_g).aligns = {}
        except Exception as e:
            raise e

        if self.is_mimo:
            self.fake_hr = self.fake_hr.reshape(b, k, c, h, w)

    def grad_net(self, net='net_g', current_iter=0):
        self.optimizer_d.zero_grad()
        self.optimizer_d2.zero_grad()
        self.optimizer_g.zero_grad()

        if net == 'net_g':
            # optimize net_g
            for p in self.net_d.parameters():
                p.requires_grad = False

            for p in self.net_d2.parameters():
                p.requires_grad = False

            for p in self.net_g.parameters():
                p.requires_grad = True

            if self.startup_iter is not None and current_iter < self.startup_iter:
                # Startup-iter
                fix_layers = self.fix_and_grad.get('fix', [])
                grad_layers = self.fix_and_grad.get('grad', [])

                # Dynamic fix_layers
                try:
                    dynamic_fix_layers = self.get_bare_model(self.net_g).fix_module_list()
                    for layer_name in dynamic_fix_layers:
                        if layer_name not in fix_layers:
                            fix_layers.append(layer_name)
                except Exception as e:
                    pass

                for n, param in self.net_g.named_parameters():
                    fix_flag = False
                    for n_ in fix_layers:
                        fix_flag = True if n_ in n else fix_flag

                    for n_ in grad_layers:
                        fix_flag = False if n_ in n else fix_flag
                    param.requires_grad = not fix_flag

        elif net == 'net_d':
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            for p in self.net_d2.parameters():
                p.requires_grad = False

            for p in self.net_g.parameters():
                p.requires_grad = False

        elif net == 'net_d2':
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = False

            for p in self.net_d2.parameters():
                p.requires_grad = True

            for p in self.net_g.parameters():
                p.requires_grad = False

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        if self.is_mimo:
            b, k, c, h, w = self.gt.shape
        else:
            b, c, h, w = self.gt.shape

        if self.cri_gan:
            if not self.skip_gen_g:
                self.grad_net('net_d', current_iter)

                self.infer(gt_inv=False, latent_dis=False, dis=True, step=current_iter)

                # Logistic loss
                l_d = self.cri_gan(self.real_pred, True, is_disc=True) + \
                        self.cri_gan(self.fake_pred, False, is_disc=True)
                loss_dict['l_d'] = l_d
                # In wgan, real_score should be positive and fake_score should be
                # negative
                loss_dict['real_score'] = self.real_pred.detach().mean()
                loss_dict['fake_score'] = self.fake_pred.detach().mean()
                # l_d.backward(retain_graph=True)

                # TODO: keep r1 regularization?
                if current_iter % self.net_d_reg_every == 0:
                    self.gt.requires_grad = True
                    gt_ = self.gt.reshape(-1, c, h, w)
                    real_pred, _ = self.net_d(gt_)
                    l_d_r1 = r1_penalty(real_pred, gt_)
                    l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                    # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
                    # error will arise: RuntimeError: Expected to have finished
                    # reduction in the prior iteration before starting a new one.
                    # This error indicates that your module has parameters that were
                    # not used in producing loss.
                    loss_dict['l_d_r1'] = l_d_r1.detach().mean()
                    # l_d_r1.backward()

                if current_iter % self.net_d_reg_every == 0:
                    l_d.backward(retain_graph=True)
                    l_d_r1.backward()
                else:
                    l_d.backward()
                torch.nn.utils.clip_grad_norm_(self.net_d.parameters(),
                                                max_norm=self.grad_clip_norm)
                self.optimizer_d.step()
                self.clean_cache(gt_inv=False, latent_dis=False, dis=True)

            if not self.skip_latent_g:
                # optimize d2
                self.grad_net('net_d2', current_iter)
                self.infer(gt_inv=True, latent_dis=True, dis=False, step=current_iter)

                # Logistic loss
                l_latent_d = self.cri_gan(self.real_latent_pred, True, is_disc=True) + self.cri_gan(
                    self.fake_latent_pred, False, is_disc=True)
                loss_dict['l_latent_d'] = l_latent_d
                # In wgan, real_score should be positive and fake_score should be
                # negative
                loss_dict['real_latent_score'] = self.real_latent_pred.detach().mean()
                loss_dict['fake_latent_score'] = self.fake_latent_pred.detach().mean()
                # l_d.backward(retain_graph=True)

                # TODO: keep r1 regularization?
                if current_iter % self.net_d_reg_every == 0:
                    gt_ = self.gt_latents
                    gt_.requires_grad = True
                    real_pred, _ = self.net_d2(gt_)
                    l_d_r1 = r1_penalty(real_pred, gt_)
                    l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                    # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
                    # error will arise: RuntimeError: Expected to have finished
                    # reduction in the prior iteration before starting a new one.
                    # This error indicates that your module has parameters that were
                    # not used in producing loss.
                    loss_dict['l_latent_d_r1'] = l_d_r1.detach().mean()
                    l_d_r1.backward()

                l_latent_d.backward()
                torch.nn.utils.clip_grad_norm_(self.net_d2.parameters(),
                                                max_norm=self.grad_clip_norm)

                self.optimizer_d2.step()
                self.clean_cache(gt_inv=True, latent_dis=True, dis=False)

        ####################################### optimize g ########################################
        self.grad_net('net_g', current_iter)

        self.infer(gt_inv=not self.skip_latent_g, latent_dis=not self.skip_latent_g, dis=not self.skip_gen_g,
                    step=current_iter)
        l_total = 0

        gt = getattr(self, self.which_gt)

        if self.cri_gan:
            if not self.skip_gen_g:
                # nonsaturating loss
                l_g = self.cri_gan(self.fake_pred, True, is_disc=False)
                loss_dict['l_g'] = l_g
                l_total += l_g

            if not self.skip_latent_g:
                l_latent_g = self.cri_gan(self.fake_latent_pred, True, is_disc=False)
                loss_dict['l_latent_g'] = l_latent_g
                l_total += l_latent_g

        # Identity loss
        if self.cri_id is not None:
            l_id, l_sim, l_id_logs = self.cri_id(self.fake_hr, gt, self.lr, mimo_id=self.is_mimo,
                                                    score=self.lq_size)
            if self.is_mimo:
                loss_dict['l_id_target'] = l_id
                l_total += l_id
                if l_sim > 1e-5:
                    loss_dict['l_id_ref'] = l_sim
                    l_total += l_sim
            else:
                loss_dict['l_id_target'] = l_id
                l_total += l_id

        if self.cri_ldm is not None:
            gt_ = gt.reshape(-1, c, h, w)
            fake_hr = self.fake_hr.reshape(-1, c, h, w)
            l_ldm = self.cri_ldm(fake_hr, gt_)
            loss_dict['l_ldm'] = l_ldm
            l_total += l_ldm

        # Pixel loss
        if self.cri_pix is not None:
            l_pix = self.cri_pix(self.fake_hr.reshape(-1, c, h, w), gt.reshape(-1, c, h, w))
            loss_dict['l_pix'] = l_pix
            l_total += l_pix

        # Perceptual loss
        if self.cri_perceptual is not None:
            l_percep, l_style = self.cri_perceptual(self.fake_hr.reshape(-1, c, h, w), gt.reshape(-1, c, h, w))
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # latent regularization
        if self.cri_latent_reg is not None:
            l_latent_reg = self.cri_latent_reg(self.delta,
                                                torch.zeros_like(self.latents).detach())
            l_total += l_latent_reg
            loss_dict['l_latent_reg'] = l_latent_reg

        if self.cri_latent is not None:
            l_latent = self.cri_latent(self.gt_latents['ori_latents'], self.latents['latents'])
            l_total += l_latent
            loss_dict['l_latent'] = l_latent

        if self.cri_aug is not None:
            l_aug = self.cri_aug(self.latents['aug_lats'], self.latents['cyc_lats'])
            l_total += l_aug
            loss_dict['l_aug'] = l_aug

        if self.cri_mask is not None:
            l_bin, l_area = self.cri_mask(self.aligns)
            l_total += (l_bin + l_area)
            loss_dict['l_bin'] = l_bin
            loss_dict['l_area'] = l_area
            
        if self.cri_clip is not None:
            l_clip = self.cri_clip(self.fake_hr.reshape(-1, c, h, w))

            l_total += l_clip
            loss_dict['l_clip'] = l_clip

        if self.cri_clip_direct is not None:
            l_clip_direct = self.cri_clip_direct(self.src_image.reshape(-1, c, h, w), self.fake_hr.reshape(-1, c, h, w))
            l_total += l_clip_direct
            loss_dict['l_clip_direct'] = l_clip_direct

        if self.cri_contextual is not None:
            l_contextual = self.cri_contextual(self.fake_hr.reshape(-1, c, h, w), gt.reshape(-1, c, h, w))
            l_total += l_contextual
            loss_dict['l_contextual'] = l_contextual

        # TODO: keep path regulrization?
        if self.cri_gan and current_iter % self.net_g_reg_every == 0 and self.fake_hr.shape[0] > 1:
            # path_batch_size = max(1, batch // self.opt['train']['path_batch_shrink'])
            # noise = self.mixing_noise(path_batch_size, self.mixing_prob)
            # fake_img, latents = self.net_g(noise, return_latents=True)
            l_g_path, path_lengths, self.mean_path_length = g_path_regularize(self.fake_hr.reshape(-1, c, h, w),
                                                                                self.latents,
                                                                                self.mean_path_length)

            l_g_path = (self.path_reg_weight * self.net_g_reg_every * l_g_path + 0 * self.fake_hr[0, 0, 0, 0, 0])
            # TODO:  why do we need to add 0 * fake_img[0, 0, 0, 0]
            # l_g_path.backward()
            loss_dict['l_g_path'] = l_g_path.detach().mean()
            loss_dict['path_length'] = path_lengths

        if self.cri_gan and current_iter % self.net_g_reg_every == 0 and self.fake_hr.shape[0] > 1:
            l_total.backward(retain_graph=True)
            l_g_path.backward()
        else:
            l_total.backward(retain_graph=True)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.get_bare_model(self.net_g).parameters(), max_norm=self.grad_clip_norm)            
        self.optimizer_g.step()
        self.clean_cache(gt_inv=not self.skip_latent_g, latent_dis=not self.skip_latent_g, dis=not self.skip_gen_g)

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA
        self.model_ema(decay=0.5 ** (32 / (10 * 1000)))

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # assert dataloader is None, 'Validation dataloader should be None.'

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        save_lq_and_gt = self.opt['val'].get('save_lq_and_gt', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if self.is_mimo:
                gt_pth = val_data['gt_path'][0][0]
            else:
                gt_pth = val_data['gt_path'][0]
            img_name = osp.splitext(osp.basename(gt_pth))[0]

            self.feed_data(val_data)
            self.test()

            lq = tensor2img(self.lr[0, ...], min_max=(-1, 1))
            gt = tensor2img(self.gt[0, ...], min_max=(-1, 1))
            fake_hr = tensor2img(self.fake_hr[0, ...], min_max=(-1, 1))
            
            try:
                keys = self.aligns.keys()
                keys = sorted(keys)
                masks = []
                for key_ in keys:
                    mask = self.aligns[key_][:, 2:, ...]
                    mask = F.interpolate(mask, size=(1024, 1024))
                    masks.append(mask)
                masks = torch.cat(masks, dim=3)
                masks = tensor2img(masks[0, ...], min_max=(0, 1))
            except Exception as e:
                masks = None

            metric_data['img'] = fake_hr
            metric_data['img2'] = gt

            del self.lr
            del self.gt
            del self.fake_hr
            del self.aligns
            torch.cuda.empty_cache()

            if self.is_mimo:
                # for i in range(len(gt)):
                if save_img:
                    keyword = 'train' if self.opt['is_train'] else 'test'
                    save_img_lq_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                                f'{keyword}_{current_iter}', f'{img_name}_{idx}_lq.jpg')
                    save_img_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                             f'{keyword}_{current_iter}', f'{img_name}_{idx}.jpg')
                    save_img_gt_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                                f'{keyword}_{current_iter}', f'{img_name}_{idx}_gt.jpg')
                    save_img_mask_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                                f'{keyword}_{current_iter}', f'{img_name}_{idx}_mask.jpg')

                try:
                    if save_lq_and_gt:
                        if masks is not None:
                            imwrite(masks, save_img_mask_path)

                    imwrite(fake_hr, save_img_path)
                    
                except Exception as e:
                    get_root_logger().warning(e)

                metric_data['img'] = fake_hr
                metric_data['img2'] = gt

                if with_metrics:
                    # calculate metrics
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            else:
                if save_img:
                    keyword = 'train' if self.opt['is_train'] else 'test'
                    save_img_lq_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                                f'{keyword}_{current_iter}', f'{img_name}_lq.jpg')
                    save_img_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                             f'{keyword}_{current_iter}', f'{img_name}.jpg')
                    save_img_gt_path = osp.join(self.opt['path']['visualization'], f'{keyword}',
                                                f'{keyword}_{current_iter}', f'{img_name}_gt.jpg')

                    if save_lq_and_gt:
                        imwrite(lq, save_img_lq_path)
                        imwrite(gt, save_img_gt_path)

                    imwrite(fake_hr, save_img_path)

                    # add sample images to tb_logger
                    # result = (result / 255.).astype(np.float32)
                    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    # if tb_logger is not None:
                    #     tb_logger.add_image('samples', result, global_step=current_iter, dataformats='HWC')
                if with_metrics:
                    # calculate metrics
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save_param(self, param, param_label, current_iter):
        import os
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{param_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        
        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(param, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')


    def save(self, epoch, current_iter):
        if self.save_checkpoint:
            self.save_network([self.net_g, self.net_g_ema],\
                                'net_g', current_iter, \
                                param_key=['params', 'params_ema'],\
                                keywords=['modulation', 'feats_conv'])
            if not self.skip_gen_g:
                self.save_network(self.net_d, 'net_d', current_iter)
            if not self.skip_latent_g:
                self.save_network(self.net_d2, 'net_d2', current_iter)
        if self.save_delta_latent:
            self.save_param(self.get_bare_model(self.net_g).delta_latent, 'delta_lat', current_iter)
        # self.save_training_state(epoch, current_iter)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups] + \
               [param_group['lr'] for param_group in self.optimizers[1].param_groups]

    def save_network(self, net, net_label, current_iter, param_key='params', keywords=[]):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = osp.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            state_dict_ = OrderedDict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]

                save_ = False
                if len(keywords) < 1:
                    save_ = True
                for k in keywords:
                    if k in key:
                        save_ = True
                if save_:
                    state_dict_[key] = param.cpu()
            save_dict[param_key_] = state_dict_

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

if __name__ == '__main__':
    import yaml
    import src.scripts

    with open('/dataset/xinyang/newdata_backup/workspace/OOD-GAN-inversion/options/train/E4E_Face.yml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt['is_train'] = True
    opt['dist'] = False
    model = ood_faceGAN_Model(opt)

