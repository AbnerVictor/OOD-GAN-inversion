import random
import time
import os
import cv2
from os import path as osp
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Face_Dataset(data.Dataset):
    def __init__(self, opt):
        super(Face_Dataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        gt_folder_list = opt.get('dataroot_gt_list', None)
        self.lr_folder = opt.get('dataroot_lr', None)
        self.scale = opt.get('scale', None) #
        self.id_annos = opt.get('id_annos', None)
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.slice = opt.get('slice', [0, None])
        self.gt_size = opt.get('gt_size', None)
        # self.degrade_opt = opt.get('degrade_opt', None)
        # self.degrade_opt_2 = opt.get('degrade_opt_2', None)
        # self.degrade_2_prob = opt.get('degrade_2_prob', 0.5)
        self.mode = opt.get('mode', 'mix_id')  # mode: 'mix_id' or 'sep_id'
        self.max_length = opt.get('max_length', 1)
        self.random_seed = opt.get('random_seed', None)
        self.fix_input = opt.get('fix_input', False)

        if self.random_seed is not None:
            random.seed(self.random_seed)

        # if self.io_backend_opt['type'] == 'lmdb':
        # self.io_backend_opt['db_paths'] = self.gt_folder
        # if not self.gt_folder.endswith('.lmdb'):
        #     raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        # with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
        #     self.paths = [line.split('.')[0] for line in fin]
        # else:
        # FFHQ has 70000 images in total
        # self.paths = [osp.join(self.gt_folder, f'{v:08d}.png') for v in range(70000)]

        # if self.degrade_opt is not None:
        #     self.degradation = GFPGAN_degradation(self.degrade_opt)
        # else:
        self.degradation = None

        # if self.degrade_opt_2 is not None:
        #     self.degradation_2 = GFPGAN_degradation(self.degrade_opt_2)
        # else:
        self.degradation_2 = None

        if gt_folder_list is not None:
            self.gt_folder = [osp.expanduser(folder) for folder in gt_folder_list]

        if not isinstance(self.gt_folder, list):
            names = os.listdir(self.gt_folder)
            if self.mode == 'mix_id':
                names = [n for n in names if ('.png' in n or '.jpg' in n)]
                try:
                    names.sort(key=lambda x: int(x[:-4]))
                except Exception:
                    pass
            else:
                names.sort(key=lambda x: int(x))

            self.gt_paths = [osp.join(self.gt_folder, name) for name in names]
            if self.slice[1] is None:
                self.slice[1] = len(self.gt_paths)
            self.gt_paths = self.gt_paths[self.slice[0]:self.slice[1]]

            if self.lr_folder is not None:
                self.lr_paths = [osp.join(self.lr_folder, name) for name in names]
            else:
                self.lr_paths = None
        else:
            self.gt_paths = []
            self.lr_paths = [] if self.lr_folder is not None else None
            for i in range(len(self.gt_folder)):
                gt_folder = self.gt_folder[i]
                names = os.listdir(gt_folder)
                if self.mode == 'mix_id':
                    names = [n for n in names if ('.png' in n or '.jpg' in n)]
                    try:
                        names.sort(key=lambda x: int(x[:-4]))
                    except:
                        pass
                else:
                    try:
                        names.sort(key=lambda x: int(x))
                    except:
                        pass

                gt_paths = [osp.join(gt_folder, name) for name in names]
                self.gt_paths.extend(gt_paths)

                if self.lr_folder is not None:
                    lr_folder = self.lr_folder[i]
                    lr_paths = [osp.join(lr_folder, name) for name in names]
                    self.lr_paths.extend(lr_paths)

    def get_path_by_ind(self, index):
        # load gt image
        gt_path = self.gt_paths[index]

        if self.lr_paths is not None:
            lr_path = self.lr_paths[index]
        else:
            lr_path = None

        if self.mode == 'mix_id':
            if self.fix_input or self.max_length == 1:
                gt_path = [gt_path for _ in range(self.max_length)]
                if lr_path is not None:
                    lr_path = [lr_path for _ in range(self.max_length)]
            else:
                sample_ = random.sample(range(0, self.__len__()), self.max_length)
                gt_path = [self.gt_paths[i] for i in sample_]
                if lr_path is not None:
                    lr_path = [self.lr_paths[i] for i in sample_]

        if self.mode == 'sep_id':
            names = os.listdir(gt_path)
            names = [n for n in names if ('.png' in n or '.jpg' in n)]
            names.sort(key=lambda x: int(x[:-4]))
            if self.fix_input:
                if self.max_length is not None:
                    sample_ = random.sample(range(len(names)), 1)[0]
                    gt_path = [osp.join(gt_path, names[sample_]) for i in range(self.max_length)]
                else:
                    raise Exception('length of fix input?')
            else:
                if self.max_length is not None:
                    if len(names) >= self.max_length:
                        sample_ = random.sample(range(0, len(names)), self.max_length)
                    else:
                        inds = [i for i in range(0, len(names))]
                        sample_ = random.sample(range(0, len(names)), len(names)) + \
                                  [random.choice(inds) for _ in range(self.max_length - len(names))]
                    tmp_ = []
                    for j in sample_:
                        tmp_.append(names[j])
                    names = tmp_
                names.sort(key=lambda x: int(x[:-4]))
                gt_path = [osp.join(gt_path, name) for name in names]
                if lr_path is not None:
                    lr_path = [osp.join(lr_path, name) for name in names]

        return gt_path, lr_path

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path, lr_path = self.get_path_by_ind(index)

        gt_img_bytes = []
        lr_img_bytes = []

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                for path in gt_path:
                    gt_img_bytes.append(self.file_client.get(path))

                if lr_path is not None:
                    for path in lr_path:
                        lr_img_bytes.append(self.file_client.get(path))
                else:
                    lr_img_bytes = None
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path, lr_path = self.get_path_by_ind(index, get_sub_paths=True)
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_gt = []
        img_lr = []
        lq_size = []
        for bytes in gt_img_bytes:
            img = imfrombytes(bytes, float32=True)

            if self.gt_size is not None:
                img = cv2.resize(img, dsize=(self.gt_size, self.gt_size), interpolation=cv2.INTER_LINEAR)

            img_gt.append(img)

        if lr_img_bytes is not None:
            deg_type = torch.ones(len(img_gt))
            for bytes in lr_img_bytes:
                img = imfrombytes(bytes, float32=True)

                if self.gt_size is not None:
                    img = cv2.resize(img, dsize=(self.gt_size, self.gt_size), interpolation=cv2.INTER_LINEAR)

                img_lr.append(img)
        else:
            deg_type = torch.ones(len(img_gt))
            for i in range(len(img_gt)):
                lq_size_ = None
                if self.degradation is None and self.scale is not None:
                    h, w, c = img_gt[i].shape
                    img_lr_ = cv2.resize(img_gt[i], dsize=(h // self.scale, w // self.scale),
                                        interpolation=cv2.INTER_LINEAR)
                elif self.degradation is not None:
                    prob = random.random()
                    if self.degradation_2 is None or prob < 1 - self.degrade_2_prob:
                        img_gt_, img_lr_, lq_size_ = self.degradation.degrade_process(img_gt[i])
                        img_gt[i] = img_gt_
                    else:
                        deg_type[i] = 0
                        img_gt_, img_lr_, lq_size_ = self.degradation_2.degrade_process(img_gt[i])
                        img_gt[i] = img_gt_
                else:
                    img_lr_ = img_gt[i]

                if lq_size_ is None:
                    lq_size_ = img_lr_.shape[0]
                else:
                    lq_size_ = lq_size_[0]

                img_lr.append(img_lr_)
                lq_size.append(torch.tensor(lq_size_))

        # if self.degradation is not None:
        img_aug = augment(img_gt + img_lr, hflip=self.opt.get('use_hflip', False), rotation=False)
        img_gt = img_aug[:len(img_gt)]
        img_lr = img_aug[len(img_gt):]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lr = img2tensor(img_lr, bgr2rgb=True, float32=True)

        if self.mean is not None and self.std is not None:
            for i in range(len(img_gt)):
                normalize(img_gt[i], self.mean, self.std, inplace=True)
                normalize(img_lr[i], self.mean, self.std, inplace=True)

        img_gt = torch.stack(img_gt, dim=0)
        img_lr = torch.stack(img_lr, dim=0)
        if len(lq_size) > 0:
            lq_size = torch.stack(lq_size, dim=0)
        else:
            lq_size = None

        return {'gt': img_gt, 'lr': img_lr, 'lq_size': lq_size, 'gt_path': gt_path, 'deg_type': deg_type}

    def __len__(self):
        return len(self.gt_paths)


if __name__ == '__main__':
    opt = {
        'io_backend': {'type': 'disk'},
        'dataroot_gt': None,
        'dataroot_gt_list':
            [r'\\192.168.100.201\Media-Dev\Experiment_Log\xin.yang\dataset\BasicSR\Image_Generation\CelebA\CelebAMask-HQ\CelebA-HQ-split\train',
             r'\\192.168.100.201\Media-Dev\Experiment_Log\xin.yang\dataset\BasicSR\Image_Generation\CelebA\CelebAMask-HQ\CelebA-HQ-split\val'],
        'mode': 'mix_id',
        # 'dataroot_gt':
        #     r'\\192.168.100.201\Media-Dev\Experiment_Log\xin.yang\dataset\BasicSR\Image_Generation\CelebA\CelebAMask-HQ\CelebA-HQ-identity-sep-eval',
        # 'dataroot_lr':
        #     r'\\192.168.100.201\Media-Dev\Experiment_Log\xin.yang\dataset\BasicSR\Image_Generation\CelebA\CelebAMask-HQ\CelebA-HQ-identity-sep-eval-LQ',
        # 'degrade_opt': {'random_seed': 0},
        # 'degrade_opt_2': {'random_seed': 0},
        # 'random_seed': 1,
        # 'degrade_2_prob': 0.3,
        # 'max_length': None,
        'fix_input': False,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
    loader = Face_Dataset(opt)
    data_ = loader.__getitem__(7000)
    print(data_['gt'].shape)
    print(data_['lq_size'])
    print(data_['gt_path'])
    print(data_['deg_type'])

    import matplotlib.pyplot as plt
    from basicsr.utils import tensor2img

    gt = tensor2img(data_['gt'], rgb2bgr=False, min_max=(-1, 1))
    lr = tensor2img(data_['lr'], rgb2bgr=False, min_max=(-1, 1))

    plt.figure()
    plt.imshow(gt)
    plt.figure()
    plt.imshow(lr)
    plt.show(block=True)