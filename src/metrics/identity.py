import torch
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
from src.losses.id_loss import IDLoss

id_loss = None

@METRIC_REGISTRY.register()
def calculate_identity(img, img2, net='experiments/pretrained_models/ir_se/model_ir_se50.pth', cuda=True, input_is_tensor=False, **kwargs):
    """Calculate ArcFace loss 

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: id loss result.
    """

    global id_loss

    def np2tensor(x):
        """

        Args:
            x: RGB [0 ~ 255] HWC ndarray

        Returns: RGB [-1, 1]

        """
        return torch.tensor((x * 2 / 255.0) - 0.5).permute(2, 0, 1).unsqueeze(0).float()

    # np2tensor
    if not input_is_tensor:
        img = np2tensor(img)
        img2 = np2tensor(img2)

    if id_loss is None:
        id_loss = IDLoss(ckpt=net)

    if torch.cuda.is_available() and cuda:
        img = img.cuda()
        img2 = img2.cuda()
        id_loss = id_loss.cuda()

    with torch.no_grad():
        loss, sim_improvement, id_logs = id_loss(img2, img, img)
    return 1 - loss.view(1).cpu().numpy()[0]

