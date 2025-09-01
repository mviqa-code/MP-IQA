import argparse
from functools import partial

import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

from config import *
from build_load import *


def reshape_transform(tensor, height, width, visual):
    if visual:
        result = tensor[:, 2:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    else:
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def _resize(img, resize):
    h, w = img.shape[:2]
    scale = resize / min(h, w)
    if h < w:
        new_h = resize
        new_w = math.ceil(w * scale)
    else:
        new_w = resize
        new_h = math.ceil(h * scale)
    new_size = (new_w, new_h)
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_img


def _pad(img, pad, pad_value):
    h, w = img.shape[:2]
    pad_h = (pad - h % pad) % pad
    pad_w = (pad - w % pad) % pad
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad_h, pad_w


def main(config, args):
    model = load_MPIQA(config).cuda()
    model.eval()
    target_layers = [model.image_encoder]
    targets = [ClassifierOutputTarget(0)]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.DATA.NORM_MEAN, std=config.DATA.NORM_STD)
        ]
    )
    images = os.listdir(args.image_dir)
    os.makedirs(args.save_path, exist_ok=True)
    for name in tqdm(images):
        img = cv2.cvtColor(cv2.imread(os.path.join(args.image_dir, name)), cv2.COLOR_BGR2RGB)
        resized_img = _resize(img, config.DATA.RESIZE)
        np_img = np.float32(resized_img) / 255
        padded_img, pad_h, pad_w = _pad(resized_img, 16, config.DATA.PAD_VALUE)
        tensor_img = transform(padded_img)
        tensor_img = tensor_img.unsqueeze(0).cuda()
        height = int(tensor_img.shape[-2] / config.MODEL.CLIP.VISION_PATCH_SIZE)
        width = int(tensor_img.shape[-1] / config.MODEL.CLIP.VISION_PATCH_SIZE)
        custom_reshape = partial(reshape_transform, height=height, width=width, visual=config.visual)
        cam = EigenCAM(model, target_layers, reshape_transform=custom_reshape)
        grayscale_cam = cam(input_tensor=tensor_img, targets=targets)
        grayscale_cam = grayscale_cam[0, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]
        cam_img = show_cam_on_image(np_img, grayscale_cam)
        cv2.imwrite(os.path.join(args.save_path, f'cam_{name}'), cam_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eigen_CAM')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/VOC0712/test',
    )
    parser.add_argument(
        '--ann_path',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/VOC0712/test_annotation/instances_test.json',
        help='测试标注文件(COCO格式)路径'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='/DATA2/chenyuqing_program/MP-IQA-main/checkpoints/voc0712/mpiqa/20250707_180453_10.pth',
        help='MPIQA模型权重文件路径'
    )
    parser.add_argument(
        '--save_path',
        default='/DATA2/chenyuqing_program/MP-IQA-main/results/cam/voc0712'
    )
    parser.add_argument(
        '--textual',
        type=bool,
        default=True,
        help='使用文本提示学习器'
    )
    parser.add_argument(
        '--visual',
        type=bool,
        default=True,
        help='使用视觉提示学习器'
    )

    args = parser.parse_args()
    config = get_config(args)
    main(config, args)