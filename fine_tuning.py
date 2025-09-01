import argparse
import math

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim as optim
from timm.utils import AverageMeter

from config import *
from utils import *
from build_load import *


class DATA_TEST(data.Dataset):
    def __init__(self, image_dir, map_path, transform, config):
        samples = []
        with open(map_path, 'r', encoding='utf-8') as file:
            for line in file:
                a = line.strip().split('\t')
                name, map = a[0], float(a[1])
                map = math.exp(1. - map) - 1.
                image_path = os.path.join(image_dir, name)
                samples.append((image_path, map))

        self.samples = samples
        self.transform = transform
        self.resize = config.DATA.RESIZE
        self.pad = config.DATA.PAD
        self.pad_value = config.DATA.PAD_VALUE

    def __getitem__(self, index):
        img_path, map = self.samples[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        resized_img, scale = resize_(img, self.resize)
        padded_img = pad_(resized_img, self.pad, self.pad_value)
        tensor_img = self.transform(padded_img)
        return tensor_img, torch.tensor(map)

    def __len__(self) -> int:
        return len(self.samples)


def resize_(img, resize):
    h, w = img.shape[:2]
    scale = resize / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_size = (new_w, new_h)
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_img, scale


def pad_(img, pad, pad_value):
    h, w = img.shape[:2]
    pad_h = (pad - h % pad) % pad
    pad_w = (pad - w % pad) % pad
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img


def build_data(config, args):
    dataset = DATA_TEST(
        image_dir=config.image_dir,
        map_path=args.map_path,
        transform=build_transform(config),
        config=config)

    return dataset


def load_data_loader(config, args):
    dataset = build_data(config, args)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, pin_memory=True)
    return data_loader


def load_ours_model(config):
    clip_model = model.CLIP(embed_dim=config.MODEL.CLIP.EMBED_DIM,
                            image_resolution=config.MODEL.CLIP.IMAGE_RESOLUTION,
                            vision_layers=config.MODEL.CLIP.VISION_LAYERS,
                            vision_width=config.MODEL.CLIP.VISION_WIDTH,
                            vision_patch_size=config.MODEL.CLIP.VISION_PATCH_SIZE,
                            context_length=config.MODEL.CLIP.CONTEXT_LENGTH,
                            vocab_size=config.MODEL.CLIP.VOCAB_SIZE,
                            transformer_width=config.MODEL.CLIP.TRANSFORMER_WIDTH,
                            transformer_heads=config.MODEL.CLIP.TRANSFORMER_HEADS,
                            transformer_layers=config.MODEL.CLIP.TRANSFORMER_LAYERS)
    for param in clip_model.parameters():
        param.requires_grad = False
    backbone = resnet.ResNet(depth=config.MODEL.AG_BACKBONE_DEPTH)
    for param in backbone.parameters():
        param.requires_grad = False
    mpiqa = MPIQA(config, clip_model, backbone)
    weights = torch.load(config.weights)
    mpiqa.load_state_dict(weights)
    return mpiqa


def main(config, args):
    set_seed(config.SEED)
    cudnn.benchmark = True
    data_loader_test = load_data_loader(config, args)
    model = load_ours_model(config).cuda()

    optimizer = optim.AdamW(model.parameters(),
                            eps=config.OPTIMIZER.EPS,
                            betas=config.OPTIMIZER.BETAS,
                            lr=0.0003,
                            weight_decay=config.OPTIMIZER.WEIGHT_DECAY)

    smoothl1 = nn.SmoothL1Loss()

    loss_meter = AverageMeter()
    loss_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(10):
        loss_meter.reset()
        model.train()
        for n_iter, (img, map) in enumerate(data_loader_test):
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            map = map.cuda(non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=True):
                pred_score = model(img)
            loss = smoothl1(pred_score, map)
            loss_scaler.scale(loss).backward()
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            torch.cuda.synchronize()
            if n_iter % 10 == 0:
                print(f"Epoch[{epoch + 1}] Iteration[{n_iter + 1}/{len(data_loader_test)}] Loss[{loss_meter.avg:.3f}]")
        torch.save(model.state_dict(),
                   os.path.join('checkpoints',
                                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{str(epoch + 1)}.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/TrashCan/material_version/val',
        help='测试图像存储目录路径'
    )
    parser.add_argument(
        '--ann_path',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/TrashCan/material_version/instances_val_trashcan.json',
        help='测试标注文件(COCO格式)路径'
    )
    parser.add_argument(
        '--map_path',
        type=str,
        default='/DATA2/chenyuqing_program/MP-IQA-main/results/image_map/trashcan/faster-rcnn/map.txt'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='/DATA2/chenyuqing_program/MP-IQA-main/checkpoints/trashcan/wo-location/20250709_154128_30.pth',
        help='mpiqa模型权重文件路径'
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
        default=False,
        help='使用视觉提示学习器'
    )
    args = parser.parse_args()
    config = get_config(args)
    main(config, args)