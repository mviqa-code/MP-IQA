import argparse
import time
from datetime import timedelta

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim as optim
from timm.utils import AverageMeter
from timm.scheduler.cosine_lr import CosineLRScheduler

from config import *
from utils import *
from build_load import *


def main(config):
    # log文件
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name=f"{config.IQA_NAME}")
    create_logger(logger, output_dir=log_dir)
    logger.info(f'log文件保存目录路径: {log_dir}')

    # 设置seed
    set_seed(config.SEED)
    cudnn.benchmark = True

    # 创建数据
    data_loader_train = build_data_loader(config, mode='train')

    # 创建模型
    model = build_MPIQA(config).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"总参数数量: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数数量: {trainable_params}")
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"冻结参数数量: {frozen_params}")

    # 创建优化器、lrScheduler
    optimizer = optim.AdamW(model.parameters(),
                            eps=config.OPTIMIZER.EPS,
                            betas=config.OPTIMIZER.BETAS,
                            lr=config.OPTIMIZER.BASE_LR,
                            weight_decay=config.OPTIMIZER.WEIGHT_DECAY)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=config.EPOCHS,
                                     lr_min=config.SCHEDULER.LR_MIN,
                                     cycle_decay=config.SCHEDULER.CYCLE_DECAY,
                                     warmup_lr_init=config.SCHEDULER.WARMUP_LR,
                                     warmup_t=config.SCHEDULER.WARMUP_EPOCHS)

    # 损失函数
    bce = nn.BCEWithLogitsLoss()
    smoothl1 = nn.SmoothL1Loss()
    loss_meter = AverageMeter()
    loss_scaler = torch.cuda.amp.GradScaler()

    # 训练阶段
    start_time = time.monotonic()
    logger.info('start training')
    for epoch in range(config.EPOCHS):
        logger.info(f'Epoch{epoch+1} training')
        loss_meter.reset()
        lr_scheduler.step(epoch+1)
        model.train()
        for n_iter, (img, bboxes, labels) in enumerate(data_loader_train):
            optimizer.zero_grad()
            img = img.cuda(non_blocking=True)
            bboxes = bboxes.cuda(non_blocking=True)
            gt_masks = load_object_mask(bboxes, (img.shape[2], img.shape[3]), config.MODEL.CLIP.VISION_PATCH_SIZE)
            gt_masks = gt_masks.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=True):
                pred_score, score_anchor, logits_category, logits_location = model(img, train_mode=True)

            if logits_category is not None:
                category_loss = bce(logits_category, labels)
            else:
                category_loss = 0.0
            if logits_location is not None:
                location_loss = bce(logits_location, gt_masks)
            else:
                location_loss = 0.0
            score_loss = smoothl1(pred_score, score_anchor)
            loss = score_loss + config.LAMBDA_1 * category_loss + config.LAMBDA_2 * location_loss

            loss_scaler.scale(loss).backward()
            loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            torch.cuda.synchronize()

            if n_iter % 10 == 0:
                logger.info(
                    f"Epoch[{epoch+1}] Iteration[{n_iter + 1}/{len(data_loader_train)}] Loss[{loss_meter.avg:.3f}] "
                    f"Category_loss[{category_loss:.3f}] Location_loss[{location_loss:.3f}] Score_loss[{score_loss:.3f}] "
                    f"Base Lr[{lr_scheduler._get_lr(epoch)[0]:.2e}]")

        if config.EPOCHS - epoch <= 10:
            torch.save(
                model.state_dict(),
                os.path.join(
                    'checkpoints',
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{str(epoch+1)}.pth'
                )
            )
    end_time = time.monotonic()
    total_time = timedelta(seconds=end_time - start_time)
    logger.info(f'train running time: {total_time}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ours图像质量评价方法的训练参数配置')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/RUOD/train',
        help='训练图像存储目录路径'
    )
    parser.add_argument(
        '--ann_path',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/RUOD/train_annotation/instances_train.json',
        help='训练标注文件(COCO格式)路径'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_results/RUOD/faster-rcnn/resnet50/train/checkpoints/epoch_12.pth',
        help='预训练目标检测模型权重文件路径'
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
    main(config)