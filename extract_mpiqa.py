import argparse

from tqdm import tqdm

from config import *
from utils import *
from build_load import *


def main(config):
    set_seed(config.SEED)
    data_loader_test = build_data_loader(config, mode='output')
    model = load_MPIQA(config).cuda()
    model.eval()

    img_names = []
    quality_scores = {'mpiqa': []}
    with torch.no_grad():
        for img_name, img in tqdm(data_loader_test):
            img = img.cuda()
            pred_score = model(img)
            img_names.append(img_name[0])
            quality_scores['mpiqa'].append(pred_score.item())

    result_dir = 'results'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_ours.txt'), 'w') as f:
        for name, score in zip(img_names, quality_scores['mpiqa']):
            f.write(f"{name}\t{score}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--weights',
        type=str,
        default='/DATA2/chenyuqing_program/MP-IQA-main/checkpoints/trashcan/wo-location/20250710_093958_10.pth',
        help='MPIQA模型权重文件路径'
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