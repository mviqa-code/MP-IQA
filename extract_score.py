import argparse

from tqdm import tqdm
from scipy import stats

from config import *
from build_load import *


def main(config):
    data_loader_test = build_data_loader(config, mode='output')
    model = load_MPIQA(config).cuda()
    model.eval()
    pd_scores = []
    gt_scores = []
    with torch.no_grad():
        for (img, utility) in tqdm(data_loader_test):
            img = img.cuda()
            pd_score = model(img)
            pd_scores.append(pd_score.item())
            gt_scores.append(utility.item())
    plcc, _ = stats.pearsonr(pd_scores, gt_scores)
    srcc, _ = stats.spearmanr(pd_scores, gt_scores)
    print(f"PLCC[{plcc}], SRCC[{srcc}]")

    os.makedirs(config.result_dir, exist_ok=True)
    with open(os.path.join(config.result_dir, f"{config.image_dir.split('/')[1]}_results.txt"), 'w') as f:
        for gt, pd in zip(gt_scores, pd_scores):
            f.write(f"{gt}\t{pd}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/RUOD/images',
        help='训练图像存储目录路径'
    )
    parser.add_argument(
        '--test_ann_path',
        type=str,
        default='data/RUOD/metas/test_metas.json',
        help='测试标注文件(COCO格式)路径'
    )
    parser.add_argument(
        '--categories',
        # default=[
        #     'person', 'bird', 'cat', 'cow', 'dog',
        #     'horse', 'sheep', 'aeroplane', 'bicycle', 'boat',
        #     'bus', 'car', 'motorbike', 'train', 'bottle',
        #     'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        # ]   # VOC0712
        # default=[
        #     "araneae", "coleoptera", "diptera", "hemiptera", "hymenoptera", "lepidoptera", "odonata"
        # ] # ArTaxOr
        # default=[
        #     'rov', 'plant', 'animal_fish', 'animal_starfish', 'animal_shells',
        #     'animal_crab', 'animal_eel', 'animal_etc', 'trash_etc', 'trash_fabric',
        #     'trash_fishing_gear', 'trash_metal', 'trash_paper', 'trash_plastic', 'trash_rubber',
        #     'trash_wood'
        # ] # TrashCan
        default=[
            'holothurian', 'echinus', 'scallop', 'starfish', 'fish',
            'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish'
        ] # RUOD
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='checkpoints/mpiqa_ruod.pth',
        help='MPIQA模型权重文件路径'
    )
    parser.add_argument(
        '--result_dir',
        default='results'
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