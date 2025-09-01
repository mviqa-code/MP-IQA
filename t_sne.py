import argparse

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import *
from build_load import *


class Extractor:
    def __init__(self, model):
        self.model = model
        self._semantic_feature_hooks()

    def _semantic_feature_hooks(self):
        self.model.multimodal_decoder.ln.register_forward_hook(self._save_feature)

    def _save_feature(self, module, input, output):
        opt = torch.mean(output, dim=1)
        self.feature = opt

    def __call__(self, img):
        opt = self.model(img)
        return self.feature


def load_label(path):
    _dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            l = line.strip().split('\t')
            name, label = l[0], int(float(l[1]) * 10) if float(l[1]) < 1.0 else 9
            _dict[name] = label
    return _dict


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = build_data_loader(config, mode='output')
    model = load_MPIQA(config).eval().to(device)
    ext = Extractor(model)
    l_dict = load_label(args.map_file)
    f_dict = {}

    with torch.no_grad():
        for img_name, img in tqdm(data_loader):
            _feature = ext(img.to(device))
            _feature = _feature.squeeze().detach().cpu().numpy()
            f_dict[img_name[0]] = _feature

    feat_label = {k: (l_dict[k], f_dict[k]) for k in l_dict}
    array_labels = [v[0] for v in feat_label.values()]
    array_labels = np.array(array_labels)
    array_features = [v[1] for v in feat_label.values()]
    array_features = np.array(array_features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(array_features)
    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=200, # 700
                max_iter=2000,
                verbose=1)
    X_tsne_2d = tsne.fit_transform(features_scaled)

    plt.scatter(X_tsne_2d[:, 0],
                X_tsne_2d[:, 1],
                c=array_labels,
                s=5,
                cmap='rainbow',
                alpha=1)
    plt.axis('off')
    plt.savefig('t-sne.svg', bbox_inches='tight', pad_inches=0)
    plt.savefig('t-sne.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t-SNE')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='/DATA2/chenyuqing_program/mmdetection-main/my_datasets/TrashCan/material_version/val',
        help='图像存储目录路径'
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
        default='/DATA2/chenyuqing_program/MP-IQA-main/checkpoints/trashcan/visual-only/20250708_213142_30.pth',
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
    parser.add_argument(
        '--map_file',
        default='/DATA2/chenyuqing_program/MP-IQA-main/results/image_map/trashcan/faster-rcnn/map.txt'
    )
    args = parser.parse_args()
    config = get_config(args)
    main(args, config)




