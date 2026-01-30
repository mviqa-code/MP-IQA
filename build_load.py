import os
import math
import json
from collections import defaultdict

import cv2
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

from model.clip import clip
from model.clip import model
from model.mpiqa import MPIQA


class DATASET_TRAIN(data.Dataset):
    def __init__(self, image_dir, annotation_path, transform, config):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_bbox_category_dict = defaultdict(list)
        for ann in data['annotations']:
            img_bbox_category_dict[ann['image_id']].append({
                'bbox': ann['bbox'],
                'category_id': ann['category_id']
            })

        id_cat = {}
        for cat in data['categories']:
            id_cat[cat['id']] = cat['name']

        samples = []
        for img in data['images']:
            img_name = img['file_name']
            image_path = os.path.join(image_dir, img_name)
            bbox_category = img_bbox_category_dict.get(img['id'], [])
            bboxes = [i['bbox'] for i in bbox_category]
            category_labels = list({i['category_id'] for i in bbox_category})
            categories = [id_cat[c] for c in category_labels]
            quality_label = img['map']
            samples.append((image_path, bboxes, categories, quality_label))

        self.categories = config.categories
        self.samples = samples
        self.transform = transform
        self.resize = config.DATA.RESIZE
        self.pad = config.DATA.PAD
        self.pad_value = config.DATA.PAD_VALUE

    def __getitem__(self, index):
        img_path, bboxes, categories, quality = self.samples[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        resized_img, scale = _resize(img, self.resize)
        padded_img = _pad(resized_img, self.pad, self.pad_value)
        tensor_img = self.transform(padded_img)

        scaled_bboxes = [[math.ceil(x * scale) for x in bbox] for bbox in bboxes]
        scaled_bboxes = _bbox_xywh_to_xyxy(scaled_bboxes)
        scaled_bboxes = torch.tensor(scaled_bboxes)

        category_labels = [1. if c in categories else 0. for c in self.categories]
        category_labels = torch.tensor(category_labels)

        quality_label = torch.tensor(quality)
        
        return tensor_img, scaled_bboxes, category_labels, quality_label

    def __len__(self) -> int:
        return len(self.samples)


class DATASET_TEST(data.Dataset):
    def __init__(self, image_dir, annotation_path, transform, config):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for img in data['images']:
            image_name = img['file_name']
            image_path = os.path.join(image_dir, image_name)
            quality_label = img['map']
            samples.append((image_path, quality_label))

        self.samples = samples
        self.transform = transform
        self.resize = config.DATA.RESIZE
        self.pad = config.DATA.PAD
        self.pad_value = config.DATA.PAD_VALUE

    def __getitem__(self, index):
        img_path, quality = self.samples[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        resized_img, scale = _resize(img, self.resize)
        padded_img = _pad(resized_img, self.pad, self.pad_value)
        tensor_img = self.transform(padded_img)
        quality_label = torch.tensor(quality)
        return tensor_img, quality_label

    def __len__(self) -> int:
        return len(self.samples)


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
    return resized_img, scale


def _pad(img, pad, pad_value):
    h, w = img.shape[:2]
    pad_h = (pad - h % pad) % pad
    pad_w = (pad - w % pad) % pad
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img


def _bbox_xywh_to_xyxy(bboxes):
    bboxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]
    return bboxes


def build_transform(config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.DATA.NORM_MEAN, std=config.DATA.NORM_STD)
        ]
    )
    return transform


def build_dataset(config, mode):
    if mode == 'train':
        dataset = DATASET_TRAIN(
            image_dir=config.image_dir,
            annotation_path=config.train_ann_path,
            transform=build_transform(config),
            config=config
        )
    elif mode == 'output':
        dataset = DATASET_TEST(
            image_dir=config.image_dir,
            annotation_path=config.test_ann_path,
            transform=build_transform(config),
            config=config
        )
    else:
        raise ValueError(f"mode为{mode}, 既不是'train'，又不是'output'.")
    return dataset


def build_data_loader(config, mode):
    dataset = build_dataset(config, mode)
    if mode == 'train':
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, pin_memory=True)
    elif mode == 'output':
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    else:
        raise ValueError(f"mode为{mode}, 既不是'train'，又不是'output'")
    return data_loader


def build_MPIQA(config):
    url = clip._MODELS[config.MODEL.CLIP.BACKBONE]
    clip_model_path = clip._download(url)
    clip_model = torch.jit.load(clip_model_path).eval()
    clip_model = clip.build_model(clip_model.state_dict())
    for param in clip_model.parameters():
        param.requires_grad = False

    mpiqa = MPIQA(config, clip_model)
    return mpiqa


def load_MPIQA(config):
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
    mpiqa = MPIQA(config, clip_model)
    weights = torch.load(config.weights)
    mpiqa.load_state_dict(weights)
    return mpiqa


def load_object_mask(bboxes_batch, image_size, patch_size):
    img_h, img_w = image_size
    h, w = img_h // patch_size, img_w // patch_size
    masks = []
    for bboxes in bboxes_batch:
        mask = torch.zeros(h, w)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, x2 = x1 // patch_size, x2 // patch_size
            y1, y2 = y1 // patch_size, y2 // patch_size
            mask[y1:y2, x1:x2] = 1.0
        masks.append(mask.view(-1))
    return torch.stack(masks)