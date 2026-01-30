from yacs.config import CfgNode as CN

_C = CN()
_C.IQA_NAME = 'MP-IQA'
_C.SEED = 1024
_C.EPOCHS = 30
_C.LAMBDA_1 = 0.01
_C.LAMBDA_2 = 0.01

_C.DATA = CN()
_C.DATA.NORM_MEAN = [0.485, 0.456, 0.406]
_C.DATA.NORM_STD = [0.229, 0.224, 0.225]
_C.DATA.RESIZE = 224
_C.DATA.BATCH_SIZE = 4
_C.DATA.PAD = 32
_C.DATA.PAD_VALUE = [0, 0, 0]

_C.OPTIMIZER = CN()
_C.OPTIMIZER.EPS = 1e-8
_C.OPTIMIZER.BETAS = (0.9, 0.999)
_C.OPTIMIZER.BASE_LR = 0.0003
_C.OPTIMIZER.WEIGHT_DECAY = 0.001

_C.SCHEDULER = CN()
_C.SCHEDULER.LR_MIN = 0.0000002
_C.SCHEDULER.CYCLE_DECAY = 0.1
_C.SCHEDULER.WARMUP_LR = 0.00001
_C.SCHEDULER.WARMUP_EPOCHS = 5

_C.MODEL = CN()
_C.MODEL.DIM = 512
_C.MODEL.VISION_WIDTH = 768
_C.MODEL.DROPOUT = 0.1
_C.MODEL.CTX_DIM = 512
_C.MODEL.N_CTX_C = 8
_C.MODEL.N_CTX_L = 4
_C.MODEL.MMD_NHEAD = 8
_C.MODEL.MMD_NUM_LAYERS = 6

_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.BACKBONE = "ViT-B-16"
_C.MODEL.CLIP.EMBED_DIM = 512
_C.MODEL.CLIP.IMAGE_RESOLUTION = 224
_C.MODEL.CLIP.VISION_LAYERS = 12
_C.MODEL.CLIP.VISION_WIDTH = 768
_C.MODEL.CLIP.VISION_PATCH_SIZE = 16
_C.MODEL.CLIP.CONTEXT_LENGTH = 77
_C.MODEL.CLIP.VOCAB_SIZE = 49408
_C.MODEL.CLIP.TRANSFORMER_WIDTH = 512
_C.MODEL.CLIP.TRANSFORMER_HEADS = 8
_C.MODEL.CLIP.TRANSFORMER_LAYERS = 12


def update_config(config, args):
    config.defrost()
    if hasattr(args, 'image_dir'):
        config.image_dir = args.image_dir
    if hasattr(args, 'train_ann_path'):
        config.train_ann_path = args.train_ann_path
    if hasattr(args, 'test_ann_path'):
        config.test_ann_path = args.test_ann_path
    if hasattr(args, 'categories'):
        config.categories = args.categories
    if hasattr(args, 'weights'):
        config.weights = args.weights
    if hasattr(args, 'result_dir'):
        config.result_dir = args.result_dir
    if hasattr(args, 'textual'):
        config.textual = args.textual
    if hasattr(args, 'visual'):
        config.visual = args.visual
    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config
