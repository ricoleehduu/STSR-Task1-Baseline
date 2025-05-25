# src/transforms/transforms.py
from typing import Dict, Any
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd,
    ToTensord, EnsureTyped, NormalizeIntensityd, RandGaussianNoised,
    AsDiscreted # 用于验证后处理
)

KEYS = ["image", "label"]

def get_train_transforms(config: Dict[str, Any]) -> Compose:
    """获取训练数据预处理和增强流程"""
    cfg_t = config['transforms']
    return Compose([
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes=cfg_t['orientation']),
        Spacingd(
            keys=KEYS,
            pixdim=cfg_t['spacing'],
            mode=("bilinear", "nearest") # 图像用双线性插值，标签用最近邻
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg_t['intensity_norm']['a_min'],
            a_max=cfg_t['intensity_norm']['a_max'],
            b_min=cfg_t['intensity_norm']['b_min'],
            b_max=cfg_t['intensity_norm']['b_max'],
            clip=cfg_t['intensity_norm']['clip']
        ),
        CropForegroundd(keys=KEYS, source_key="label", margin=10) if cfg_t['crop_foreground'] else lambda x: x, # 条件应用
        # --- Augmentations ---
        RandCropByPosNegLabeld(
            keys=KEYS,
            label_key="label",
            spatial_size=cfg_t['spatial_crop_size'],
            pos=cfg_t['rand_crop_pos_ratio'], # 正样本比例 (前景)
            neg=1.0 - cfg_t['rand_crop_pos_ratio'], # 负样本比例 (背景)
            num_samples=config['training'].get('num_samples_per_volume', 4), # 每个 volume 采样多少 patch
            image_key="image",
            image_threshold=0, # 基于图像强度阈值定义前景 (可选)
        ),
        RandFlipd(keys=KEYS, prob=cfg_t['rand_flip_prob'], spatial_axis=0), # 左右翻转
        RandRotate90d(keys=KEYS, prob=cfg_t['rand_rotate90_prob'], max_k=3, spatial_axes=(0, 1)), # xy 平面旋转
        # RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1), # 可选高斯噪声
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=cfg_t['rand_scale_intensity_prob']),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=cfg_t['rand_shift_intensity_prob']),
        # --- Final Steps ---
        EnsureTyped(keys=KEYS, dtype=np.float32), # 确保类型正确
        ToTensord(keys=KEYS)
    ])

def get_val_transforms(config: Dict[str, Any]) -> Compose:
    """获取验证数据预处理流程 (通常没有随机增强)"""
    cfg_t = config['transforms']
    return Compose([
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes=cfg_t['orientation']),
        Spacingd(
            keys=KEYS,
            pixdim=cfg_t['spacing'],
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg_t['intensity_norm']['a_min'],
            a_max=cfg_t['intensity_norm']['a_max'],
            b_min=cfg_t['intensity_norm']['b_min'],
            b_max=cfg_t['intensity_norm']['b_max'],
            clip=cfg_t['intensity_norm']['clip']
        ),
        CropForegroundd(keys=KEYS, source_key="label", margin=10) if cfg_t['crop_foreground'] else lambda x: x,
        # --- Final Steps ---
        EnsureTyped(keys=KEYS, dtype=np.float32),
        ToTensord(keys=KEYS)
    ])

# 用于验证/推断后处理，将模型输出 (logits) 转换为分割图 (类别索引) 和 one-hot 编码
def get_post_transforms(config: Dict[str, Any]) -> Compose:
    num_classes = config['data']['num_classes']
    return Compose([
        EnsureTyped(keys=["pred", "label"]),
        AsDiscreted(keys=["pred"], argmax=True, to_onehot=num_classes),
        AsDiscreted(keys=["label"], to_onehot=num_classes),
    ])