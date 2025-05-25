# src/data_utils/dataset.py
import os
import glob
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
import numpy as np
from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
from monai.utils import first

def get_data_files(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    """
    扫描图像和标签目录，创建文件字典列表。
    假设图像和标签文件名对应（例如，去除 '_ct'/'_seg' 后相同）。
    """
    images = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(label_dir, "*.nii.gz")))

    data_dicts = []
    label_map = {os.path.basename(p).replace("_seg", "").replace("_label", ""): p for p in labels} # 稍微灵活一点的匹配

    for img_path in images:
        img_name = os.path.basename(img_path).replace("_ct", "").replace("_image", "")
        if img_name in label_map:
            data_dicts.append({"image": img_path, "label": label_map[img_name]})
        else:
            print(f"Warning: No label found for image {img_path}")

    print(f"Found {len(data_dicts)} image/label pairs.")
    if not data_dicts:
        raise ValueError("No matching image/label pairs found. Check directories and naming conventions.")
    return data_dicts

# def get_data_files(root_dir: str) -> List[Dict[str, str]]:
#     """
#     扫描 KiTS19 数据集结构，提取图像和标签配对。
#     目录结构如下：
#     root_dir/
#         case_00000/
#             imaging.nii.gz
#             segmentation.nii.gz
#         case_00001/
#             ...
#     """


#     case_dirs = sorted(glob.glob(os.path.join(root_dir, "case_*")))[:5]
    
#     data_dicts = []

#     for case_dir in case_dirs:
#         img_path = os.path.join(case_dir, "imaging.nii.gz")
#         label_path = os.path.join(case_dir, "segmentation.nii.gz")
#         if os.path.exists(img_path) and os.path.exists(label_path):
#             data_dicts.append({"image": img_path, "label": label_path})
#         else:
#             print(f"Warning: Missing files in {case_dir}")

#     print(f"Found {len(data_dicts)} image/label pairs.")
#     return data_dicts

def prepare_dataloaders(config: Dict[str, Any], train_transforms, val_transforms) -> Dict[str, DataLoader]:
    """
    准备训练和验证的 DataLoader。
    """
    data_files = get_data_files(config['data']['image_dir'], config['data']['label_dir'])


    train_files, val_files = train_test_split(
        data_files,
        test_size=config['data']['val_split'],
        random_state=config['data']['random_seed']
    )

    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    # --- 选择 Dataset 类型 ---
    # CacheDataset: 将所有数据加载到内存，适合内存充足且数据量不超大的情况
    # PersistentDataset: 将处理后的数据缓存到磁盘，适合大数据集或内存不足，需要指定 cache_dir
    cache_rate = config['data'].get('cache_rate', 1.0)
    num_workers = config['data'].get('num_workers', 4)

    # 检查是否使用 PersistentDataset
    use_persistent_dataset = config['data'].get('use_persistent_dataset', False)
    persistent_cache_dir = config['data'].get('persistent_cache_dir', './persistent_cache')

    if use_persistent_dataset:
        if not os.path.exists(persistent_cache_dir):
            os.makedirs(persistent_cache_dir)
        print(f"Using PersistentDataset with cache dir: {persistent_cache_dir}")
        train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=os.path.join(persistent_cache_dir, 'train'))
        val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=os.path.join(persistent_cache_dir, 'val'))
    elif cache_rate > 0:
         print(f"Using CacheDataset with cache_rate: {cache_rate}")
         train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
         val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    else:
        print("Using standard Dataset (no caching)")
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)


    # --- 创建 DataLoader ---
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # 如果使用GPU，通常建议开启
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1, # 验证时通常 batch_size 为 1
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 检查数据加载 (可选)
    # check_ds = Dataset(data=train_files, transform=train_transforms)
    # check_loader = DataLoader(check_ds, batch_size=1)
    # first_item = first(check_loader)
    # print("First item shapes - Image:", first_item["image"].shape, "Label:", first_item["label"].shape)


    return {"train": train_loader, "val": val_loader}