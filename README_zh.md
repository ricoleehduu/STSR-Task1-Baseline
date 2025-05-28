# STSR-Task1-Baseline

[EN](./README.md)  [ZH](./README_zh.md)  

## 项目概述

**挑战赛主页**：https://songhen15.github.io/STSdevelop.github.io/miccai2025/index.html

**赛事主页**：https://www.codabench.org/competitions/6468/

**赛事描述**：对牙根髓管进行精确分割可以更清晰地显示其形态、分支和弯曲，从而有助于制定更精细的填充策略。然而，在 CBCT 图像中标注牙根髓管区域本身就是一项劳动密集型工作，需要投入大量的时间和人力资源。在Miccai STSR 2025挑战任务1中，我们扩大数据集并为不同的牙齿（包括牙齿和相应的根毛管）提供更细粒度的注释。预计该分割算法将准确地分割恒牙（包括智慧牙齿）和相应的根毛管。

## 项目结构

```
monai_cbct_segmentation/
├── configs/                    # 配置文件目录
│   └── train_config.yaml       # 训练主配置文件
├── data/                       # 数据存放示例目录 (实际数据可能在别处)
│   ├── images/                 # 存放原始 CBCT 图像 (e.g., patient01_ct.nii.gz)
│   └── labels/                 # 存放对应的分割标签 (e.g., patient01_seg.nii.gz)
├── notebooks/                  # (可选) Jupyter notebooks 用于探索、可视化
│   └── data_exploration.ipynb
├── outputs/                    # 训练输出目录 (自动创建)
│   ├── logs/                   # 训练日志
│   ├── checkpoints/            # 模型权重文件
│   └── results/                # (可选) 验证/测试结果可视化
├── src/                        # 源代码目录
│   ├── data_utils/             # 数据处理相关模块
│   │   └── dataset.py          # 数据集定义、数据加载、划分
│   ├── losses/                 # 损失函数定义 (或使用 MONAI 内置)
│   ├── metrics/                # 评估指标定义 (或使用 MONAI 内置)
│   ├── models/                 # 模型架构定义 (或使用 MONAI 内置)
│   ├── transforms/             # MONAI 数据增强和预处理流程
│   │   └── transforms.py       # 定义训练和验证的 transform pipelines
│   ├── utils/                  # 通用工具函数
│   │   └── utils.py            # 日志记录、路径处理等
│   └── __init__.py
├── train.py                    # 训练主脚本
├── requirements.txt            # Python 依赖库
└── README.md                   # 项目说明文档
```

## 如何运行

1. **修改配置**: 打开 `configs/train_config.yaml`，**务必**修改 `data.image_dir` 和 `data.label_dir` 为你实际的数据路径。仔细检查并调整其他参数，特别是 `transforms.spacing`, `transforms.intensity_norm`, `transforms.spatial_crop_size`, `training.batch_size` 等，这些参数对性能和显存消耗影响很大。
2. **开始训练**: 在项目根目录下运行： `bash python train.py --config configs/train_config.yaml`
3. 监控训练:
   - 控制台会输出日志信息（也会保存在 `outputs/logs/` 目录下）。
   - 如果 `logging.use_tensorboard` 设置为 `True`，可以在另一个终端中运行 `tensorboard --logdir outputs/tb_logs`，然后在浏览器中打开 TensorBoard 页面 (通常是 `http://localhost:6006`) 查看损失、学习率、验证指标以及可视化结果。
4. 查看结果:
   - 训练好的模型权重会保存在 `outputs/checkpoints/` 目录下（`checkpoint.pth.tar` 是最新的，`model_best.pth.tar` 是验证集上效果最好的）。
   - 日志文件保存在 `outputs/logs/`。

## 配置文件 (`configs/train_config.yaml`)

这是管理所有超参数和设置的核心文件。

```yaml
# configs/train_config.yaml

project_name: "cbct_segmentation_baseline"

# 1. Data Settings
data:
  image_dir: "/path/to/your/cbct/images"  # <--- 修改为你的图像路径
  label_dir: "/path/to/your/cbct/labels"  # <--- 修改为你的标签路径
  num_classes: 3                         # 背景(0), 牙齿(1), 根管(2)
  val_split: 0.2                         # 验证集比例
  random_seed: 42                        # 数据划分随机种子
  cache_rate: 1.0                        # 缓存数据的比例 (1.0=全部缓存到内存, 0=不缓存)
                                         # 对于大数据集, 考虑 PersistentDataset 或减小 cache_rate
  num_workers: 4                         # DataLoader 工作进程数

# 2. Preprocessing & Augmentation Settings (MONAI Transforms)
transforms:
  # 通用预处理
  orientation: "RAS"                     # 标准化方向
  spacing: [0.5, 0.5, 0.5]               # 重采样到指定体素间距 (mm) - 根据你的数据调整!
  intensity_norm:
    a_min: -1000.0                       # Intensity windowing min (根据数据调整)
    a_max: 1000.0                        # Intensity windowing max (根据数据调整)
    b_min: 0.0                           # Target range min
    b_max: 1.0                           # Target range max
    clip: True
  crop_foreground: True                  # 是否基于标签裁剪前景区域
  # 训练时数据增强
  spatial_crop_size: [96, 96, 96]        # 训练时随机裁剪的 patch 大小 (根据 GPU 显存调整)
  rand_crop_pos_ratio: 0.8               # RandCropByPosNegLabeld 正样本比例
  rand_flip_prob: 0.5                    # 随机翻转概率 (左右)
  rand_rotate90_prob: 0.5                # 随机90度旋转概率
  rand_scale_intensity_prob: 0.1         # 随机缩放强度概率
  rand_shift_intensity_prob: 0.1         # 随机偏移强度概率

# 3. Model Settings
model:
  name: "UNet"                           # 模型名称 (MONAI 内置)
  spatial_dims: 3                        # 3D 数据
  in_channels: 1                         # 输入通道数 (CBCT 通常为 1)
  out_channels: 3                        # 输出通道数 (背景+牙齿+根管)
  channels: [16, 32, 64, 128, 256]       # UNet 各层通道数
  strides: [2, 2, 2, 2]                  # UNet 下采样步长
  num_res_units: 2                       # 每个 UNet block 中的残差单元数
  norm: "BATCH"                          # Normalization layer (BATCH, INSTANCE)
  dropout: 0.1                           # Dropout rate

# 4. Training Settings
training:
  device: "cuda:0"                       # 训练设备 ("cuda:0", "cpu")
  batch_size: 2                          # 批大小 (根据 GPU 显存调整)
  num_epochs: 100                        # 训练轮数
  optimizer: "AdamW"                     # 优化器 (Adam, AdamW, SGD)
  learning_rate: 0.0001                  # 学习率
  weight_decay: 0.00001                  # 权重衰减 (AdamW)
  lr_scheduler: "CosineAnnealingLR"      # 学习率调度器 (None, CosineAnnealingLR, ReduceLROnPlateau)
  scheduler_params:                      # 调度器参数 (根据选择的调度器)
    T_max: 100                           # For CosineAnnealingLR (通常等于 num_epochs)
    # eta_min: 0.000001                  # Optional: for CosineAnnealingLR
    # factor: 0.5                        # For ReduceLROnPlateau
    # patience: 10                       # For ReduceLROnPlateau

  loss_function: "DiceCELoss"            # 损失函数 (DiceLoss, DiceCELoss, FocalLoss)
  loss_params:
    to_onehot_y: True                  # 将标签转换为 one-hot
    softmax: True                      # 对模型输出应用 Softmax
    include_background: False          # 计算 Loss 时是否包含背景类 (通常不包含)

  # Validation Settings
  validation_interval: 5                 # 每隔多少 epoch 验证一次
  metrics: ["MeanDice"]                  # 评估指标 (MeanDice)
  metric_params:
    include_background: False          # 计算 Dice 时是否包含背景类
    reduction: "mean_batch"            # Dice 指标聚合方式

  # Checkpoint Settings
  checkpoint_dir: "outputs/checkpoints"  # 模型保存路径 (相对于项目根目录)
  save_best_only: True                   # 只保存验证集上效果最好的模型
  best_metric: "val_mean_dice"           # 用于判断最佳模型的指标名称 (需与 metrics 对应)
  monitor_mode: "max"                    # "max" 或 "min" (Dice 越高越好)

# 5. Logging Settings
logging:
  log_dir: "outputs/logs"                # 日志文件保存路径
  log_interval: 50                       # 每隔多少 batch 打印一次训练日志
  use_tensorboard: True                  # 是否使用 TensorBoard
  tensorboard_dir: "outputs/tb_logs"     # TensorBoard 日志路径
```

## 说明和 Baseline 细节

- **框架**: 使用 MONAI 作为核心框架，利用其针对医疗影像优化的数据加载、变换、模型和工具。底层使用 PyTorch。
- 数据处理:
  - `LoadImaged`: 加载 nii.gz 文件。
  - `EnsureChannelFirstd`: 确保数据维度是 `[C, H, W, D]`。
  - `Orientationd`: 将所有数据旋转到统一的物理方向（如 RAS），这对于 CBCT 很重要，因为采集方向可能不同。
  - `Spacingd`: 将所有数据重采样到统一的体素间距，消除分辨率差异的影响。**这是非常关键的一步，`pixdim` 需要根据你的数据特性仔细设置**。
  - `ScaleIntensityRanged`: 对图像进行灰度值窗口化和归一化，将感兴趣的灰度范围映射到 [0, 1]。**`a_min`, `a_max` 需要根据 CBCT 的 HU 值范围调整**。
  - `CropForegroundd`: 基于标签（或图像强度）裁剪掉大部分背景区域，减少计算量。
  - `RandCropByPosNegLabeld`: 在训练时，从原始图像中随机裁剪出固定大小 (e.g., 96x96x96) 的 Patch 进行训练，可以有效利用显存并增加数据多样性。`pos`/`neg` 参数控制采样时包含前景的概率。
  - **数据增强**: 随机翻转、旋转、强度缩放/偏移等增加模型鲁棒性。
- 模型:
  - 默认使用 **UNet**，这是一个经典的医疗影像分割模型。提供了通道数、步长、残差单元等配置。也加入了 **VNet** 选项，它是另一种常用的 3D 分割网络。
  - 输入通道为 1 (灰度 CBCT)，输出通道为 3 (背景、牙齿、根管)。
- 损失函数:
  - 默认使用 **DiceCELoss** (`DiceLoss` + `CrossEntropyLoss`)。这种组合通常比单独使用 DiceLoss 更稳定，尤其是在训练早期或类别不平衡时。`to_onehot_y=True` 和 `softmax=True` 是必需的，因为我们的标签是类别索引，而模型输出是 logits。`include_background=False` 通常在计算分割损失时排除背景类。
- 优化器与调度器:
  - 默认 **AdamW**，一种常用的带有权重衰减的优化器。
  - 默认 **CosineAnnealingLR** 学习率调度器，可以在训练过程中平滑地降低学习率。也提供了 `ReduceLROnPlateau` 选项。
- 验证:
  - 使用 **Sliding Window Inference** 对整个验证图像进行预测，避免了验证时也需要裁剪 Patch 的问题，能更准确地评估模型在完整图像上的性能。
  - 使用 **DiceMetric** 计算平均 Dice 系数（前景类别的平均值）。Dice 是衡量分割任务重叠度的常用指标。
  - 通过 `post_pred` 和 `post_label` 将模型输出和标签转换为 one-hot 格式，以符合 `DiceMetric` 的输入要求。
- 检查点:
  - 保存最新的模型 (`checkpoint.pth.tar`) 和在验证集上表现最好的模型 (`model_best.pth.tar`)。
  - 支持从检查点恢复训练 (`resume_checkpoint` 参数)。
- 日志与可视化:
  - 使用 Python `logging` 模块记录详细日志。
  - 集成 TensorBoard，可视化训练/验证损失、学习率、验证 Dice 以及部分验证样本的分割结果切片。
- 缓存策略:
  - 提供了 `CacheDataset` (内存缓存) 和 `PersistentDataset` (磁盘缓存) 的选项，根据数据集大小和可用内存选择。`PersistentDataset` 更适合非常大的数据集。

## 额外工具说明

- **nnUnet**：本项目中已经额外安装并支持使用 nnUNetv2 框架，使用 nnUNetv2 训练请参考其官方文档：https://github.com/MIC-DKFZ/nnUNet.

## 后续改进方向

- **超参数调优**: 学习率、优化器、Batch Size、Patch Size、Loss 函数权重、数据增强强度等都需要根据实际数据和效果进行调整。可以使用 Optuna 等工具进行自动化调优。
- **更复杂的模型**: 尝试其他先进的分割网络，如 nnU-Net (可能需要适配其框架或借鉴其思路)、Swin UNETR 等 Transformer 기반 模型。
- **数据增强策略**: 探索更高级的数据增强方法，如弹性变形 (Elastic Deformation)、对比度调整 (Contrast Adjustment)、Gamma 校正等。
- **损失函数**: 针对类别不平衡问题，可以尝试 Focal Loss、Tversky Loss 或组合不同的损失函数。
- **后处理**: 加入连通组件分析等后处理步骤，去除小的假阳性预测。
- **集成学习**: 训练多个模型并集成它们的预测结果。
- **数据预处理**: 探索更精细的去噪、伪影去除等预处理方法。
- **评估指标**: 增加其他评估指标，如 Hausdorff Distance、Surface Dice 等，更全面地评估分割质量。
- **交叉验证**: 使用 K-Fold 交叉验证获得更可靠的模型性能评估。