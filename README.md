# STSR-Task1-Baseline

[EN](./README.md)  [ZH](./README_zh.md)  

## Project Overview

**Challenge Website**: [https://songhen15.github.io/STSdevelop.github.io/miccai2025/index.html](https://songhen15.github.io/STSdevelop.github.io/miccai2025/index.html)  
**Competition Page**: [https://www.codabench.org/competitions/6468/](https://www.codabench.org/competitions/6468/)  

**Challenge Description**:  
Accurate segmentation of root pulp canals in CBCT images allows for clearer visualization of their morphology, branches, and curvatures, facilitating the development of refined filling strategies. However, annotating root pulp canal regions in CBCT images is inherently labor-intensive, requiring substantial time and human resources. In the Miccai STSR 2025 Challenge Task 1, we expand the dataset and provide finer-grained annotations for different teeth (including teeth and corresponding root canals). The segmentation algorithm is expected to accurately segment permanent teeth (including wisdom teeth) and their corresponding root canals.

---

## Project Structure

```
monai_cbct_segmentation/
├── configs/                    # Configuration files
│   └── train_config.yaml       # Main training configuration
├── data/                       # Example data directory (actual data may be elsewhere)
│   ├── images/                 # Raw CBCT images (e.g., patient01_ct.nii.gz)
│   └── labels/                 # Corresponding segmentation labels (e.g., patient01_seg.nii.gz)
├── notebooks/                  # (Optional) Jupyter notebooks for exploration/visualization
│   └── data_exploration.ipynb
├── outputs/                    # Training output directory (auto-created)
│   ├── logs/                   # Training logs
│   ├── checkpoints/            # Model weights
│   └── results/                # (Optional) Validation/test result visualizations
├── src/                        # Source code
│   ├── data_utils/             # Data processing modules
│   │   └── dataset.py          # Dataset definition, data loading, and splitting
│   ├── losses/                 # Loss function definitions (or use MONAI built-in)
│   ├── metrics/                # Evaluation metrics (or use MONAI built-in)
│   ├── models/                 # Model architectures (or use MONAI built-in)
│   ├── transforms/             # MONAI data augmentation and preprocessing
│   │   └── transforms.py       # Define transform pipelines for training/validation
│   ├── utils/                  # Utility functions
│   │   └── utils.py            # Logging, path handling, etc.
│   └── __init__.py
├── train.py                    # Training script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## How to Run

1. **Modify Configuration**: Open `configs/train_config.yaml`, **must** update `data.image_dir` and `data.label_dir` to your actual data paths. Review and adjust parameters like `transforms.spacing`, `transforms.intensity_norm`, `transforms.spatial_crop_size`, and `training.batch_size` (critical for performance and GPU memory usage).
2. **Start Training**: Run in the project root directory:  
   ```bash
   python train.py --config configs/train_config.yaml
   ```
3. **Monitor Training**:  
   - Logs will be printed to the console and saved in `outputs/logs/`.
   - If `logging.use_tensorboard=True`, run `tensorboard --logdir outputs/tb_logs` in a new terminal, then open [http://localhost:6006](http://localhost:6006) in a browser to view loss curves, learning rate, validation metrics, and visualizations.
4. **Check Results**:  
   - Trained model weights saved in `outputs/checkpoints/` (`checkpoint.pth.tar` is the latest, `model_best.pth.tar` is the best on validation).
   - Logs saved in `outputs/logs/`.

---

## Configuration File (`configs/train_config.yaml`)

Core file for managing hyperparameters and settings:

```yaml
# configs/train_config.yaml

project_name: "cbct_segmentation_baseline"

# 1. Data Settings
data:
  image_dir: "/path/to/your/cbct/images"  # <--- Modify to your image path
  label_dir: "/path/to/your/cbct/labels"  # <--- Modify to your label path
  num_classes: 3                         # Background(0), Teeth(1), Root Canal(2)
  val_split: 0.2                         # Validation set ratio
  random_seed: 42                        # Seed for data splitting
  cache_rate: 1.0                        # Cache ratio (1.0=all cached in memory)
                                         # For large datasets, consider PersistentDataset or reduce cache_rate
  num_workers: 4                         # DataLoader workers

# 2. Preprocessing & Augmentation Settings (MONAI Transforms)
transforms:
  # Universal preprocessing
  orientation: "RAS"                     # Standardize orientation
  spacing: [0.5, 0.5, 0.5]               # Resample to target voxel spacing (mm) - adjust per your data!
  intensity_norm:
    a_min: -1000.0                       # Intensity windowing min (adjust per data)
    a_max: 1000.0                        # Intensity windowing max (adjust per data)
    b_min: 0.0                           # Target range min
    b_max: 1.0                           # Target range max
    clip: True
  crop_foreground: True                  # Crop foreground region based on labels
  # Training augmentation
  spatial_crop_size: [96, 96, 96]        # Random patch size for training (adjust per GPU memory)
  rand_crop_pos_ratio: 0.8               # RandCropByPosNegLabeld foreground ratio
  rand_flip_prob: 0.5                    # Random flip probability (left-right)
  rand_rotate90_prob: 0.5                # Random 90-degree rotation probability
  rand_scale_intensity_prob: 0.1         # Random intensity scaling probability
  rand_shift_intensity_prob: 0.1         # Random intensity shifting probability

# 3. Model Settings
model:
  name: "UNet"                           # Model name (MONAI built-in)
  spatial_dims: 3                        # 3D data
  in_channels: 1                         # Input channels (CBCT is grayscale)
  out_channels: 3                        # Output channels (background+teeth+root canal)
  channels: [16, 32, 64, 128, 256]       # UNet layer channels
  strides: [2, 2, 2, 2]                  # UNet downsampling strides
  num_res_units: 2                       # Residual units per block
  norm: "BATCH"                          # Normalization layer (BATCH, INSTANCE)
  dropout: 0.1                           # Dropout rate

# 4. Training Settings
training:
  device: "cuda:0"                       # Training device ("cuda:0", "cpu")
  batch_size: 2                          # Batch size (adjust per GPU memory)
  num_epochs: 100                        # Total epochs
  optimizer: "AdamW"                     # Optimizer (Adam, AdamW, SGD)
  learning_rate: 0.0001                  # Learning rate
  weight_decay: 0.00001                  # Weight decay (AdamW)
  lr_scheduler: "CosineAnnealingLR"      # Scheduler (None, CosineAnnealingLR, ReduceLROnPlateau)
  scheduler_params:                      # Scheduler parameters
    T_max: 100                           # For CosineAnnealingLR (usually = num_epochs)
    # eta_min: 0.000001                  # Optional: for CosineAnnealingLR
    # factor: 0.5                        # For ReduceLROnPlateau
    # patience: 10                       # For ReduceLROnPlateau

  loss_function: "DiceCELoss"            # Loss (DiceLoss, DiceCELoss, FocalLoss)
  loss_params:
    to_onehot_y: True                    # Convert labels to one-hot
    softmax: True                        # Apply Softmax to model outputs
    include_background: False            # Include background in loss calculation

  # Validation Settings
  validation_interval: 5                 # Validate every N epochs
  metrics: ["MeanDice"]                  # Evaluation metrics (MeanDice)
  metric_params:
    include_background: False            # Include background in Dice calculation
    reduction: "mean_batch"              # Dice aggregation method

  # Checkpoint Settings
  checkpoint_dir: "outputs/checkpoints"  # Model save path (relative to project root)
  save_best_only: True                   # Save only the best model
  best_metric: "val_mean_dice"           # Metric name for best model selection
  monitor_mode: "max"                    # "max" or "min" (Dice higher is better)

# 5. Logging Settings
logging:
  log_dir: "outputs/logs"                # Log file path
  log_interval: 50                       # Print training logs every N batches
  use_tensorboard: True                  # Use TensorBoard
  tensorboard_dir: "outputs/tb_logs"     # TensorBoard log path
```

---

## Implementation Details

- **Framework**: Built on MONAI with PyTorch backend, leveraging medical imaging-optimized data loaders, transforms, and models.
- **Data Processing**:
  - `LoadImaged`: Load nii.gz files.
  - `EnsureChannelFirstd`: Ensure data shape `[C, H, W, D]`.
  - `Orientationd`: Standardize physical orientation (RAS).
  - `Spacingd`: Resample to uniform voxel spacing (critical for resolution consistency).
  - `ScaleIntensityRanged`: Windowing and normalization of HU values (adjust `a_min/a_max` per CBCT characteristics).
  - `CropForegroundd`: Remove background to reduce computation.
  - `RandCropByPosNegLabeld`: Random patch sampling with foreground probability control.
  - **Augmentation**: Random flipping, rotation, intensity scaling/shift.
- **Model**:
  - Default: **UNet** (classic medical segmentation architecture). Configurable channels, strides, residual units.
  - Alternative: **VNet** (3D segmentation network).
- **Loss Function**:
  - Default: **DiceCELoss** (DiceLoss + CrossEntropyLoss). More stable than DiceLoss alone.
- **Optimizer & Scheduler**:
  - Default: **AdamW** with **CosineAnnealingLR**.
- **Validation**:
  - **Sliding Window Inference** for full-image evaluation.
  - **DiceMetric** for mean Dice coefficient calculation.
- **Checkpointing**:
  - Saves latest model (`checkpoint.pth.tar`) and best model (`model_best.pth.tar`).
- **Logging & Visualization**:
  - Detailed logging with TensorBoard integration.
- **Caching Strategy**:
  - Supports `CacheDataset` (memory) and `PersistentDataset` (disk).

---

## Additional Tools

- **nnUNet Integration**: This project includes pre-installed nnUNetv2 framework. For nnUNetv2 training, refer to official documentation: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

---

## Future Improvement Directions

- **Hyperparameter Tuning**: Optimize learning rate, optimizer, batch size, patch size, loss weights, and augmentation strength. Tools like Optuna can automate this process.
- **Advanced Models**: Explore nnU-Net, Swin UNETR, or Transformer-based architectures.
- **Augmentation Strategies**: Elastic deformation, contrast adjustment, Gamma correction.
- **Loss Functions**: Focal Loss, Tversky Loss, or hybrid loss combinations.
- **Postprocessing**: Connected component analysis to remove false positives.
- **Ensemble Learning**: Combine predictions from multiple models.
- **Preprocessing**: Denoising, artifact removal.
- **Evaluation Metrics**: Add Hausdorff Distance, Surface Dice.
- **Cross-Validation**: K-Fold cross-validation for robust performance assessment.

---

*Translation completed while preserving original Markdown structure and technical accuracy.*