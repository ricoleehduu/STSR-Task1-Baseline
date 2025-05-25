# train.py
import os
import random
import yaml
import logging
import torch
import numpy as np
from tqdm import tqdm
from monai.data import decollate_batch # 用于分离 batch 数据进行指标计算
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.networks.nets import UNet, VNet, SegResNet # 添加 VNet 选项
from monai.inferers import sliding_window_inference # 用于验证时处理大图像
from monai.utils import set_determinism


from src.data_utils.dataset import prepare_dataloaders
from src.transforms.transforms import get_train_transforms, get_val_transforms, get_post_transforms
from src.utils.utils import setup_logger, set_seed, save_checkpoint, load_checkpoint
from datetime import datetime



# TensorBoard support (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not found. Install with 'pip install tensorboard'")

def train_epoch(model, loader, optimizer, loss_fn, device, epoch, config, tb_writer=None, log_interval=50):
    """训练一个 Epoch"""
    model.train()
    epoch_loss = 0.0
    step = 0
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} [Train]")

    for batch_idx, batch_data in progress_bar:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        

        optimizer.zero_grad()
        outputs = model(inputs)


        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(loader) - 1:
            avg_batch_loss = epoch_loss / step
            progress_bar.set_postfix({
                "loss": f"{avg_batch_loss:.4f}",
                "lr": f"{current_lr:.6f}"
            })
            if tb_writer:
                tb_writer.add_scalar("Loss/train_batch", avg_batch_loss, epoch * len(loader) + batch_idx)
                tb_writer.add_scalar("LR", current_lr, epoch * len(loader) + batch_idx)

    avg_epoch_loss = epoch_loss / len(loader)
    logging.info(f"Epoch {epoch+1} Train Avg Loss: {avg_epoch_loss:.4f}")
    if tb_writer:
         tb_writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)

    return avg_epoch_loss

def validate_epoch(model, loader, loss_fn, metric_fn, post_pred, post_label, device, epoch, config, tb_writer=None):
    """验证一个 Epoch"""
    model.eval()
    val_loss = 0.0
    metric_fn.reset() # 重置指标计算器

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1} [Validate]")

    with torch.no_grad():
        for batch_idx, batch_data in progress_bar:
            val_images, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)

            # 使用滑动窗口推理处理可能的大图像或不同大小的图像
            roi_size = config['transforms']['spatial_crop_size']
            sw_batch_size = config['training'].get('validation_sw_batch_size', 4) # 滑动窗口内部的 batch size
            val_outputs = sliding_window_inference(
                val_images, roi_size, sw_batch_size, model, overlap=0.5, mode='gaussian' # 使用高斯平滑重叠区域
            )

            # 计算验证 Loss (可选，但有助于监控)
            loss = loss_fn(val_outputs, val_labels)
            val_loss += loss.item()

            # 解包 batch -> 单个样本
            val_output_list = decollate_batch(val_outputs)
            val_label_list = decollate_batch(val_labels)

            # 包装成字典 {'pred': ..., 'label': ...} 再送入 transforms
            data_dicts = [{"pred": o, "label": l} for o, l in zip(val_output_list, val_label_list)]

            # 后处理（包括 argmax 和 one-hot）
            val_outputs_post = [post_pred(d) for d in data_dicts]
            val_labels_post = [post_label(d) for d in data_dicts]

             # --- 取出 tensor 列表 ---
            val_outputs_tensor_list = [d["pred"] for d in val_outputs_post]
            val_labels_tensor_list = [d["label"] for d in val_labels_post]

            # --- 计算指标 ---
            metric_fn(y_pred=val_outputs_tensor_list, y=val_labels_tensor_list)


            progress_bar.set_postfix({"val_loss": f"{val_loss / (batch_idx + 1):.4f}"})


    # 聚合指标
    metric_result_tensor = metric_fn.aggregate()
    metric_result = metric_result_tensor.mean().item()

    metric_fn.reset() # 为下一轮验证重置

    avg_val_loss = val_loss / len(loader)

    logging.info(f"Epoch {epoch+1} Validation Avg Loss: {avg_val_loss:.4f}")
    logging.info(f"Epoch {epoch+1} Validation Mean Dice: {metric_result:.4f}")

    if tb_writer:
        tb_writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        tb_writer.add_scalar("Dice/validation", metric_result, epoch)
        # 可视化一些样本 (可选)
        if epoch % config['training'].get('vis_interval', 10) == 0: # 每 10 个 epoch 可视化一次
            try:
                # 取第一个验证样本进行可视化
                first_val_output = val_outputs_post[0]['pred'] # 获取 one-hot 预测
                first_val_label = val_labels_post[0]['label'] # 获取 one-hot 标签
                first_val_image = val_images[0] # 获取原始图像

                # 选择中间切片进行可视化 (z 轴)
                mid_slice = first_val_image.shape[-1] // 2

                # 叠加预测和标签到图像上 (需要转换为 RGB)
                # 注意：这里的可视化比较基础，可以用 matplotlib 或其他库做得更精细
                img_slice = first_val_image[0, :, :, mid_slice].cpu().numpy()
                pred_slice = torch.argmax(first_val_output, dim=0)[ :, :, mid_slice].cpu().numpy() # 转为类别索引
                label_slice = torch.argmax(first_val_label, dim=0)[ :, :, mid_slice].cpu().numpy() # 转为类别索引

                # 创建一个简单的 RGB 图像用于 TensorBoard
                vis_image = np.stack([img_slice] * 3, axis=0) # 灰度图转 RGB
                # 在预测和标签位置上色 (简单示例：牙齿-红色, 根管-绿色)
                vis_image[0, pred_slice == 1] = 1.0 # Red channel for teeth pred
                vis_image[1, pred_slice == 2] = 1.0 # Green channel for root canal pred
                vis_image[2, label_slice != 0] = 0.5 # Blue channel for ground truth (semi-transparent)

                tb_writer.add_image(f"Validation/sample_{batch_idx}_slice_{mid_slice}",
                                    torch.tensor(vis_image), global_step=epoch, dataformats='CHW')
            except Exception as e:
                logging.warning(f"Could not visualize validation sample: {e}")

    return avg_val_loss, metric_result


def main(config_path: str):
    """主训练函数"""
    # --- 1. 加载配置 ---
    with open(config_path, 'r',encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded:")
    print(yaml.dump(config, indent=2))

    # --- 2. 设置环境 ---
    set_seed(config['data']['random_seed'])
    log_dir = config['logging']['log_dir']
    setup_logger(log_dir, f"{config['project_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 设置 TensorBoard
    tb_writer = None
    if TENSORBOARD_AVAILABLE and config['logging']['use_tensorboard']:
        tb_log_dir = config['logging'].get('tensorboard_dir', os.path.join(log_dir, 'tb_logs'))
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logging.info(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # --- 3. 数据加载 ---
    logging.info("Preparing data loaders...")
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    dataloaders = prepare_dataloaders(config, train_transforms, val_transforms)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    logging.info("Data loaders ready.")
    
    # --- 4. 模型初始化 ---
    logging.info(f"Initializing model: {config['model']['name']}")
    model_name = config['model'].get('name', 'UNet').lower()
    if model_name == 'unet':
        model = UNet(
            spatial_dims=config['model']['spatial_dims'],
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            channels=config['model']['channels'],
            strides=config['model']['strides'],
            num_res_units=config['model']['num_res_units'],
            norm=config['model']['norm'],
            dropout=config['model'].get('dropout', 0.0)
        ).to(device)
    elif model_name == 'vnet':
         model = VNet(
            spatial_dims=config['model']['spatial_dims'],
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            dropout_prob=config['model'].get('dropout', 0.5) # VNet uses dropout_prob
            # Add other VNet specific parameters from config if needed
        ).to(device)
    elif model_name == 'segresnet':
        model = SegResNet(
            spatial_dims=config['model']['spatial_dims'],
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            init_filters=config['model'].get('init_filters', 16),
            blocks_down=config['model'].get('blocks_down', [1, 2, 2, 4]),
            blocks_up=config['model'].get('blocks_up', [1, 1, 1]),
            dropout_prob=config['model'].get('dropout_prob', 0.2)
        ).to(device)
    else:
        raise ValueError(f"Unsupported model name: {config['model']['name']}")
    logging.info("Model initialized.")
    # 打印模型结构和参数量 (可选)
    # from torchsummary import summary
    # summary(model, input_size=(config['model']['in_channels'], *config['transforms']['spatial_crop_size']))


    # --- 5. 损失函数、优化器、学习率调度器 ---
    logging.info(f"Setting up loss function: {config['training']['loss_function']}")
    loss_name = config['training']['loss_function'].lower()
    loss_params = config['training'].get('loss_params', {})
    if loss_name == 'diceceloss':
        loss_function = DiceCELoss(**loss_params)
    elif loss_name == 'diceloss':
        loss_function = DiceLoss(**loss_params)
    elif loss_name == 'focalloss':
         loss_function = FocalLoss(**loss_params) # 需要确保参数匹配
    else:
        raise ValueError(f"Unsupported loss function: {config['training']['loss_function']}")

    logging.info(f"Setting up optimizer: {config['training']['optimizer']}")
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    wd = config['training'].get('weight_decay', 0)
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # Adam 也有 weight_decay
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")

    scheduler = None
    scheduler_name = config['training'].get('lr_scheduler')
    if scheduler_name:
        logging.info(f"Setting up LR scheduler: {scheduler_name}")
        scheduler_params = config['training'].get('scheduler_params', {})
        if scheduler_name.lower() == 'cosineannealinglr':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_name.lower() == 'reducelronplateau':
             # 注意：ReduceLROnPlateau 需要在验证步骤后 step
             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config['training']['monitor_mode'], **scheduler_params)
        else:
            logging.warning(f"Unsupported LR scheduler: {scheduler_name}. No scheduler will be used.")

    # --- 6. 评估指标 和 后处理 ---
    logging.info(f"Setting up metrics: {config['training']['metrics']}")
    metric_name = config['training']['metrics'][0].lower() # 假设只有一个主要指标用于 best model
    metric_params = config['training'].get('metric_params', {})
    if metric_name == 'meandice':
        dice_metric = DiceMetric(**metric_params)
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    # 获取后处理 transforms (用于将模型输出转为 one-hot 格式给 metric)
    post_pred = get_post_transforms(config) # Post transforms for predictions
    post_label = get_post_transforms(config) # Post transforms for labels

    

    # --- 7. 训练循环 ---
    num_epochs = config['training']['num_epochs']
    val_interval = config['training']['validation_interval']
    checkpoint_dir = config['training']['checkpoint_dir']
    best_metric_name = config['training']['best_metric']
    monitor_mode = config['training']['monitor_mode']

    best_metric_val = -np.inf if monitor_mode == 'max' else np.inf
    start_epoch = 0

    # --- 可选：加载检查点 ---
    resume_checkpoint = config['training'].get('resume_checkpoint')
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            start_epoch, best_metric_val = load_checkpoint(resume_checkpoint, model, optimizer)
            logging.info(f"Resuming training from epoch {start_epoch + 1}")
        except Exception as e:
            logging.error(f"Could not load checkpoint: {e}. Starting training from scratch.")

    logging.info("Starting training loop...")
    for epoch in range(start_epoch, num_epochs):
        # --- Train ---
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, epoch, config, tb_writer, config['logging']['log_interval'])
        
        # --- LR Scheduler Step (Epoch-based like CosineAnnealing) ---
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # --- Validate ---
        if (epoch + 1) % val_interval == 0 or epoch == num_epochs - 1:
            val_loss, current_metric_val = validate_epoch(model, val_loader, loss_function, dice_metric, post_pred, post_label, device, epoch, config, tb_writer)

            # --- LR Scheduler Step (Validation-based like ReduceLROnPlateau) ---
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(current_metric_val) # 或者使用 val_loss

            # --- Save Checkpoint ---
            is_best = False
            if monitor_mode == 'max' and current_metric_val > best_metric_val:
                best_metric_val = current_metric_val
                is_best = True
                logging.info(f"New best {best_metric_name}: {best_metric_val:.4f} at epoch {epoch + 1}")
            elif monitor_mode == 'min' and current_metric_val < best_metric_val: # e.g., if monitoring loss
                best_metric_val = current_metric_val
                is_best = True
                logging.info(f"New best {best_metric_name}: {best_metric_val:.4f} at epoch {epoch + 1}")

            checkpoint_data = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_metric_val,
                'config': config # 保存配置以供参考
            }
            save_checkpoint(checkpoint_data, is_best, checkpoint_dir)

            if config['training']['save_best_only'] and not is_best:
                # 如果只保存最好的，并且当前不是最好的，可以选择删除旧的 'checkpoint.pth.tar'
                # 但通常保留最新的 checkpoint 也是有用的
                pass
        else:
             logging.info(f"Epoch {epoch+1} completed. Skipping validation in this epoch.")


    logging.info("Training finished.")
    logging.info(f"Best Validation {best_metric_name}: {best_metric_val:.4f}")

    # --- 8. 清理 / 结束 ---
    if tb_writer:
        tb_writer.close()

    # 如果使用 PersistentDataset，清理缓存 (可选)
    if config['data'].get('use_persistent_dataset', False) and config['data'].get('clean_persistent_cache', False):
         try:
            import shutil
            cache_dir = config['data'].get('persistent_cache_dir', './persistent_cache')
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                logging.info(f"Removed PersistentDataset cache directory: {cache_dir}")
         except Exception as e:
             logging.warning(f"Could not remove persistent cache directory: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MONAI CBCT Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to the training configuration file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    main(args.config)