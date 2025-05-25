# src/utils/utils.py
from typing import Dict, Any
import logging
import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

def setup_logger(log_dir: str, log_name: str = "train.log"):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode='a')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info(f"Logging setup complete. Log file: {os.path.join(log_dir, log_name)}")

def set_seed(seed: int):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CuDNN 的确定性设置，可能会影响性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 设置为 False 以保证完全确定性

def save_checkpoint(state: Dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint_seg.pth.tar", best_filename: str = "model_best_seg.pth.tar"):
    """保存模型检查点"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_filepath)
        logging.info(f"Best model saved to {best_filepath}")

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info(f"Loaded model weights from {checkpoint_path}")

    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', -np.inf if config['training']['monitor_mode'] == 'max' else np.inf) # Default based on mode

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"Loaded optimizer state from {checkpoint_path}")

    logging.info(f"Resuming training from epoch {start_epoch + 1}, best metric: {best_metric:.4f}")

    return start_epoch, best_metric
