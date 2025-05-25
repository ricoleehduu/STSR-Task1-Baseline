import os
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, ToTensord, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.utils import set_determinism

# === 推理配置 ===
model_path = r"C:\Users\Administrator\Desktop\miccai_STS25_c1\monai_cbct_segmentation\outputs\checkpoints\model_best.pth.tar"  # 最佳模型路径
input_image_path = r"E:\DATASET\A-MICCAI-Challenge-Task1\image\image\1031.nii.gz"    # 推理图像路径
output_save_path = "C:/Users/Administrator/Desktop/miccai_STS25_c1/monai_cbct_segmentation/outputs/results/patient1031_pred.nii.gz"  # 保存预测输出
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 数据预处理参数（需与你 config.yaml 中一致）===
spacing = (0.5, 0.5, 0.5)
a_min, a_max = -1000.0, 1000.0
b_min, b_max = 0.0, 1.0
orientation = "RAS"
roi_size = (96, 96, 96)  # patch size
sw_batch_size = 4
num_classes = 13

# === 加载模型 ===
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
    channels=[16, 32, 64, 128, 256],
    strides=[2, 2, 2, 2],
    num_res_units=2,
    norm='BATCH'
).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# === 图像加载与预处理 ===
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes=orientation),
    Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
    ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
    ToTensord(keys=["image"])
])

# 输入应为 dict
data_dict = transforms({"image": input_image_path})
img = data_dict["image"].unsqueeze(0).to(device)  # 添加 batch 维度
affine = data_dict["image_meta_dict"]["affine"]   # 获取原始 affine，用于保存


# === 推理 ===
with torch.no_grad():
    pred = sliding_window_inference(img, roi_size, sw_batch_size, model, overlap=0.5, mode="gaussian")

# === 后处理 & 保存结果 ===
post_pred = Compose([
    AsDiscrete(argmax=True)
])
pred = post_pred(pred)[0].cpu().numpy().astype(np.uint8)

# 使用原图 affine 保存预测 nii
nib.save(nib.Nifti1Image(pred, affine), output_save_path)
print(f"预测结果已保存至: {output_save_path}")
