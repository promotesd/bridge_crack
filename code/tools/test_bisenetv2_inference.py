# F:\大智若愚\桥梁裂缝数据集\bridge_crack\code\tools\test_bisenetv2_inference.py

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from tqdm import tqdm

import ctypes
from ctypes import wintypes, windll

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from models.bisenetv2 import BiSeNetV2
from dataset.bridge_crack_dataset import BridgeCrackDataset, RandomScaleAndCrop
from torch.utils.data import DataLoader

def get_short_path_name(long_path: str) -> str:
    """

    """
    buf = ctypes.create_unicode_buffer(260)  # MAX_PATH=260
    GetShortPathNameW = windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    GetShortPathNameW.restype = wintypes.DWORD

    result = GetShortPathNameW(long_path, buf, len(buf))
    if result > 0 and result < len(buf):
        return buf.value
    else:
        return long_path  # 如果失败就返回原路径


def _update_confmat(confmat, pred, true, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask].astype(int) + pred[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    confmat += hist

def _compute_mIoU_from_confmat(confmat):
    diag = np.diag(confmat)
    sum_rows = confmat.sum(axis=1)
    sum_cols = confmat.sum(axis=0)
    denom = (sum_rows + sum_cols - diag + 1e-10)
    iou = diag / denom
    miou = np.nanmean(iou)
    return miou

def _compute_F1_macro_from_confmat(confmat):
    num_classes = confmat.shape[0]
    diag = np.diag(confmat)
    sum_rows = confmat.sum(axis=1)
    sum_cols = confmat.sum(axis=0)
    F1_list = []
    for c in range(num_classes):
        tp = diag[c]
        fp = sum_cols[c] - tp
        fn = sum_rows[c] - tp
        denom = (2*tp + fp + fn + 1e-10)
        f1_c = 2 * tp / denom
        F1_list.append(f1_c)
    return float(np.mean(F1_list))

def main():
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.random_seed)

    # 1) 指定测试权重路径
    checkpoint_path = r"F:\大智若愚\桥梁裂缝数据集\bridge_crack\code\train_process\bisenetv2_finetune_20250306_113456\epoch_15_fold_1.pth"

    # 2) 加载模型
    model = BiSeNetV2(n_classes=cfg.num_classes, aux_mode='train').to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[Info] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[Warning] checkpoint path not found: {checkpoint_path}")

    model.eval()

    # 3) 构建测试集 DataLoader
    from dataset.bridge_crack_dataset import collect_image_label_paths
    (noise_img_paths, noise_lbl_paths,
     non_steel_img_paths, non_steel_lbl_paths,
     steel_img_paths, steel_lbl_paths,
     synth_img_paths, synth_lbl_paths) = collect_image_label_paths(cfg.data_root)

    # 测试集中：只用 noise + non_steel + steel
    test_img_paths = noise_img_paths + non_steel_img_paths + steel_img_paths
    test_lbl_paths = noise_lbl_paths + non_steel_lbl_paths + steel_lbl_paths

    # 推理时一般不做随机变换
    test_dataset = BridgeCrackDataset(
        test_img_paths,
        test_lbl_paths,
        transform=RandomScaleAndCrop(scale_range=(1.0,1.0), crop_size=(256,256), hflip_prob=0.0)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 4) 结果保存文件夹
    save_dir = r"..\bridge_crack\result\test"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Will save test images to: {save_dir}")

    # 先把这个中文路径也转为短路径
    short_save_dir = get_short_path_name(save_dir)
    print(f"[Debug] short_save_dir => {short_save_dir}")

    # 5) 混淆矩阵
    num_classes = cfg.num_classes
    confmat = np.zeros((num_classes, num_classes), dtype=np.float64)

    # 6) 遍历测试集
    for idx, (imgs, labels) in enumerate(tqdm(test_loader, desc="Testing", leave=True)):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            if isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
            else:
                main_out = outputs

            preds = main_out.argmax(dim=1)

        # 拼接图像并保存
        img_np = imgs[0].cpu().numpy().transpose(1,2,0)  # (H,W,3)
        lbl_np = labels[0].cpu().numpy()
        pred_np = preds[0].cpu().numpy()

        # 恢复可视化(如有归一化需逆变换，这里仅做clip)
        img_np = (img_np * 255.0).clip(0,255).astype(np.uint8)
        lbl_vis = (lbl_np * 255).astype(np.uint8)
        pred_vis = (pred_np * 255).astype(np.uint8)

        lbl_vis = cv2.cvtColor(lbl_vis, cv2.COLOR_GRAY2BGR)
        pred_vis = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

        concat_vis = np.concatenate([img_np, lbl_vis, pred_vis], axis=1)
        h, w, c = concat_vis.shape

        save_name = f"test_{idx:04d}.png"
        # 原长路径
        full_long_path = os.path.join(save_dir, save_name)
        # 转成短路径
        full_short_path = os.path.join(short_save_dir, save_name)

        # Debug
        print(f"[Debug] concat_vis shape={concat_vis.shape}, index={idx}")
        print(f"[Debug] full_long_path={full_long_path}")
        print(f"[Debug] short_path={full_short_path}")

        # 用短路径写文件
        ret = cv2.imwrite(full_short_path, concat_vis)
        print(f"[Debug] cv2.imwrite => success={ret}")

        # 更新混淆矩阵
        _update_confmat(confmat, pred_np, lbl_np, num_classes)

    # 7) 统计 mIoU, F1
    miou_val = _compute_mIoU_from_confmat(confmat)
    f1_val = _compute_F1_macro_from_confmat(confmat)

    # 写入 metrics.txt
    metrics_txt_long = os.path.join(save_dir, "metrics.txt")
    metrics_txt_short = os.path.join(short_save_dir, "metrics.txt")

    with open(metrics_txt_short, 'w', encoding='utf-8') as f:
        f.write(f"Number of test samples: {len(test_dataset)}\n")
        f.write(f"mIoU: {miou_val:.4f}\n")
        f.write(f"F1:   {f1_val:.4f}\n")

    print(f"[Info] Results saved to => {save_dir}")
    print(f"[Info] Final: mIoU={miou_val:.4f}, F1={f1_val:.4f}")


if __name__ == "__main__":
    main()
