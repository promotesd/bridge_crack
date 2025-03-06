import os
import sys
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1) 导入 tqdm 以显示进度条
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.bridge_crack_dataset import (
    collect_image_label_paths,
    create_kfold_datasets,
    RandomScaleAndCrop
)
from config.config import Config
from models.Unetmodel1 import EyeCRNet

# 2) 从 bridge_metrics.py 导入评估函数
from bridge_metrics import evaluate_metrics



def calculate_metrics(pred, label, num_classes=2):
    """
    保留原先简易的mIoU计算，不做改动。
    """
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        label_c = (label == c).float()
        inter = (pred_c * label_c).sum()
        union = (pred_c + label_c).sum() - inter
        if union == 0:
            iou = 0
        else:
            iou = inter / union
        ious.append(iou.item())
    miou = sum(ious) / len(ious)
    return miou

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    保持原逻辑，只在循环外添加 tqdm 进度条。
    """
    model.train()
    total_loss = 0.0
    # tqdm 包裹训练循环
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # EyeCRNet输出 [N, num_classes, H, W]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    """
    保持原逻辑，只在循环外添加 tqdm 进度条。
    """
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    # tqdm 包裹验证循环
    for imgs, labels in tqdm(dataloader, desc="Validation", leave=False):
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            miou = calculate_metrics(preds, labels, num_classes=2)
            total_miou += miou

    val_loss = total_loss / len(dataloader)
    val_miou = total_miou / len(dataloader)
    return val_loss, val_miou

def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.random_seed)

    # 1. 获取真实数据 (Noise + Non-steel + Steel)
    (noise_img_paths, noise_lbl_paths,
     non_steel_img_paths, non_steel_lbl_paths,
     steel_img_paths, steel_lbl_paths,
     synth_img_paths, synth_lbl_paths) = collect_image_label_paths(cfg.data_root)

    # 合并真实数据
    all_img_paths = noise_img_paths + non_steel_img_paths + steel_img_paths
    all_lbl_paths = noise_lbl_paths + non_steel_lbl_paths + steel_lbl_paths

    # 2. K折拆分
    kfold_datasets = create_kfold_datasets(
        all_img_paths, all_lbl_paths,
        n_splits=cfg.k_folds,
        shuffle=True,
        random_seed=cfg.random_seed,
        transform=RandomScaleAndCrop(scale_range=(0.75,1.25), crop_size=(256,256))
    )

    start_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('..', 'train_process', f'eyecrnet_finetune_{start_time_str}')
    os.makedirs(log_dir, exist_ok=True)

    # 如果要加载合成数据集预训练权重，在此指定
    pretrained_weight = None

    fold_results = []
    for fold_idx, (train_dataset, val_dataset) in enumerate(kfold_datasets):
        print(f"=== EyeCRNet Finetune Fold {fold_idx+1}/{cfg.k_folds} ===")

        model = EyeCRNet(n_classes=cfg.num_classes)
        model = model.to(device)

        if pretrained_weight is not None:
            state = torch.load(pretrained_weight, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weight from {pretrained_weight}")

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                                shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=False, persistent_workers=False # 每轮后销毁 worker
                                )

        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou_simple': [],
            'val_mIoU_eval': [],
            'val_F1': [],
            'val_FPS': [],
            'val_FLOPs': [],
            'val_Params': []
        }

        for epoch in range(cfg.num_epochs):
            # 原有训练逻辑
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            # 原有验证逻辑
            val_loss, val_miou_simple = validate_one_epoch(model, val_loader, criterion, device)

            # 3) 使用 bridge_metrics.py 的 evaluate_metrics 做详细指标统计
            #    (会再次遍历 val_loader，但不改变原有的 val_loss/val_miou_simple 计算)
            val_metrics = evaluate_metrics(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=2,            # 背景 + 裂缝
                input_size=(1,3,256,256), # 用于统计FLOPs
                warmup=2,
                max_iter=len(val_loader)
            )
            val_miou_eval = val_metrics['mIoU']
            val_f1        = val_metrics['F1']
            val_fps       = val_metrics['FPS']
            val_flops     = val_metrics['FLOPs']
            val_params    = val_metrics['Params']

            print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val mIoU(simple): {val_miou_simple:.4f} | "
                  f"Val mIoU(eval): {val_miou_eval:.4f} | "
                  f"F1: {val_f1:.4f} | FPS: {val_fps:.2f}")

            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_miou_simple'].append(val_miou_simple)
            history['val_mIoU_eval'].append(val_miou_eval)
            history['val_F1'].append(val_f1)
            history['val_FPS'].append(val_fps)
            history['val_FLOPs'].append(val_flops)
            history['val_Params'].append(val_params)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(),
                           os.path.join(log_dir, f"best_model_fold_{fold_idx+1}.pth"))

            if (epoch+1) % cfg.save_interval == 0:
                torch.save(model.state_dict(),
                           os.path.join(log_dir, f"epoch_{epoch+1}_fold_{fold_idx+1}.pth"))

        fold_results.append({
            'fold': fold_idx+1,
            'best_val_loss': best_val_loss,
            'history': history
        })

        # 写日志
        csv_path = os.path.join(log_dir, f"finetune_log_fold_{fold_idx+1}.csv")
        with open(csv_path, 'w') as f:
            # 增加了详细列
            f.write("epoch,train_loss,val_loss,val_miou_simple,val_miou_eval,val_F1,val_FPS,val_FLOPs,val_Params\n")
            for e in range(cfg.num_epochs):
                f.write(
                    f"{e+1},"
                    f"{history['train_loss'][e]:.4f},"
                    f"{history['val_loss'][e]:.4f},"
                    f"{history['val_miou_simple'][e]:.4f},"
                    f"{history['val_mIoU_eval'][e]:.4f},"
                    f"{history['val_F1'][e]:.4f},"
                    f"{history['val_FPS'][e]:.2f},"
                    f"{history['val_FLOPs'][e]:.2f},"
                    f"{history['val_Params'][e]:.2f}\n"
                )

    print("EyeCRNet Finetuning completed.")
    for r in fold_results:
        print(f"Fold {r['fold']} best val loss = {r['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()
