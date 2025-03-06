# F:\大智若愚\桥梁裂缝数据集\bridge_crack\code\tools\train_eyecrnet_synthetic.py

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

# 保持原有相对路径导入方式
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.bridge_crack_dataset import (
    collect_image_label_paths,
    create_kfold_datasets,
    RandomScaleAndCrop
)
from config.config import Config
# 引入 EyeCRNet
from models.Unetmodel1 import EyeCRNet

# 2) 从 bridge_metrics.py 导入高级指标评估函数
from bridge_metrics import evaluate_metrics



def calculate_metrics(pred, label, num_classes=2):
    """
    保留原先简单的 IoU 计算函数，不做改动。
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
    单个 epoch 的训练逻辑，增加 tqdm 进度条 + 若有多输出则进行辅助加权。
    """
    model.train()
    total_loss = 0.0

    # tqdm 包裹训练循环
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # EyeCRNet 假设输出 [N, num_classes, H, W]
                              # 或在将来扩展为 [main_out, aux1, aux2, ...]
        # 判断是否多输出
        if isinstance(outputs, (list, tuple)):
            # main_out是第一个
            main_out = outputs[0]
            loss = criterion(main_out, labels)
            # 对 aux 输出做加权 (如 0.4)，仅当确有多输出时
            for aux_out in outputs[1:]:
                aux_loss = criterion(aux_out, labels)
                loss += 0.4 * aux_loss
        else:
            # 如果是单一输出
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    """
    验证集上计算 val_loss、val_mIoU(简易版)，并调用 evaluate_metrics 再统计更多指标。
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
            if isinstance(outputs, (list, tuple)):
                main_out = outputs[0]
            else:
                main_out = outputs

            loss = criterion(main_out, labels)
            total_loss += loss.item()

            preds = main_out.argmax(dim=1)
            miou = calculate_metrics(preds, labels, num_classes=2)
            total_miou += miou

    val_loss = total_loss / len(dataloader)
    val_miou_simple = total_miou / len(dataloader)

    # 额外再调用 evaluate_metrics 做更丰富的指标统计 (mIoU, F1, FPS, FLOPs, Params等)
    metrics_dict = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=2,            # 2类(背景+裂缝)，或 cfg.num_classes
        input_size=(1,3,256,256), # 统计FLOPs时使用的输入形状，可根据实际情况调整
        warmup=2,
        max_iter=len(dataloader)  # 遍历整个验证集
    )

    return val_loss, val_miou_simple, metrics_dict

def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.random_seed)

    # 收集数据集路径
    (noise_img_paths, noise_lbl_paths,
     non_steel_img_paths, non_steel_lbl_paths,
     steel_img_paths, steel_lbl_paths,
     synth_img_paths, synth_lbl_paths) = collect_image_label_paths(cfg.data_root)

    # 仅合成数据
    all_img_paths = synth_img_paths
    all_lbl_paths = synth_lbl_paths

    # K折数据集拆分
    kfold_datasets = create_kfold_datasets(
        all_img_paths, all_lbl_paths,
        n_splits=cfg.k_folds,
        shuffle=True,
        random_seed=cfg.random_seed,
        transform=RandomScaleAndCrop(scale_range=(0.75,1.25), crop_size=(256,256))
    )

    # 创建日志目录
    start_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('..', 'train_process', f'eyecrnet_synthetic_{start_time_str}')
    os.makedirs(log_dir, exist_ok=True)

    fold_results = []
    for fold_idx, (train_dataset, val_dataset) in enumerate(kfold_datasets):
        print(f"=== EyeCRNet Synthetic Fold {fold_idx+1}/{cfg.k_folds} ===")

        model = EyeCRNet(n_classes=cfg.num_classes).to(device)

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
            # 训练
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            # 验证
            val_loss, val_miou_simple, metrics_dict = validate_one_epoch(model, val_loader, criterion, device)
            # metrics_dict 包含 {mIoU, F1, FPS, FLOPs, Params}
            val_miou_eval = metrics_dict['mIoU']
            val_f1        = metrics_dict['F1']
            val_fps       = metrics_dict['FPS']
            val_flops     = metrics_dict['FLOPs']
            val_params    = metrics_dict['Params']

            print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val mIoU(simple): {val_miou_simple:.4f} | "
                  f"Val mIoU(eval): {val_miou_eval:.4f} | "
                  f"F1: {val_f1:.4f} | FPS: {val_fps:.2f}")

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
                torch.save(
                    model.state_dict(),
                    os.path.join(log_dir, f"best_model_fold_{fold_idx+1}.pth")
                )

            if (epoch+1) % cfg.save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(log_dir, f"epoch_{epoch+1}_fold_{fold_idx+1}.pth")
                )

        fold_results.append({
            'fold': fold_idx+1,
            'best_val_loss': best_val_loss,
            'history': history
        })

        # 写csv日志
        csv_path = os.path.join(log_dir, f"train_log_fold_{fold_idx+1}.csv")
        with open(csv_path, 'w') as f:
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

    print("EyeCRNet Synthetic Training completed.")
    for r in fold_results:
        print(f"Fold {r['fold']} best val loss = {r['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
