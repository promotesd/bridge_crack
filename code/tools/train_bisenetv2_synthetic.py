# F:\大智若愚\桥梁裂缝数据集\bridge_crack\code\tools\train_bisenetv2_synthetic.py

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

# 新增：导入 tqdm
from tqdm import tqdm

# 用相对路径引用项目内的模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.bridge_crack_dataset import (
    collect_image_label_paths,
    create_kfold_datasets,
    RandomScaleAndCrop
)
from config.config import Config
from models.bisenetv2 import BiSeNetV2

# 从bridge_metrics.py导入指标评估函数
from bridge_metrics import evaluate_metrics


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    单个epoch的训练逻辑：计算交叉熵loss并反向传播
    """
    model.train()
    total_loss = 0.0

    # 在训练循环外层加上 tqdm
    for imgs, labels in tqdm(dataloader, desc="Training Epoch", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # BiSeNetV2在train模式下会输出[main_out, aux2, aux3, ...]

        main_out = outputs[0]  # 只取第一个主输出计算主loss
        loss = criterion(main_out, labels)

        # BiSeNetV2 通常有4个aux输出(若网络结构不变的话)，常见做法: loss += 0.4 * aux_loss
        # 0.4 是一个常用、合理的加权系数，有助于在训练早期稳定收敛
        for aux_out in outputs[1:]:
            aux_loss = criterion(aux_out, labels)
            loss += 0.4 * aux_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_one_epoch(model, dataloader, criterion, device):
    """
    验证集上计算 val_loss，并使用 evaluate_metrics 评估 mIoU、F1、FPS、FLOPs、Params 等。
    返回 (val_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0

    # 在验证循环外层加上 tqdm
    for imgs, labels in tqdm(dataloader, desc="Validation Epoch", leave=False):
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            main_out = outputs[0]
            loss = criterion(main_out, labels)
            total_loss += loss.item()

    val_loss = total_loss / len(dataloader)

    # 调用 evaluate_metrics 计算 mIoU、F1、FPS、FLOPs、Params
    # 注意：evaluate_metrics 会重新跑一遍 dataloader，以获取完整预测
    metrics_dict = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=2,           # 2类：背景+裂缝；或从cfg.num_classes
        input_size=(1,3,256,256),# 用于统计FLOPs的dummy输入大小，可根据实际情况修改
        warmup=2,                # 可根据需要调小调大
        max_iter=len(dataloader) # 完整遍历验证集
    )

    return val_loss, metrics_dict


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.random_seed)

    # 1. 仅获取合成数据集的图像与标签
    (noise_img_paths, noise_lbl_paths,
     non_steel_img_paths, non_steel_lbl_paths,
     steel_img_paths, steel_lbl_paths,
     synth_img_paths, synth_lbl_paths) = collect_image_label_paths(cfg.data_root)

    all_img_paths = synth_img_paths
    all_lbl_paths = synth_lbl_paths

    # 2. 进行K折数据集拆分
    kfold_datasets = create_kfold_datasets(
        all_img_paths, all_lbl_paths,
        n_splits=cfg.k_folds,
        shuffle=True,
        random_seed=cfg.random_seed,
        transform=RandomScaleAndCrop(scale_range=(0.75,1.25), crop_size=(256,256))
    )

    # 3. 开始K折训练
    start_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # 日志目录
    log_dir = os.path.join('..', 'train_process', f'bisenetv2_synthetic_{start_time_str}')
    os.makedirs(log_dir, exist_ok=True)

    fold_results = []
    for fold_idx, (train_dataset, val_dataset) in enumerate(kfold_datasets):
        print(f"\n=== Fold {fold_idx+1}/{cfg.k_folds} ===")

        # 初始化模型
        model = BiSeNetV2(n_classes=cfg.num_classes, aux_mode='train')
        model = model.to(device)

        # 优化器 & 损失
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        # DataLoader
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
            'val_mIoU': [],
            'val_F1': [],
            'val_FPS': [],
            'val_FLOPs': [],
            'val_Params': [],
        }

        for epoch in range(cfg.num_epochs):
            # 训练1个epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            # 验证
            val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)
            # val_metrics包含: { "mIoU", "F1", "FPS", "FLOPs", "Params" }
            val_miou = val_metrics["mIoU"]
            val_f1   = val_metrics["F1"]
            val_fps  = val_metrics["FPS"]
            val_flops= val_metrics["FLOPs"]
            val_params=val_metrics["Params"]

            print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val mIoU: {val_miou:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val FPS: {val_fps:.2f}")

            # 保存到history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mIoU'].append(val_miou)
            history['val_F1'].append(val_f1)
            history['val_FPS'].append(val_fps)
            history['val_FLOPs'].append(val_flops)
            history['val_Params'].append(val_params)

            # 更新best_val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(log_dir, f"best_model_fold_{fold_idx+1}.pth")
                )

            # 也可定期保存
            if (epoch+1) % cfg.save_interval == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(log_dir, f"epoch_{epoch+1}_fold_{fold_idx+1}.pth")
                )

        # fold结束，保存历史记录
        fold_results.append({
            'fold': fold_idx+1,
            'best_val_loss': best_val_loss,
            'history': history
        })

        # 将每个fold的训练过程写入csv
        csv_path = os.path.join(log_dir, f"train_log_fold_{fold_idx+1}.csv")
        with open(csv_path, 'w') as f:
            # 写表头
            f.write("epoch,train_loss,val_loss,val_mIoU,val_F1,val_FPS,val_FLOPs,val_Params\n")
            for e in range(cfg.num_epochs):
                f.write(f"{e+1},{history['train_loss'][e]:.4f},"
                        f"{history['val_loss'][e]:.4f},"
                        f"{history['val_mIoU'][e]:.4f},"
                        f"{history['val_F1'][e]:.4f},"
                        f"{history['val_FPS'][e]:.2f},"
                        f"{history['val_FLOPs'][e]:.2f},"
                        f"{history['val_Params'][e]:.2f}\n")

    # 所有fold结束，做一个简单汇总
    print("\nTraining completed. Summary:")
    for r in fold_results:
        print(f"Fold {r['fold']} => best_val_loss = {r['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
