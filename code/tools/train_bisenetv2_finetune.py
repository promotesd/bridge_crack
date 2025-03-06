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

# 1) 添加 tqdm 用于进度条
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.bridge_crack_dataset import (
    collect_image_label_paths,
    create_kfold_datasets,
    RandomScaleAndCrop
)
from config.config import Config
from models.bisenetv2 import BiSeNetV2

# 2) 从 bridge_metrics 导入高级指标评估函数
from bridge_metrics import evaluate_metrics


def calculate_metrics(pred, label, num_classes=2):
    """
    原先的mIoU计算函数，不做改动大逻辑。
    修正: 避免 'int' object has no attribute 'item'，将 iou 转为纯 Python float
    pred, label 均为 torch.Tensor (N,H,W) 或 (H,W) 时，需要先保证 shape 兼容
    这里只是简单地写成 2D => 2D case
    """
    ious = []
    for c in range(num_classes):
        # 这里 pred_c, label_c 都是 bool -> float -> sum => scalar
        pred_c = (pred == c).float()
        label_c = (label == c).float()
        inter = (pred_c * label_c).sum().item()   # 这样就拿到 python float
        union = (pred_c + label_c).sum().item() - inter
        if union == 0:
            iou = 0.0
        else:
            iou = inter / union  # python float
        ious.append(iou)
    miou = sum(ious) / len(ious)
    return float(miou)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    单个epoch的训练逻辑，增加了 tqdm 进度条 + aux输出加权。
    """
    model.train()
    total_loss = 0.0

    # 用 tqdm 包裹 dataloader
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # BiSeNetV2在train模式下会输出 [main_out, aux1, aux2, ...]

        main_out = outputs[0]
        loss = criterion(main_out, labels)

        # 若辅助输出可提高精度，则对各aux输出加权
        for aux_out in outputs[1:]:
            aux_loss = criterion(aux_out, labels)
            loss += 0.4 * aux_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    """
    验证集上计算 val_loss 和 val_miou (基于原有简单逻辑)。
    之后再调用 evaluate_metrics 做更丰富的指标统计。
    """
    model.eval()
    total_loss = 0.0
    total_miou = 0.0

    # 用 tqdm 包裹 dataloader
    for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)

            main_out = outputs[0]
            loss = criterion(main_out, labels)
            total_loss += loss.item()

            preds = main_out.argmax(dim=1)
            # 这里 preds, labels 都是 torch.Tensor (N,H,W)
            # 需要在 batch 内再计算
            # 也可以只取 batch 平均
            # 这里只是做简单加和
            # 1) 先for
            for b_idx in range(preds.shape[0]):
                iou_val = calculate_metrics(preds[b_idx], labels[b_idx], num_classes=2)
                total_miou += iou_val

    # 计算简单验证指标
    val_loss = total_loss / len(dataloader)
    # total_miou / (len(dataloader)*batch_size)
    # 由于 total_miou 是对dataloader里所有图像的 iou累加, 需除以图像总数
    # 1) 先算图像总数
    total_images = len(dataloader.dataset)
    # 2) batch_size = dataloader.batch_size 也行
    # 这里就用 total_images
    average_miou = total_miou / float(total_images)

    # 调用 evaluate_metrics 获取 mIoU, F1, FPS, FLOPs, Params 等
    # 注意：evaluate_metrics 会再次遍历 val_loader 以统计完整预测
    metrics_dict = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=2,            # 2类(背景+裂缝)，或改 cfg.num_classes
        input_size=(1,3,256,256), # 用于统计FLOPs的输入形状，可酌情修改
        warmup=2,
        max_iter=len(dataloader)  # 遍历完整验证集
    )

    return val_loss, average_miou, metrics_dict


def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(cfg.random_seed)

    # 1. 获取真实数据 (Noise + Non-steel + Steel)，合并
    (noise_img_paths, noise_lbl_paths,
     non_steel_img_paths, non_steel_lbl_paths,
     steel_img_paths, steel_lbl_paths,
     synth_img_paths, synth_lbl_paths) = collect_image_label_paths(cfg.data_root)

    all_img_paths = noise_img_paths + non_steel_img_paths + steel_img_paths
    all_lbl_paths = noise_lbl_paths + non_steel_lbl_paths + steel_lbl_paths

    # 2. K折数据集
    kfold_datasets = create_kfold_datasets(
        all_img_paths, all_lbl_paths,
        n_splits=cfg.k_folds,
        shuffle=True,
        random_seed=cfg.random_seed,
        transform=RandomScaleAndCrop(scale_range=(0.75,1.25), crop_size=(256,256))
    )

    # 创建日志目录
    start_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('..', 'train_process', f'bisenetv2_finetune_{start_time_str}')
    os.makedirs(log_dir, exist_ok=True)

    # 如果想加载合成数据集预训练的权重，在此指定路径
    pretrained_weight = r'F:\大智若愚\桥梁裂缝数据集\bridge_crack\code\train_process\bisenetv2_synthetic_20250305_220720\epoch_20_fold_2.pth'
    fold_results = []

    for fold_idx, (train_dataset, val_dataset) in enumerate(kfold_datasets):
        print(f"=== Finetune Fold {fold_idx+1}/{cfg.k_folds} ===")

        model = BiSeNetV2(n_classes=cfg.num_classes, aux_mode='train')
        model = model.to(device)
        if pretrained_weight is not None:
            state = torch.load(pretrained_weight, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weight from {pretrained_weight}")

        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=False,
            persistent_workers=False # 每轮后销毁 worker
        )

        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_mIoU2': [],  # evaluate_metrics返回的mIoU
            'val_F1': [],
            'val_FPS': [],
            'val_FLOPs': [],
            'val_Params': []
        }

        for epoch in range(cfg.num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_miou, metrics_dict = validate_one_epoch(model, val_loader, criterion, device)

            val_miou2 = metrics_dict['mIoU']
            val_f1    = metrics_dict['F1']
            val_fps   = metrics_dict['FPS']
            val_flops = metrics_dict['FLOPs']
            val_params= metrics_dict['Params']

            print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val mIoU(simple): {val_miou:.4f} | "
                  f"Val mIoU(eval): {val_miou2:.4f} | "
                  f"F1: {val_f1:.4f} | "
                  f"FPS: {val_fps:.2f}")

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_miou'].append(val_miou)
            history['val_mIoU2'].append(val_miou2)
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

        csv_path = os.path.join(log_dir, f"finetune_log_fold_{fold_idx+1}.csv")
        with open(csv_path, 'w') as f:
            f.write("epoch,train_loss,val_loss,val_miou_simple,val_miou_eval,val_F1,val_FPS,val_FLOPs,val_Params\n")
            for e in range(cfg.num_epochs):
                f.write(
                    f"{e+1},"
                    f"{history['train_loss'][e]:.4f},"
                    f"{history['val_loss'][e]:.4f},"
                    f"{history['val_miou'][e]:.4f},"
                    f"{history['val_mIoU2'][e]:.4f},"
                    f"{history['val_F1'][e]:.4f},"
                    f"{history['val_FPS'][e]:.2f},"
                    f"{history['val_FLOPs'][e]:.2f},"
                    f"{history['val_Params'][e]:.2f}\n"
                )

    print("Finetuning completed.")
    for r in fold_results:
        print(f"Fold {r['fold']} best val loss = {r['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()

