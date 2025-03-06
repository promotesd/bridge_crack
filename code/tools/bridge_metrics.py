# -*- coding: utf-8 -*-
"""
@File   : bridge_metrics.py
@Desc   : 通过增量式混淆矩阵统计 mIoU、F1，并计算 FPS、FLOPs、Params 等
          有效避免一次性存储大规模预测结果导致的内存溢出 MemoryError
"""

import time
import torch
import numpy as np
from tqdm import tqdm

try:
    from thop import profile
except ImportError:
    profile = None

# sklearn只用于极少数学函数，这里不用它的 F1 实现，而是自己基于混淆矩阵做F1
# from sklearn.metrics import f1_score


__all__ = ["evaluate_metrics"]


def _update_confmat(confmat, pred, true, num_classes):
    """
    对 batch/单图进行混淆矩阵累加
    pred, true: shape=(H,W) 的 numpy array
    """
    valid_mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[valid_mask].astype(int) + pred[valid_mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    confmat += hist


def _compute_mIoU_from_confmat(confmat):
    """
    在累计完成的混淆矩阵上计算 mIoU
    confmat.shape = (num_classes, num_classes)
    """
    # IoU_c = TP_c / (TP_c + sum_{pred != c} confmat[...,c] + sum_{true != c} confmat[c,...])
    #       = diag / (row_sum + col_sum - diag)
    diag = np.diag(confmat)
    sum_rows = confmat.sum(axis=1)
    sum_cols = confmat.sum(axis=0)
    denom = (sum_rows + sum_cols - diag + 1e-10)  # 防止除0
    iou = diag / denom
    miou = np.nanmean(iou)
    return miou


def _compute_F1_macro_from_confmat(confmat):
    """
    在累计完成的混淆矩阵上计算多分类 macro-F1
    对每个类别 c:
      TP_c = confmat[c,c]
      FP_c = sum_{r != c} confmat[r,c]
      FN_c = sum_{r != c} confmat[c,r]
      F1_c = 2 * TP_c / (2*TP_c + FP_c + FN_c) = 2 P R / (P + R)
    然后对所有类别取平均
    """
    num_classes = confmat.shape[0]
    diag = np.diag(confmat)
    sum_rows = confmat.sum(axis=1)
    sum_cols = confmat.sum(axis=0)
    F1_list = []
    for c in range(num_classes):
        tp = diag[c]
        fp = sum_cols[c] - tp
        fn = sum_rows[c] - tp
        denom = (2 * tp + fp + fn + 1e-10)
        f1_c = 2 * tp / denom
        F1_list.append(f1_c)
    return float(np.mean(F1_list))


def count_flops_and_params(model, input_size=(1,3,512,512), device='cpu'):
    """
    使用 thop 来统计模型的 FLOPs 和参数量.
    """
    if not profile:
        raise ImportError("Please install thop via `pip install thop` to use FLOPs calculation.")

    dummy_input = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,))
    return flops, params


def evaluate_metrics(model,
                     dataloader,
                     device="cuda",
                     num_classes=2,
                     input_size=(1,3,512,512),
                     warmup=5,
                     max_iter=50):
    """
    评估多种指标: mIoU, F1, FPS, FLOPs, Params
    通过增量式更新混淆矩阵计算 mIoU / F1，避免一次性存所有预测导致 MemoryError

    Args:
        model: 已加载好权重的 PyTorch 模型
        dataloader: 测试/验证集 DataLoader
        device: "cuda" / "cpu"
        num_classes: 分割类别数 (含背景)
        input_size: 用于统计FLOPs的输入大小 (N, C, H, W)
        warmup: 测 FPS 时先跑几次不计时
        max_iter: 测 FPS 时最多跑多少个 batch

    Returns:
        metrics_dict: {
            "mIoU": float,
            "F1":   float,
            "FPS":  float,
            "FLOPs":float,
            "Params":float
        }
    """
    model.to(device)
    model.eval()

    # 1) 统计 FLOPs & Params (只做一次)
    flops, params = count_flops_and_params(model, input_size, device)

    # 2) warmup
    print("[Info] Starting warmup for FPS measurement ...")
    with torch.no_grad():
        w_iter = 0
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)
            w_iter += 1
            if w_iter >= warmup:
                break

    print("[Info] Warmup done. Now measuring FPS & building confmat ...")
    total_time = 0.0
    n_count = 0

    # 3) 增量式计算：混淆矩阵 shape = [num_classes, num_classes]
    confmat = np.zeros((num_classes, num_classes), dtype=np.float64)

    # 4) 遍历数据集 (只处理 max_iter 个 batch 用于FPS统计 & 评估)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if i >= max_iter:
                break

            images = images.to(device)

            # 计时开始
            t1 = time.time()
            logits = model(images)
            # 如果是CUDA，需要同步保证计时准确
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.time()

            total_time += (t2 - t1)
            n_count += images.size(0)

            # 如果模型在 train 模式下会返回多个输出，取第一个
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            # preds -> (N,H,W)
            preds_argmax = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()

            # 累加混淆矩阵
            for b_idx in range(preds_argmax.shape[0]):
                pred_2d = preds_argmax[b_idx]
                true_2d = labels_np[b_idx]
                _update_confmat(confmat, pred_2d, true_2d, num_classes)

    # 5) 计算FPS
    fps = n_count / (total_time + 1e-10)

    # 6) 根据最终混淆矩阵 compute mIoU & F1
    miou_val = _compute_mIoU_from_confmat(confmat)
    f1_val = _compute_F1_macro_from_confmat(confmat)

    metrics_dict = {
        "mIoU":   float(miou_val),
        "F1":     float(f1_val),
        "FPS":    float(fps),
        "FLOPs":  float(flops),
        "Params": float(params),
    }
    return metrics_dict


if __name__ == "__main__":
    """
    下面是一个简单的演示
    """
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    class DummySegModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, 1, 1)
            self.out = nn.Conv2d(16, num_classes, 1, 1, 0)
        def forward(self, x):
            x = nn.functional.relu(self.conv(x))
            x = self.out(x)
            return x

    # 构建一个简单模型
    model = DummySegModel(num_classes=2)

    # 构造一个 dummy 的 dataloader
    images = torch.randn(10, 3, 512, 512)
    labels = torch.randint(0, 2, (10, 512, 512))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # 调用评估函数
    results = evaluate_metrics(
        model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        num_classes=2,
        input_size=(1,3,512,512),  # FLOPs输入大小
        warmup=2,
        max_iter=5
    )
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
