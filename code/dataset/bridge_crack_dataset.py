import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import KFold

class BridgeCrackDataset(Dataset):
    """
    通用数据集类，可加载“图片-标签”对，包含数据增强、预处理等。
    默认是做二分类分割：0=背景，1=裂缝。
    """

    def __init__(self, image_paths, label_paths, transform=None):
        """
        Args:
            image_paths (list): 图像文件路径列表
            label_paths (list): 标签文件路径列表 (与image_paths对应)
            transform (callable, optional): 数据增强transform
        """
        super().__init__()
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        # 读取图像
        image = Image.open(img_path).convert('RGB')
        # 读取标签，假设是单通道的PNG图
        label = Image.open(lbl_path)

        # 如果需要 PIL -> Tensor，再进行数据增强
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label

def collect_image_label_paths(root_dir):
    """
    收集所有图像和标签的绝对路径或相对路径
    root_dir: 数据集根目录，如: F:/大智若愚/桥梁裂缝数据集/bridge_crack/data
    假设结构：
        ├── Noise images
        │    ├── Image
        │    └── Label
        ├── Non-steel crack images
        │    ├── Image
        │    └── Label
        ├── Steel crack images
        │    ├── Image
        │    └── Label
        └── Synthesized crack image dataset
             ├── Synthesized crack images
             └── Synthesized crack labels
    返回：
        noise_img_paths, noise_lbl_paths
        non_steel_img_paths, non_steel_lbl_paths
        steel_img_paths, steel_lbl_paths
        synth_img_paths, synth_lbl_paths
    """
    # 根据实际数据文件命名进行匹配
    noise_root = os.path.join(root_dir, 'Noise images')
    noise_img_paths = sorted(glob.glob(os.path.join(noise_root, 'Image', '*.jpg')))
    noise_lbl_paths = sorted(glob.glob(os.path.join(noise_root, 'Label', '*.png')))

    non_steel_root = os.path.join(root_dir, 'Non-steel crack images')
    non_steel_img_paths = sorted(glob.glob(os.path.join(non_steel_root, 'Image', '*.jpg')))
    non_steel_lbl_paths = sorted(glob.glob(os.path.join(non_steel_root, 'Label', '*.png')))

    steel_root = os.path.join(root_dir, 'Steel crack images')
    steel_img_paths = sorted(glob.glob(os.path.join(steel_root, 'Image', '*.jpg')))
    steel_lbl_paths = sorted(glob.glob(os.path.join(steel_root, 'Label', '*.png')))

    synth_root = os.path.join(root_dir, 'Synthesized crack image dataset')
    synth_img_paths = sorted(glob.glob(os.path.join(synth_root, 'Synthesized crack images', '*.jpg')))
    synth_lbl_paths = sorted(glob.glob(os.path.join(synth_root, 'Synthesized crack labels', '*.png')))

    return (noise_img_paths, noise_lbl_paths,
            non_steel_img_paths, non_steel_lbl_paths,
            steel_img_paths, steel_lbl_paths,
            synth_img_paths, synth_lbl_paths)

class RandomScaleAndCrop:
    """
    多尺度+随机裁剪 + 随机水平翻转等数据增强示例
    scale_range: (min_scale, max_scale)
    crop_size: (h, w)
    """

    def __init__(self, scale_range=(0.75, 1.25), crop_size=(256, 256), hflip_prob=0.5):
        self.scale_range = scale_range
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob

    def __call__(self, image, label):
        # 随机水平翻转
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 随机缩放
        scale = random.uniform(*self.scale_range)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)

        # 随机裁剪
        # 如果缩放后图像不足crop_size，可以先padding
        pad_h = max(self.crop_size[0] - new_h, 0)
        pad_w = max(self.crop_size[1] - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            # 填充为0
            image = T.functional.pad(image, (0,0,pad_w,pad_h), fill=0)
            label = T.functional.pad(label, (0,0,pad_w,pad_h), fill=0)

        # 再随机裁剪
        w, h = image.width, image.height
        x1 = random.randint(0, w - self.crop_size[1])
        y1 = random.randint(0, h - self.crop_size[0])

        image = image.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))
        label = label.crop((x1, y1, x1 + self.crop_size[1], y1 + self.crop_size[0]))

        # 转为Tensor
        image = T.functional.to_tensor(image)
        label = torch.from_numpy(np.array(label, dtype=np.int64))

        return image, label

def create_kfold_datasets(all_img_paths, all_lbl_paths, n_splits=5, shuffle=True, random_seed=2025, transform=None):
    """
    使用KFold将 (all_img_paths, all_lbl_paths) 拆分成 k折 (train_idx, val_idx) 集合.
    返回 list，包含 k 个 (train_dataset, val_dataset) 元组
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    datasets = []

    for train_idx, val_idx in kf.split(all_img_paths):
        train_img_paths = [all_img_paths[i] for i in train_idx]
        train_lbl_paths = [all_lbl_paths[i] for i in train_idx]
        val_img_paths = [all_img_paths[i] for i in val_idx]
        val_lbl_paths = [all_lbl_paths[i] for i in val_idx]

        train_dataset = BridgeCrackDataset(train_img_paths, train_lbl_paths, transform=transform)
        val_dataset = BridgeCrackDataset(val_img_paths, val_lbl_paths, transform=transform)
        datasets.append((train_dataset, val_dataset))
    return datasets
