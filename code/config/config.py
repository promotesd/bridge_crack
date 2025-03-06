import os

class Config:
    def __init__(self):
        # 数据集根目录
        self.data_root = os.path.join('..', '..', 'data') 
        # 注意：由于你在 tools 下，这里用相对路径 "../.."
        # 即 F:/大智若愚/桥梁裂缝数据集/bridge_crack/data

        # 训练超参数
        self.lr = 1e-3
        self.batch_size = 4
        self.num_workers = 0
        self.num_epochs = 20

        # K折
        self.k_folds = 5

        # 模型保存相关
        self.save_interval = 1  # 每多少个 epoch 保存一次模型
        self.num_classes = 2    # 二分类：背景 + 裂缝

        # 其他可选
        self.random_seed = 2025
        self.device = 'cuda'  # 或 'cpu'
