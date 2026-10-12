"""
数据集定义 (dataset.py)
对标: ultralytics/data/dataset.py 中的 YOLODataset

职责：

  - 数据集的创建与管理
  - 标签的加载与预处理
  - collate_fn 多目标打包
  - 目前仍使用虚拟数据（后续将替换为真实 DOTA 数据集）。
"""

import torch
from torch.utils.data import Dataset, DataLoader

class MultiscaleFakeDataset(Dataset):
    def __init__(self, num_samples=100, img_size=256, num_classes=2):
        super().__init__()
        self.images = []
        self.targets = []

        channels = num_classes + 8
        grid_sizes = [img_size // 8, img_size // 16, img_size // 32]

        for _ in range(num_samples):
            image = torch.randn(3, img_size, img_size)
            target_list = [torch.randn(channels, gs, gs) for gs in grid_sizes]

            self.images.append(image)
            self.targets.append(target_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
        
        
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [torch.stack([item[1][s] for item in batch]) for s in range(3)]
    return images, targets
    
def build_dataloader(dataset, batch_size, shuffle=True):

    """
    DataLoader 的工厂函数。
    对标: ultralytics/data/build.py::build_dataloader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False,
    )