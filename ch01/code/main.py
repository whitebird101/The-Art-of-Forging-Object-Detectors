"""
第1章：打造第一根"烧火棍" —— 单文件极简卷积网络

目标：用一个文件跑通深度学习的完整闭环
  虚拟数据 → 数据加载 → 极简CNN模型 → MSE Loss → 训练循环 → 验证 → 保存权重

本章的模型极其简陋（3层Conv + 1x1 Head），Loss 是粗暴的 MSE，
数据是随机噪声，但它展示了深度卷积网络的 **五大要素**：
  1. 数据集 (Dataset)
  2. 模型 (Model)
  3. 损失函数 (Loss / Criterion)
  4. 优化器 (Optimizer)
  5. 训练循环 (Training Loop)

后续每一章都会在此基础上逐步替换、丰富这五大要素，
最终演化为与 Ultralytics YOLO v11 OBB 完全一致的工业级代码。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. 数据集定义 —— 虚拟数据（后续第6/12章将替换为真实 DOTA 数据集）
# ==============================================================================
class SimpleFakeDataset(Dataset):
    """
    生成随机张量模拟目标检测数据集。

    关键概念：
    - 输入图像: [B, 3, H, W]      — B张3通道彩色图像
    - 网格标签: [B, C, H/8, W/8]  — 模型输出的特征图尺寸是输入的1/8
      其中 C = num_classes + 8，这里的8模拟了边界框属性通道
      （真实YOLO中是 4*reg_max + nc + angle 等）

    为什么标签也是"网格"形状？
    → 因为检测模型的输出就是一张特征图，每个网格点对应原图的一块区域，
      预测该区域内有无物体及其属性。训练时，标签要和输出形状对齐才能算Loss。
    """

    def __init__(self, num_samples=100, img_size=256, num_classes=2, num_targets=3):
        super().__init__()
        self.images = []
        self.targets = []

        # 下采样倍率: 3层 stride=2 卷积 → 2^3 = 8
        grid_size = img_size // 8
        # 输出通道 = 类别数 + 边界框属性数(模拟8个)
        channels = num_classes + 8

        for _ in range(num_samples):
            single_img = torch.randn(3, img_size, img_size)
            single_target = torch.randn(channels, grid_size, grid_size)
            self.images.append(single_img)
            self.targets.append(single_target)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


# ==============================================================================
# 2. 模型定义 —— 3层卷积+1x1检测头（后续第3章加BN/SiLU, 第5-6章用YAML动态构建）
# ==============================================================================
class SimpleDetector(nn.Module):
    """
    极简检测模型：
      Backbone: 3层 Conv2d(stride=2) + ReLU → 特征图缩小8倍
      Head:     1x1 Conv2d → 改变通道数为输出维度

    张量流动：
      输入 [B, 3, 256, 256]
        → Conv1 [B, 16, 128, 128]
        → Conv2 [B, 32,  64,  64]
        → Conv3 [B, 64,  32,  32]    ← Backbone 输出
        → Head  [B, 10,  32,  32]    ← 10 = 2类 + 8属性
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.out_channels = num_classes + 8

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # 256→128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 128→64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64→32
            nn.ReLU(),
        )

        self.head = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ==============================================================================
# 3. 验证逻辑（后续第15章将替换为完整的 mAP Validator）
# ==============================================================================
def evaluate(model, val_dataloader, criterion, device):
    """在验证集上计算平均Loss，模拟mAP指标。"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    # 模拟mAP: loss越小 → 指标越高（真实mAP需要IoU匹配+PR曲线积分）
    mock_map = 1.0 / (avg_loss + 1e-6)
    return avg_loss, mock_map


# ==============================================================================
# 4. 训练主循环（后续第4章抽离为独立 trainer.py）
# ==============================================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")

    # --- 基础配置 ---
    img_size = 256
    num_classes = 2
    batch_size = 4
    num_epochs = 20
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    # --- 数据准备 ---
    train_dataset = SimpleFakeDataset(num_samples=80, img_size=img_size, num_classes=num_classes)
    val_dataset = SimpleFakeDataset(num_samples=20, img_size=img_size, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 模型 & 优化器 ---
    model = SimpleDetector(num_classes=num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_mock_map = 0.0

    print("-" * 60)

    # --- 训练循环 ---
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()          # 清除旧梯度
            predictions = model(images)    # 前向传播
            loss = criterion(predictions, targets)  # 计算损失
            loss.backward()                # 反向传播
            optimizer.step()               # 更新权重

            train_total_loss += loss.item()

        train_avg_loss = train_total_loss / len(train_loader)

        # 验证
        val_avg_loss, val_mock_map = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"| Train Loss: {train_avg_loss:.4f} "
              f"| Val Loss: {val_avg_loss:.4f} "
              f"| Val mAP(mock): {val_mock_map:.4f}")

        # 保存 last
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))

        # 保存 best
        if val_mock_map > best_mock_map:
            best_mock_map = val_mock_map
            print(f"  --> 发现更优模型 (mAP: {best_mock_map:.4f})，已保存至 best.pt")
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))


if __name__ == "__main__":
    train_model()
