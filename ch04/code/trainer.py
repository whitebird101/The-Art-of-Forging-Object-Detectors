"""
训练器
对标：ultralytics/engine/trainer.py 中的 BaseTrainer
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SimpleTrainer:
    """管理模型训练、验证和权重保存。"""

    def __init__(self, model, train_loader, val_loader, device, lr=1e-3, save_dir="runs/weights"):
        self.model = model.to(device)                  # 模型
        self.train_loader = train_loader               # 训练集
        self.val_loader = val_loader                   # 验证集
        self.device = device                           # 计算设备
        self.save_dir = save_dir                       # 权重保存目录

        os.makedirs(save_dir, exist_ok=True)           # 创建权重目录

        self.criterion = nn.MSELoss()                  # 损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # 优化器

        self.best_val_loss = float("inf")              # 当前最优验证 Loss
        self.epoch = 0                                 # 当前 epoch
        self.num_epochs = 0                            # 总 epoch 数

    def check_model_shapes(self, img_size=256):
        """训练前检查三个尺度的输出尺寸。"""

        dummy = torch.randn(1, 3, img_size, img_size, device=self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(dummy)

        print("各尺度输出形状:")

        for i, out in enumerate(outputs, start=3):
            stride = img_size // out.shape[-1]
            print(f"  P{i}: {list(out.shape)}  (stride={stride})")

        print("-" * 60)

    def train_one_epoch(self):
        """训练一个 epoch。"""

        self.model.train()
        train_total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.num_epochs}",
            leave=False
        )

        for images, targets in pbar:
            images = images.to(self.device)
            targets = [t.to(self.device) for t in targets]

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = sum(self.criterion(out, target) for out, target in zip(outputs, targets))

            loss.backward()
            self.optimizer.step()

            train_total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return train_total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        """验证模型。"""

        self.model.eval()
        total_loss = 0.0

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = [t.to(self.device) for t in targets]

            outputs = self.model(images)
            loss = sum(self.criterion(out, target) for out, target in zip(outputs, targets))

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_model(self, filename):
        """保存模型权重。"""

        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs=20, img_size=256):
        """完整训练流程。"""

        self.num_epochs = num_epochs

        # 训练前先检查输出尺寸
        self.check_model_shapes(img_size)

        # 正式训练
        for epoch in range(num_epochs):
            self.epoch = epoch

            train_avg_loss = self.train_one_epoch()
            val_avg_loss = self.validate()

            print(
                f"Epoch [{epoch + 1:2d}/{num_epochs}] "
                f"| Train Loss: {train_avg_loss:.4f} "
                f"| Val Loss: {val_avg_loss:.4f}"
            )

            self.save_model("last.pt")

            if val_avg_loss < self.best_val_loss:
                self.best_val_loss = val_avg_loss
                self.save_model("best.pt")

                print(f"  --> 保存当前最佳模型，Val Loss = {self.best_val_loss:.4f}")
