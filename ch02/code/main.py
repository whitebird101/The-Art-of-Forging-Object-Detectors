import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MultiscaleFakeDataset(Dataset):
    def __init__(self, num_samples=100,
                 img_size=256, num_classes=2):
        super().__init__()
        self.images = []
        self.targets = []
        channels = num_classes + 8
        grid_sizes = [
            img_size // 8,   # 32 (P3, stride=8)
            img_size // 16,  # 16 (P4, stride=16)
            img_size // 32,  #  8 (P5, stride=32)
        ]
        for _ in range(num_samples):
            img = torch.randn(3, img_size, img_size)
            target_list = [torch.randn(channels, gs, gs) 
            for gs in grid_sizes]
            self.images.append(img)
            self.targets.append(target_list)
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
        
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [
        torch.stack([item[1][scale] for item in batch])
        for scale in range(3)
    ]
    return images, targets
    
class MultiscaleDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.out_channels = num_classes + 8

        # Backbone: 5层下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU()
        )  # → P3: [B, 32, 32, 32]
        self.down4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )  # → P4: [B, 64, 16, 16]
        self.down5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )  # → P5: [B, 128, 8, 8]
        # 1×1 卷积：将P5的128通道降到64，对齐P4
        self.lateral5 = nn.Conv2d(128, 64, kernel_size=1)
        # 3×3 卷积：融合上采样的P5和原始P4
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        # 1×1 卷积：将F4的64通道降到32，对齐P3
        self.lateral4 = nn.Conv2d(64, 32, kernel_size=1)
        # 3×3 卷积：融合上采样的F4和原始P3
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        # 三个独立的 1×1 检测头
        self.head3 = nn.Conv2d(
            32, self.out_channels, kernel_size=1
        )
        self.head4 = nn.Conv2d(
            64, self.out_channels, kernel_size=1
        )
        self.head5 = nn.Conv2d(
            128, self.out_channels, kernel_size=1
        )
        # 上采样器
        self.upsample = nn.Upsample(
            scale_factor=2, mode='nearest'
        )
        
    def forward(self, x):
        # Backbone
        x = self.down1(x)    # [B, 16, 128, 128]
        x = self.down2(x)    # [B, 32,  64,  64]
        p3 = self.down3(x)   # [B, 32,  32,  32]
        p4 = self.down4(p3)  # [B, 64,  16,  16]
        p5 = self.down5(p4)  # [B,128,   8,   8]
        # FPN: P5 → F4
        up5 = self.upsample(self.lateral5(p5))
        # lateral5: [B,128,8,8] → [B,64,8,8]
        # upsample: [B,64,8,8] → [B,64,16,16]
        f4 = self.fpn_conv4(
            torch.cat([up5, p4], 1)
        )
        # cat: [B,64,16,16]+[B,64,16,16]→[B,128,16,16]
        # fpn_conv4: [B,128,16,16] → [B,64,16,16]
        # FPN: F4 → F3
        up4 = self.upsample(self.lateral4(f4))
        # lateral4: [B,64,16,16] → [B,32,16,16]
        # upsample: [B,32,16,16] → [B,32,32,32]
        f3 = self.fpn_conv3(
            torch.cat([up4, p3], 1)
        )
        # cat: [B,32,32,32]+[B,32,32,32]→[B,64,32,32]
        # fpn_conv3: [B,64,32,32] → [B,32,32,32]
        # 三个检测头
        out3 = self.head3(f3)  # [B, 10, 32, 32]
        out4 = self.head4(f4)  # [B, 10, 16, 16]
        out5 = self.head5(p5)  # [B, 10,  8,  8]
        return [out3, out4, out5]
        
        
        
def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in val_dataloader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            outputs = model(images)
            loss = sum(
                criterion(out, tgt)
                for out, tgt in zip(outputs, targets)
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    mock_map = 1.0 / (avg_loss + 1e-6)
    return avg_loss, mock_map
    
    
    
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")

    img_size = 256
    num_classes = 2
    batch_size = 4
    num_epochs = 20
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    # 数据准备（使用自定义 collate_fn 处理多尺度标签）
    train_dataset = MultiscaleFakeDataset(num_samples=80, img_size=img_size, num_classes=num_classes)
    val_dataset = MultiscaleFakeDataset(num_samples=20, img_size=img_size, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiscaleDetector(num_classes=num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_mock_map = 0.0

    print("-" * 60)
    # 打印模型结构概览
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 张量维度验证
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    outs = model(dummy)
    print("各尺度输出形状:")
    for i, o in enumerate(outs):
        stride = img_size // o.shape[-1]
        print(f"  P{i+3}: {list(o.shape)}  (stride={stride})")
    print("-" * 60)
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            outputs = model(images)

            loss = sum(
                criterion(out, tgt)
                for out, tgt in zip(outputs, targets)
            )
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
        train_avg_loss = train_total_loss / len(train_loader)
        val_avg_loss, val_mock_map = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"| Train Loss: {train_avg_loss:.4f} "
              f"| Val Loss: {val_avg_loss:.4f} "
              f"| Val mAP(mock): {val_mock_map:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))

        if val_mock_map > best_mock_map:
            best_mock_map = val_mock_map
            print(f"  --> 更优模型 (mAP: {best_mock_map:.4f})，已保存至 best.pt")
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            
            
if __name__ == "__main__":
    train_model()
