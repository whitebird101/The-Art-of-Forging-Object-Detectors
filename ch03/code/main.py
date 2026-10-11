import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def gradient_experiment():
    """30 层 Conv2d + ReLU 梯度变化实验"""
    print("=" * 60)
    print("30 层 Conv2d + ReLU 梯度变化实验")
    print("=" * 60)

    # ---- 1. 搭建 30 层网络 ----

    layers_basic = []
    for i in range(30):
        c_in = 3 if i == 0 else 32
        layers_basic.append(nn.Conv2d(c_in, 32, 3, padding=1))
        layers_basic.append(nn.ReLU())

    net_basic = nn.Sequential(*layers_basic)

    # ---- 2. 构造输入 ----
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    # ---- 3. 前向传播 + 反向传播 ----
    out = net_basic(x)
    loss = out.mean()
    loss.backward()

    # ---- 4. 查看输入梯度 ----
    input_grad = x.grad.abs().mean().item()
    print(f"输入图像的梯度均值: {input_grad:.6e}")
    print("-" * 60)

    # ---- 5. 查看每一层卷积权重的梯度 ----
    conv_layer_id = 0

    for layer in layers_basic:
        if isinstance(layer, nn.Conv2d):
            conv_layer_id += 1
            grad_mean = layer.weight.grad.abs().mean().item()
            print(
                f"第 {conv_layer_id:02d} 层卷积权重梯度均值: "
                f"{grad_mean:.16f}"
            )

    print("=" * 60)
    
    
def autopad(k, p=None, d=1):
    """
    自动计算padding使得卷积输出尺寸 = 输入尺寸 / stride。
    对标: ultralytics/nn/modules/conv.py::autopad
    """
    if d > 1:
        k = d * (k - 1) + 1  # 膨胀卷积的等效核大小
    if p is None:
        p = k // 2            # 'same' padding
    return p
    
    
class Conv(nn.Module):
    """
    标准卷积三件套: Conv2d + BatchNorm2d + SiLU
    对标: ultralytics/nn/modules/conv.py::Conv
    """

    default_act = nn.SiLU()  # YOLO 的默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None,
                 g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s,
            autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act if act is True
            else act if isinstance(act, nn.Module)
            else nn.Identity()
        )
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        
def gradient_experiment_yolo_conv():
    """30 层 YOLO 风格 Conv 梯度变化实验"""
    print("=" * 60)
    print("30 层 YOLO 风格 Conv 梯度变化实验")
    print("=" * 60)

    # ---- 1. 搭建 30 层网络 ----

    layers_yolo = []
    for i in range(30):
        c_in = 3 if i == 0 else 32
        layers_yolo.append(Conv(c_in, 32, k=3, s=1))

    net_yolo = nn.Sequential(*layers_yolo)

    # ---- 2. 使用与第一次实验相同规则生成输入 ----
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    # ---- 3. 前向传播 + 反向传播 ----
    out = net_yolo(x)
    loss = out.mean()
    loss.backward()

    # ---- 4. 查看输入梯度 ----
    input_grad = x.grad.abs().mean().item()
    print(f"输入图像的梯度均值: {input_grad:.6e}")
    print("-" * 60)

    # ---- 5. 查看每个 Conv 模块内部卷积权重的梯度 ----
    for i, layer in enumerate(layers_yolo, start=1):
        grad_mean = layer.conv.weight.grad.abs().mean().item()
        print(
            f"第 {i:02d} 层卷积权重梯度均值: "
            f"{grad_mean:.16f}"
        )

    print("=" * 60)
    
    
class Bottleneck(nn.Module):
    """
    YOLO 风格 Bottleneck：
    两个 Conv + 可选残差连接

    参考：
    ultralytics/nn/modules/block.py::Bottleneck
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        # 中间通道数
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        # 只有输入输出形状一致时，才能直接做 x + F(x)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))

        if self.add:
            return x + y

        return y
        
class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        # 定义网络中需要用到的各个模块，包括分流前调整通道的cv1和最后融合数据的cv2，
        # 以及包含n个Bottleneck的复杂处理结构
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))

        for m in self.m:
            y.append(m(y[-1]))

        return self.cv2(torch.cat(y, dim=1))
        
        
class C3k2(C2f):
    """C3k2：当前使用 Bottleneck 作为内部特征提取单元。"""

    def __init__(self, c1, c2, n=1, e=0.5, g=1, shortcut=True):
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e)
        
        
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5):
        super().__init__()

        # 隐藏通道：先压缩到输入通道数的一半
        c_ = c1 // 2

        # 调整通道数
        self.cv1 = Conv(c1, c_, 1, 1, act=False)

        # 四路特征拼接后，再进行融合
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        # 5×5 最大池化，stride=1，因此不改变特征图尺寸
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x0 = self.cv1(x)

        y1 = self.m(x0)
        y2 = self.m(y1)
        y3 = self.m(y2)

        y = torch.cat([x0, y1, y2, y3], dim=1)

        return self.cv2(y)
        
class YOLOLikeDetector(nn.Module):
    """使用第三卷核心算子搭建的教学版多尺度检测器。"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.out_channels = num_classes + 8

        # ---------------- Backbone ----------------
        self.stem = Conv(3, 16, 3, 2)             # 256 -> 128
        self.down1 = Conv(16, 32, 3, 2)           # 128 -> 64
        self.stage1 = C3k2(32, 32, n=1)

        self.down2 = Conv(32, 64, 3, 2)           # 64 -> 32
        self.stage2 = C3k2(64, 64, n=1)           # P3, stride=8

        self.down3 = Conv(64, 128, 3, 2)          # 32 -> 16
        self.stage3 = C3k2(128, 128, n=1)         # P4, stride=16

        self.down4 = Conv(128, 128, 3, 2)         # 16 -> 8
        self.stage4 = C3k2(128, 128, n=1)
        self.sppf = SPPF(128, 128)                # P5, stride=32

        # ---------------- Neck：沿用第2卷 FPN ----------------
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral5 = Conv(128, 64, 1, 1)
        self.fpn4 = C3k2(192, 64, n=1)            # 64 + 128 = 192

        self.lateral4 = Conv(64, 32, 1, 1)
        self.fpn3 = C3k2(96, 32, n=1)             # 32 + 64 = 96

        # ---------------- 临时预测头 ----------------
        self.head3 = nn.Conv2d(32, self.out_channels, 1)
        self.head4 = nn.Conv2d(64, self.out_channels, 1)
        self.head5 = nn.Conv2d(128, self.out_channels, 1)

    def forward(self, x):
        # Backbone
        x = self.stem(x)                          # [B, 16, 128, 128]
        x = self.down1(x)                         # [B, 32, 64, 64]
        x = self.stage1(x)                        # [B, 32, 64, 64]

        x = self.down2(x)                         # [B, 64, 32, 32]
        p3 = self.stage2(x)                       # [B, 64, 32, 32]

        x = self.down3(p3)                        # [B, 128, 16, 16]
        p4 = self.stage3(x)                       # [B, 128, 16, 16]

        x = self.down4(p4)                        # [B, 128, 8, 8]
        x = self.stage4(x)                        # [B, 128, 8, 8]
        p5 = self.sppf(x)                         # [B, 128, 8, 8]

        # FPN：P5 -> P4
        up5 = self.up(self.lateral5(p5))          # [B, 64, 16, 16]
        f4 = torch.cat([up5, p4], dim=1)          # [B, 192, 16, 16]
        f4 = self.fpn4(f4)                        # [B, 64, 16, 16]

        # FPN：F4 -> P3
        up4 = self.up(self.lateral4(f4))          # [B, 32, 32, 32]
        f3 = torch.cat([up4, p3], dim=1)          # [B, 96, 32, 32]
        f3 = self.fpn3(f3)                        # [B, 32, 32, 32]

        # 三尺度输出
        return [
            self.head3(f3),                       # [B, C, 32, 32]
            self.head4(f4),                       # [B, C, 16, 16]
            self.head5(p5),                       # [B, C, 8, 8]
        ]
        
        
def check_model_shapes():
    model = YOLOLikeDetector(num_classes=2)
    model.eval()

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        outputs = model(x)

    print("各尺度输出形状：")
    for i, out in enumerate(outputs, start=3):
        stride = 256 // out.shape[-1]
        print(f"P{i}: {list(out.shape)}, stride={stride}")
        
        
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
    
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(images)
        loss = sum(criterion(out, target) for out, target in zip(outputs, targets))
        total_loss += loss.item()

    return total_loss / len(loader)
    
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")

    img_size = 256
    num_classes = 2
    batch_size = 4
    num_epochs = 20
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = MultiscaleFakeDataset(80, img_size, num_classes)
    val_dataset = MultiscaleFakeDataset(20, img_size, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = YOLOLikeDetector(num_classes=num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    # ---------- 训练前先做一次尺寸验算 ----------
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy)

    print("各尺度输出形状:")
    for i, out in enumerate(outputs, start=3):
        stride = img_size // out.shape[-1]
        print(f"  P{i}: {list(out.shape)}  (stride={stride})")

    print("-" * 60)

    # ---------- 正式进入训练循环 ----------
    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, targets in pbar:
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()

            outputs = model(images)
            loss = sum(criterion(out, target) for out, target in zip(outputs, targets))

            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_avg_loss = train_total_loss / len(train_loader)
        val_avg_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1:2d}/{num_epochs}] "
            f"| Train Loss: {train_avg_loss:.4f} "
            f"| Val Loss: {val_avg_loss:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
            print(f"  --> 保存当前最佳模型，Val Loss = {best_val_loss:.4f}")
        

if __name__ == "__main__":
    train_model()
