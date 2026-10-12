"""
模型定义
对标：ultralytics/nn/tasks.py 中的 DetectionModel
"""
import torch
import torch.nn as nn
from nn import Conv, C3k2, SPPF
        
class YOLOLikeDetector(nn.Module):
    """使用核心算子搭建模型。"""

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
        