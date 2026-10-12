"""
网络构建块
对标：ultralytics/nn/modules/block.py
"""

import torch
import torch.nn as nn

from .conv import Conv


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