"""
基础卷积算子
对标：ultralytics/nn/modules/conv.py
"""
import torch.nn as nn

def autopad(k, p=None, d=1):
    """自动计算 same padding。"""
    if d > 1:
        k = d * (k - 1) + 1

    if p is None:
        p = k // 2

    return p


class Conv(nn.Module):

    """标准卷积: Conv2d + BatchNorm2d + SiLU"""
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
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

    def forward_fuse(self, x):
        return self.act(self.conv(x))