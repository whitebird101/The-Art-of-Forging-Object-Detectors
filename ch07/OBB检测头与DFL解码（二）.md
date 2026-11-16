# 重塑视界 ------ OBB检测头与DFL解码（二）


上一节，我们已经知道，**一个锚点会输出** `nc + 4 × reg_max + 1` 个数值。这里面每个类别的得分可以通过 `sigmoid` 转换成概率直接用。至于表示目标框的这些离散数值，怎么把它变成我们需要的偏移量进而得到最终标注框呢？

首先就是要把 `4 × reg_max` 个离散的量，通过计算“数学期望”的方式，转换成 `left`、`top`、`right`、`bottom` 4个相对偏移量，然后再构建相应的函数，把目标框的精确位置计算出来。

在写具体的代码之前之前，先要把上一卷的成果搬过来：**把 `ch06/` 目录下的代码全部复制到 `ch07/`**（如果你还没做这一步的话）。

## 7.3 DFL解码：从概率分布到偏移量

### 7.3.1数学期望与 `1×1` 卷积

前面已经详细介绍过，DFL 需要将 16 个候选值对应的概率转换成一个确定的偏移量，也就是计算他们的数学期望。

数学期望的计算公式为：

$  
d = 0 \cdot p_0 + 1 \cdot p_1 + 2 \cdot p_2 + \cdots + 15 \cdot p_{15}  
$，

其中，$p_i$ 表示第 $i$ 个候选值对应的概率。

这个公式本质上是一次**加权求和**：每个候选值乘以自己的概率，再把所有乘积相加。

这个计算方式是不是有点眼熟？回想一下前面介绍过的卷积运算：卷积核中的权重与输入数据逐一相乘，再将乘积相加，便得到了输出值。

**数学期望的计算，与卷积运算的核心计算形式非常相似。** 既然如此，我们自然想到，能不能利用卷积来计算数学期望呢？答案自然是可以的。

只需要构造一个特殊的卷积核，将它的权重固定为：`[0, 1, 2, 3, ..., 15]`，然后把 16 个概率作为输入。这样，卷积计算得到的就是：$  0 \cdot p_0 + 1 \cdot p_1 + 2 \cdot p_2 + \cdots + 15 \cdot p_{15}  $，这正是我们需要的数学期望。

在代码中，我们使用一个输入通道数为 16、输出通道数为 1、卷积核大小为 `1×1` 的卷积层来实现这个计算：

```python
self.conv = nn.Conv2d(16, 1, kernel_size=1, bias=False)
```

这里的 `1×1` 卷积不负责提取图像的空间特征，而是借助卷积运算的形式，完成数学期望的计算。

爱琢磨的读者可能会有一个疑问：一个锚点需要计算左、上、右、下四个方向的距离，难道需要调用四次卷积吗？

其实只需要**将数据整理成合适的形状**，一次卷积就能**同时计算所有锚点的四个方向**。这也是使用卷积完成数学期望计算的好处。

以 `reg_max=16` 为例，每个锚点有四组概率分布，每组包含 16 个概率。经过前面的形状变换后，输入卷积的数据形状为 `(B, 16, 4, A)`，这里可以这样理解：

*   `B`：一次处理多少张图像，也就是有多少个特征图；
*   `16`：每个方向的候选值数量，也是输入通道数；
*   `4`：四个方向，依次对应左、上、右、下；
*   `A`：需要处理的锚点数量。

对于每个锚点，四个方向分别拥有一组独立的 16 个概率。`1×1` 卷积会在每个方向、每个锚点的位置上，独立执行相同的加权求和。虽然四个方向共用同一个卷积核，但它们的输入概率各不相同，因此会得到各自的数学期望。卷积不会把左、上、右、下四个方向混合起来。

经过一次卷积，输出形状变为：`(B, 1, 4, A)`，原来的 16 个通道被压缩为 1 个通道，而四个方向和所有锚点的位置仍然保留。最后再调整张量形状，就能得到：`(B, 4, A)`，也就是每个锚点对应的四个距离。需要注意的是：**DFL模块输出的距离仍然以特征图格数为单位**，还不是原始图像上的像素距离。后面需要乘以对应的
`stride` 完成尺度转换。

至此，DFL模块的实现思路就清楚了：先将预测分数转换为概率，再通过固定权重的 `1×1` 卷积计算数学期望，一次性得到所有锚点的四个方向距离。

### 7.3.2 DFL 模块的实现

打开 `ch07/nn/block.py`，如果文件顶部还没有导入 `torch` 和 `torch.nn`，需要补上相应的导入语句。

在文件中添加以下代码：

```python
class DFL(nn.Module):
    """
    DFL 解码模块：将四组离散概率分布转换为四个距离。
    对应 Ultralytics 中的 DFL 实现思路。
    """

    def __init__(self, c1=16):
        super().__init__()

        self.c1 = c1  # 每个方向的候选值数量，默认是 16

        # 1×1 卷积模块：输入 c1 个概率通道，输出一个加权求和值
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)

        # 卷积核不参与训练，它只负责计算数学期望
        self.conv.weight.requires_grad_(False)

        # 将卷积权重固定为候选值编号 [0, 1, ..., 15]
        bins = torch.arange(c1, dtype=torch.float32).view(1, c1, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(bins)

    def forward(self, x):
        # x 的形状为 (B, 4*c1, A)
        b, _, a = x.shape  # b：批次大小；a：锚点数量

        # 将 4×16 个通道拆分为 4 个方向、每个方向 16 个候选值
        x = x.view(b, 4, self.c1, a)

        # 将候选值维度移动到通道位置，形状变为 (B, 16, 4, A)
        x = x.transpose(2, 1)

        # 将每个方向的原始分数转换成概率
        x = x.softmax(dim=1)

        # 用固定权重 [0, 1, ..., 15] 计算数学期望
        x = self.conv(x)

        # 恢复为 (B, 4, A)，得到四个方向的距离
        return x.view(b, 4, a)
```

在 `ch07/nn/__init__.py` 中，将 `DFL` 加入现有的模块导出语句，确保其他文件可以正常导入它。

简单解释一下代码中的一些操作：

###### 1. 卷积权重初始化与固定？

数学期望的计算规则已经确定，权重就是候选值编号 `0, 1, ..., 15`，而不是让网络通过训练自行学习。因此，代码先关闭权重梯度，再用 `copy_()` 写入固定值。`torch.no_grad()` 表示这次赋值不需要记录梯度。

因为 `c1=16` ， 代码中的 `torch.arange(c1)` 函数会生成张量：`[0, 1, 2, 3, ..., 15]`，`view(1, c1, 1, 1)` 则将其调整为卷积权重需要的形状 `(1, 16, 1, 1)`。

###### 2. 为什么需要 `view()` 和 `transpose()`？

这是整个 `forward()` 中最需要理解的部分。

假设输入张量的形状为：`(B, 64, A)`，其中，`B` 是批次大小，`A` 是锚点数量，而 `64 = 4 × 16`，代表四个方向各自的 16 个预测分数。我们需要将它整理成适合计算的形状：

| 操作                  | 张量形状            | 作用                 |
| ------------------- | --------------- | ------------------ |
| 输入                  | `(B, 64, A)`    | 四个方向共 64 个预测通道     |
| `view(b, 4, 16, a)` | `(B, 4, 16, A)` | 拆分出四个方向及其候选值       |
| `transpose(2, 1)`   | `(B, 16, 4, A)` | 将 16 个候选值移动到通道维    |
| `softmax(dim=1)`    | `(B, 16, 4, A)` | 对每个方向的 16 个分数进行归一化 |
| `self.conv(x)`      | `(B, 1, 4, A)`  | 对概率分布计算加权求和        |
| `view(b, 4, a)`     | `(B, 4, A)`     | 得到四个方向的距离          |

这里重点理解两个操作。

**第一，为什么要 `transpose(2, 1)`？** `view()` 之后，张量的第二维是四个方向，第三维是每个方向的 16 个候选值。我们需要让候选值维度变成通道维，这样才能使用 `softmax(dim=1)`，并交给 `Conv2d` 计算。

因此，交换这两个维度后，张量从 `(B, 4, 16, A)` 变成 `(B, 16, 4, A)`。

**第二，为什么 `Conv2d` 能完成数学期望计算？** PyTorch 的 `Conv2d` 接收的张量形状是 `(B, C, H, W)`。在这里，`C=16` 是候选值通道，`H=4` 对应四个方向，`W=A` 对应锚点索引。

注意，这里的 `H=4` 只是为了组织数据，并不是图像的实际高度。

由于卷积核的空间尺寸是 `1×1`，它不会混合相邻位置的数据，而是在每个方向、每个锚点位置上，独立地对 16 个概率进行加权求和。卷积核权重又恰好是 `[0, 1, ..., 15]`，所以计算结果就是数学期望。最后再将输出整理成 `(B, 4, A)`，便得到了四个方向的距离。

### 7.3.4 验证 DFL 模块

代码写完后，有兴趣的读者可以通过一个简单的函数来确认一下模块计算的结果是否与数学公式一致。

在 `ch07/main.py` 中添加以下测试代码：

```python
import torch
import torch.nn.functional as F
from nn.block import DFL

def dfl_test():
    reg_max = 16

    # 一组预测分数示例
    logits = torch.tensor([
        -2.0, -1.0, 0.5, 2.0, 3.5,
         2.0,  0.5, -0.5, -1.0, -2.0,
        -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
    ])

    # 方法一：直接使用数学公式计算期望
    bins = torch.arange(reg_max, dtype=logits.dtype)
    probs = F.softmax(logits, dim=0)
    expected = (probs * bins).sum()

    # 方法二：使用 DFL 模块
    # 将同一组预测分数复制到四个方向，便于对照结果
    x = logits.repeat(4).view(1, 4 * reg_max, 1)

    dfl = DFL(reg_max)
    actual = dfl(x)

    # 四个方向使用相同的分数，因此输出应该都等于 expected
    torch.testing.assert_close(actual, expected.expand_as(actual))

    print(f"数学期望：{expected.item():.4f}")
    print("DFL 模块输出：", [round(v, 4) for v in actual.flatten().tolist()])
    print("验证通过")
    
if __name__ == "__main__":
    dfl_test()
```

正常情况下，运行 `DFL_Test` 函数，数学期望约为 `4.0916`，DFL 模块的四个输出也都应约为 `4.0916`，程序最终输出“验证通过”。

到这里，DFL 模块就实现了：**将每个方向的 16 个离散预测分数转换为一个连续距离，最终得到 `left`、`top`、`right`、`bottom` 四个偏离值。**

接下来，我们就可以把这四个偏离与锚点坐标结合起来，先计算水平框，再进一步解码旋转框。

***

## 7.4 锚点与解码

DFL模块把离散预测转换成了四个距离。接下来还需要两步：先生成每个特征格子的锚点，再根据锚点和距离计算目标框。

这几项工作因为和检测头相关，具体的代码先写在 `head.py` 文件中。

打开 `ch07/nn/head.py`，顶部确认有以下导入。已有的语句不需要重复添加：

```python
import torch
import torch.nn as nn
from .conv import Conv
import math
```

### 7.4.1 make\_anchors函数：生成锚点

**锚点就是特征图上一个格子的中心**。后面我们甚至经常用锚点来代替他所在的这个特征格。取中心是因为中心向四条边辐射距离更对称，并不意味着真实目标一定处于格子中心。

假设特征图宽为 `w`、高为 `h`，并把每个格子的边长看作 1 个单位。那么，沿 X 轴的格子索引是：`0, 1, 2, ..., w-1`，沿 Y 轴则是：`0, 1, 2, ..., h-1`，每个格子的中心比它的左上角索引多 `0.5`个单位，因此中心坐标分别是：

```text
sx = [0.5, 1.5, 2.5, ..., w-0.5]
sy = [0.5, 1.5, 2.5, ..., h-0.5]
```

这里的 X 坐标沿列方向变化，Y 坐标沿行方向变化。`torch.arange()`函数用来生成这两组一维坐标；`torch.meshgrid()` 函数则把它们扩展成两个二维坐标网格。

需要注意的是，这里得到的坐标仍然是以特征图格子为单位的坐标，而不是原始图像中的像素坐标。代码需要同时记录每个锚点对应的 `stride`，以便后续进行坐标尺度转换。

有了这个基础思路，我们就可以写 `make_anchors` 函数了，请把以下代码录入到 `nn/head.py`文件：

```python
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    根据多尺度特征图生成锚点及其对应的步长。
    对标: ultralytics/utils/tal.py::make_anchors
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        
        # X 坐标：沿列方向变化，得到每列格子的中心位置
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        
        # Y 坐标：沿行方向变化，得到每行格子的中心位置
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        
        # 把 X、Y 两组一维坐标各自组合成二维网格，第一个网格保存 Y 坐标，第二个网格保存 X 坐标
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        
        #按 (x, y) 顺序组合每个格子的中心坐标，再展平成点列表
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        
        #记录当前尺度下每个锚点对应的 stride
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)
```

这段代码基本上就是按照刚才的思路写的，有三个地方值得再仔细看一下：

###### 1、最终每个格子的坐标是怎么拼接出来的？

`torch.arange` 的默认步长是1，对应一个特征图中的1个格子。 `w` 是特征图的宽度，`torch.arange(end=w)` 生成的是**列**方向的坐标，加上偏移量0.5格，得到的结果是：`[0.5, 1.5, 2.5, ..., w-0.5]`。

同样，`torch.arange(end=h)` 生成的是**行**方向的坐标，得到的结果是： `[0.5, 1.5, 2.5, ..., h-0.5]`。

`sy, sx = torch.meshgrid(sy, sx, indexing="ij")`，注释里说，把 X、Y 两组一维坐标各自组合成二维网格，注意这里的**各自**，他的意思是，把sy的值放到每一个特征图上的格子里（其实就是每一个格子的横坐标），形成一个新的二维向量sy，新的sy中同一行的值是一样的；把sx的值放也到每一个特征图上的格子里（其实就是每一个格子的纵坐标），形成一个新的二维向量sx，新的sx中同一列的值也是一样的。

最后用 `torch.stack((sx, sy), -1)` ，拼出来每一个格子的完整坐标 `(x, y)` 对——正好对应图像的"行是 `y`、列是 `x`"约定。注意一下顺序，不要写反，否则检查起来会很困难，因为即便写反了，运行的时候是不会报错的，但出来的结果不对。

文字说起来好像有点难理解，看动图吧，再结合着这几个函数的功能，可能理解起来容易些，不要着急，慢慢捋一下。

<p align="center">
  <img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6b2219a8cdd549e781c8dbf4123f7530~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5rGf5ram6Iif:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzI5NTY4NzI5NTY0NzQ4MyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1795396824&x-orig-sign=DkMeR34BNYBva2fEzKe94hkGW0U%3D" width="85%" alt="">
</p>

###### 2、显式传入 dtype 的原因

`torch.arange` 默认生成整数张量。如果不显式传 `dtype`，加上 `0.5` 之后虽然也能变成浮点，但在混合精度训练时容易引发类型不一致的报错。所以这里一律显式传入。

###### 3、stride\_tensor 的作用

注意，锚点坐标仍然以特征图格数为单位。例如，P3 上的 `(5.5, 10.5)`，并不是原图像素坐标。两者之间差一个 `stride`。所以我们需要给每个锚点记住它所属尺度的步长，形状是 `(N, 1)`。后面解码时，把特征图上的距离乘以这个步长，才能把解码结果换算到原图尺度。

在  `nn/__init__.py` 文件中增加对 `make_anchors`方法的导出。

讲到这里，我们已经自觉不自觉的涉及到了很多张量运算。不知道读者发现没有，这一卷中，**很多代码不再使用大家熟悉的** `for` **循环，而是基于pytorch库直接对整个张量进行运算**。这个问题值得拿出来单独说一下，因为后续的内容，我们经常需要同时处理成千上万个锚点，或者计算大量预测框与标注框之间的关系。如果每次都逐个处理，不仅代码会变得繁琐，也无法充分利用 PyTorch 提供的高效张量运算。

先从一个简单的例子来直观的感受一下pytorch的张量操作。

假设我们有三个数，希望每个数都加上 10。

用循环，可以这样写：

    numbers = [1, 2, 3]
    results = []

    for x in numbers:
        results.append(x + 10)

    print(results)

输出结果：

    [11, 12, 13]

这种写法很好理解：先处理 1，再处理 2，最后处理 3。

如果使用 PyTorch 张量，也可以这样写：

    import torch

    numbers = torch.tensor([1, 2, 3])
    results = numbers + 10

    print(results)

输出结果同样是：

    tensor([11, 12, 13])

两种写法完成的是同一件事，只是实现方式不同。

    逐个计算                         张量运算

    1  →  1 + 10 = 11               ┌ 1 ┐       ┌ 11 ┐
    2  →  2 + 10 = 12       →       │ 2 │ + 10 =│ 12 │
    3  →  3 + 10 = 13               └ 3 ┘       └ 13 ┘

左边是通过循环逐个计算，右边则是直接对整个张量执行加法。

需要注意的是，**张量运算并不是另一套计算规则，而是将一批数据组织起来，利用支持批量处理的运算完成计算。** 对于这里的加法，我们不需要自己编写循环，PyTorch 就能一次得到所有结果。

实际运行时，批量运算也不意味着每一种情况下都必然更快。它的优势在于能够减少大量逐元素的 Python 操作，并让 PyTorch 更好地利用底层计算能力。

好，我们言归正传，有兴趣的读者可以用三个虚拟特征图，检查锚点总数和坐标顺序是否正确，打开 `ch07/main.py`，添加对 `make_anchors`的依赖，然后输入 `make_anchors_test` 函数：

```python
# 目标：用三个假特征图跑一遍 make_anchors，检查锚点数量和坐标范围
import torch
from nn.head import make_anchors

def make_anchors_test():
    # 三个尺度的假特征图：只关心形状，数值随便
    feats = [
        torch.randn(1, 64, 80, 80),   # P3
        torch.randn(1, 128, 40, 40),  # P4
        torch.randn(1, 256, 20, 20),  # P5
    ]
    # 三个尺度各自的步长
    strides = torch.tensor([8.0, 16.0, 32.0])

    anchors, stride_tensor = make_anchors(feats, strides, grid_cell_offset=0.5)

    print(f"锚点总数: {anchors.shape[0]}")
    print(f"锚点张量形状: {list(anchors.shape)}")
    print(f"步长张量形状: {list(stride_tensor.shape)}")

    idx = 0
    for i, (h, w) in enumerate([(80, 80), (40, 40), (20, 20)]):
        n = h * w
        pts = anchors[idx:idx + n]
        print(f"P{i+3} (stride={strides[i]:.0f}): 锚点数={n}, "
              f"x范围=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], "
              f"y范围=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
        idx += n
```

运行后你会看到类似下面的输出：

```text
锚点总数: 8400
锚点张量形状: [8400, 2]
步长张量形状: [8400, 1]
P3 (stride=8): 锚点数=6400, x范围=[0.5, 79.5], y范围=[0.5, 79.5]
P4 (stride=16): 锚点数=1600, x范围=[0.5, 39.5], y范围=[0.5, 39.5]
P5 (stride=32): 锚点数=400, x范围=[0.5, 19.5], y范围=[0.5, 19.5]
```

三个关键点都对上了：总数 `8400`、坐标从 `0.5` 开始、每个尺度的范围都是 `[0.5, 边长-0.5]`。

***

### 7.4.2 dist2bbox：四个距离 → 水平框

有了锚点坐标，再加上 `DFL` 输出的四个距离，就能计算出目标框的具体位置了。

我们先把水平框算出来，稍后考虑进一步构建旋转框（OBB）。

#### 1. 从锚点和四个距离出发

假设锚点坐标为 `(ax,ay)`，网络预测了锚点到目标框四条边的距离分别为：`left`、`top`、 `right`、`bottom`，它们之间的关系很清晰：

           左边界                  右边界
              ┌──────────────────┐
              │       ↑ top      │
              │       │          │
              │left ← ● → right  │
              │       │          │
              │       ↓ bottom   │
              └──────────────────┘
                      ● = 锚点 (ax, ay)

左上角就是"锚点往左上走"，右下角就是"锚点往右下走"。注意，图像坐标系中，X 轴向右为正，Y 轴向下为正。因此，向左、向上移动时坐标值减小，向右、向下移动时坐标值增大。

可以得到目标框左上角和右下角的坐标：

    x1 = ax - left
    y1 = ay - top

    x2 = ax + right
    y2 = ay + bottom

这样，我们就得到了水平框的基本表示形式 `xyxy`。

#### 2. 编写 dist2bbox 函数

在 `make_anchors()` 后面添加以下函数：

    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        """
        根据锚点坐标和四个边界距离，计算水平边界框。
        distance:      锚点到左、上、右、下边界的距离
        anchor_points: 锚点坐标 (x, y)
        xywh:          True 返回 (cx, cy, w, h)，False 返回 (x1, y1, x2, y2)
        dim:           四个距离所在的维度
        """

        # 1. 将四个距离拆成两组：
        lt, rb = distance.chunk(2, dim=dim)

        # 2. 根据锚点和四个距离，计算左上角、右下角
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb

        # 3. 根据需要，选择返回哪种边界框表示形式
        if xywh:
            # 中心坐标 = (左上角 + 右下角) / 2
            c_xy = (x1y1 + x2y2) / 2

            # 宽、高 = 右下角坐标 - 左上角坐标
            wh = x2y2 - x1y1

            # 按照 (cx, cy, w, h) 的顺序拼接
            return torch.cat((c_xy, wh), dim=dim)

        # 否则，就返回左上角、右下角形式的矩形表示，也就是 (x1, y1, x2, y2)
        return torch.cat((x1y1, x2y2), dim=dim)

详细拆解一下这个函数的代码。

**第一步：拆分四个距离。**

    lt, rb = distance.chunk(2, dim=dim)

这里使用了 PyTorch 的 `chunk()` 函数。它的作用是沿指定维度，将一个张量尽可能均匀地切分成指定数量的部分。`dim` 参数就是用来指定这个维度的，后面我们单独解释这个参数。

`chunk(2, dim=dim)` 中的 `2` 表示分成两份，而不是每份包含两个元素。

假设某个锚点预测的四个距离为：

    distance = [2, 3, 4, 5]
    # 依次表示 left、top、right、bottom

切分后得到：

    lt = [2, 3]  # left、top
    rb = [4, 5]  # right、bottom

这样就能分别与锚点的 `(x, y)` 坐标相减或相加。

**第二步：计算左上角和右下角。**

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

这个代码就比较简单了，代码中的减法和加法正是将前面的几何关系直接转化成了张量运算。实际处理时，PyTorch 可以一次计算大量锚点，不需要逐个锚点编写循环。

**第三步：根据需要，返回不同的边界框格式。**

计算出左上角和右下角之后，水平框就已经确定了。接下来，只是决定以什么形式输出。

当 `xywh=True` 时，表示要返回“中心点+宽高”的表示形式：

    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat((c_xy, wh), dim=dim)

当 `xywh=False` 时，函数返回“左上角+右下角”的表示形式：

    (x1, y1, x2, y2) = (8, 17, 14, 25)

这两种形式描述的是同一个水平框，只是表示方式不同。

其中，`torch.cat()` 用于沿指定维度拼接张量。例如，`c_xy` 保存中心坐标，`wh` 保存宽高，将它们拼接起来，就得到完整的 `(cx, cy, w, h)`。

#### 3. 理解 distance 的张量形状

代码中还有一个容易让人困惑的参数：`dim`。它指定了四个距离存放在 `distance` 张量的哪个维度。不同的代码可能采用不同的张量排列方式。

例如：

    # 情况一：最后一维保存四个距离
    distance.shape = (B, A, 4)
    # 使用默认参数 dim=-1

这里，`B` 表示批次大小，`A` 表示锚点数量，最后的 `4` 表示每个锚点的四个距离。此时，`chunk(2, dim=-1)` 会把最后一维拆成两个大小为 2 的部分。

如果张量排列方式是：

    # 情况二：第二维保存四个距离
    distance.shape = (B, 4, A)
    # 应使用 dim=1

那么就需要沿第二维切分，即使用 `chunk(2, dim=1)`。相应地，`anchor_points` 的形状也需要能够与切分后的张量正确进行运算。

因此， `dim` **必须与实际的张量布局保持一致。** 本函数默认使用 `dim=-1`，适用于四个距离位于最后一维的情况；如果采用 `(B, 4, A)` 的布局，就需要显式传入 `dim=1`。

***

### 7.4.3 dist2rbox：四个距离 + 角度 → 旋转框

水平框的边与图像坐标轴平行，因此四个距离可以直接用 X、Y 坐标的加减法处理。

旋转框则不同：`left`、`right`沿着目标框自身的宽度方向定义，`top`、`bottom`沿着目标框自身的高度方向定义。这两个方向可能与图像的 X、Y 轴不重合。

因此，不能简单地先用 `dist2bbox()` 算出水平框，再围绕这个水平框的中心旋转。

看图可能理解起来更容易一些：

<p align="center">
  <img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b31e16bab9c44e1ca5dde0a4660af977~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5rGf5ram6Iif:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzI5NTY4NzI5NTY0NzQ4MyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1795396824&x-orig-sign=%2BUEF5TNzTa63dKP7kj0JWHwYWBo%3D" width="95%" alt="">
</p>

理解了这个关键之后，我们可以来考虑一下如何计算旋转目标框了。我们可以分两步理解这个过程。

**第一步：计算中心相对于锚点的局部偏移。**

沿框宽度方向的偏移为：$x_f = \frac{right-left}{2}$，沿框高度方向的偏移为：$
y_f = \frac{bottom-top}{2}$。

这里的 `(x_f)` 和 `(y_f)` 是按照目标框自身方向定义的偏移量，还不能直接当作图像坐标系中的 `X、Y`
偏移。

**第二步：计算两个局部偏移在图像坐标轴上的投影。**

设预测角度为 `pred_angle` ，代码采用的角度单位是弧度。按照本卷采用的图像坐标约定，转换公式为：

$$
\Delta x = x_f\cos\theta-y_f\sin\theta
$$

$$
\Delta y = x_f\sin\theta+y_f\cos\theta
$$

可以把它理解为：局部偏移在图像 `X、Y` 方向上各自贡献了一部分投影，三角函数给出了这些投影的比例。把两个投影加回锚点坐标，就得到旋转框中心。

旋转目标框的宽度和高度不需要再乘三角函数，因为无论怎么旋转，矩形框的宽和高都不会变化，直接计算即可：

$$
w=left+right,\qquad h=top+bottom
$$

有了这个思路之后，我们在 `nn/head.py` 中添加 `dist2rbox()` 函数。

    def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
        """
        根据四个边界距离、角度和锚点，解码旋转框。
        pred_dist:    四个距离，顺序为 left、top、right、bottom
        pred_angle:   预测的旋转角度，使用弧度
        anchor_points: 锚点坐标 (ax, ay)
        dim:          四个距离所在的维度
        返回: (cx, cy, w, h)，不包含角度本身。
        """

        # 1. 将四个距离拆成 (left, top) 和 (right, bottom)两组
        lt, rb = pred_dist.chunk(2, dim=dim)
        
        # 2. 计算框中心相对于锚点的局部偏移
        xf, yf = ((rb - lt) / 2).chunk(2, dim=dim)

        # 3. # 将局部偏移投影到图像坐标系的 X、Y 方向
        cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
        x = xf * cos - yf * sin
        y = xf * sin + yf * cos

        # 4. 局部偏移转换完成后，加回锚点坐标，得到中心坐标
        xy = torch.cat((x, y), dim=dim) + anchor_points

        # 5. 宽度和高度分别是相对两侧距离之和
        wh = lt + rb

        # 6. 返回 (cx, cy, w, h)，角度由角度分支单独保留
        return torch.cat((xy, wh), dim=dim)

实际调用时，距离张量的形状为 `(B, 4, A)`，因此需要传入`dim=1`；锚点坐标的形状为 `(1, 2, A)`，角度的形状为`(B, 1, A)`。这三者的维度安排必须匹配。

注意，`dist2rbox()` 本身不把 `pred_angle` 拼接到返回值中。角度仍然是单独的预测结果，后续检测流程可以根据需要将它与中心、宽高组合成完整的旋转框表示 `(cx, cy, w, h, angle)`。

至此，我们已经完成了从四个距离到旋转框参数的解码。

接下来，可以开始实现检测头了。

