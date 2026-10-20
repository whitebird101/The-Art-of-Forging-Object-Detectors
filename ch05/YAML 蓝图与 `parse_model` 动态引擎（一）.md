
# 阵图法则：YAML 蓝图与 `parse_model` 动态引擎（一）



前4卷，我们把每一个“基础阵法”都焊死在“法宝”里。虽然法宝能用，但每换一种构型，就得把整件法宝拆碎重铸。而真正的炼器宗师，他们只需要画一张阵图，把阵法清单和连接拓扑写在“玉简”上，然后交给阵法引擎自动组装。

这张阵图，就是YAML配置文件，而自动组装引擎就是 `parse_model()`。

本卷，将带领你从"手工焊接匠"晋升为"阵图师"。

---

## 5.1 阵图密码 ── YAML 网络拓扑定义

### 5.1.1 为什么需要一张“阵图”？

回忆第 4 章的 `model.py`，我们的 `YOLOLikeDetector` 大致是这样：

```python
class YOLOLikeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.out_channels = num_classes + 8

        # ---------------- Backbone ----------------
        self.stem = Conv(3, 16, 3, 2)             # 硬编码
        self.down1 = Conv(16, 32, 3, 2)           # 硬编码
        self.stage1 = C3k2(32, 32, n=1)           # 硬编码
        ... 
        ... 
```

每一层的输入通道、输出通道、卷积核大小，全部写死在Python代码里。想换个模型结构，甚至调整一下模型配置？必须改 `model.py` 文件中的代码。这就好比一位炼器师每铸一件法器都要重新烧炉、重画图纸、重刻法阵，不说效率低，至少也很不方便。

现代工业级框架，例如 Ultralytics YOLO，通常采用的是“配置与代码分离”的思想：

- YAML：负责描述模型长什么样

- parse_model引擎：负责读取 YAML 并自动创建模型

以后换模型结构时，只需要修改 YAML文件，代码一行不动。这也是为什么很多人在修改YOLO模型时，不需要深入源码，只需要修改对用模型的YAML文件的原因。

---

### 5.1.2 YAML 是什么？

YAML 是一种人类可读的配置文件格式，纯文本文档。第一次接触不需要了解太多，先记住三条即可。

- 缩进表示层级（类似Python），一般用空格来缩进

- 冒号后跟空格表示键值对，例如 `nc: 2` 等价于Python字典 `{"nc": 2}`

- 短横线开头表示列表元素，例如 `- [-1, 1, Conv, [64, 3, 2]] ` 表示列表中的一个元素

掌握这三条，就足够读懂本卷的 YAML 了。

也许你还有点迷茫，不要紧，随着接下来我们一点点写出完整的检测模型YAML，这些规则会越来越清晰的，所以不用在这里做过多的纠结。

---

### 5.1.3 完整阵图：`yolo-obb-mini.yaml`

将 `ch04/` 文件夹下的代码复制到 `ch05/` 文件夹，然后在 `ch05` 下新建 `cfg`子文件夹，在其中新建 `yolo-obb-mini.yaml` 文件，新建完成后，文件路径是：`ch05/cfg/yolo-obb-mini.yaml`。

我们先把第4卷写死在 `model.py` 中的网络结构，等价地翻译成 YAML，先大概了解一下这张“玉简”最终长什么样。

不建议马上整段复制粘贴。可以先大致看一遍整体结构，再跟着后面的逐段解释，把内容一步一步输入到 YAML 文件中。

```yaml
# =========================================================
# yolo-obb-mini.yaml
# 与第4卷 YOLOLikeDetector 等价
# =========================================================

# ---------------- 基础参数 ----------------
nc: 2
ch: 3

# ---------------- 模型缩放系数 ----------------
# [depth_multiple, width_multiple, max_channels]
scales:
  n: [0.50, 0.25, 1024]

# ---------------- Backbone ----------------
backbone:
  # [from, repeats, module, args]

  - [-1, 1, Conv, [64, 3, 2]]                       # 0:  3 -> 16
  - [-1, 1, Conv, [128, 3, 2]]                      # 1: 16 -> 32
  - [-1, 1, C3k2, [128, False, 0.5]]                # 2: 32 -> 32

  - [-1, 1, Conv, [256, 3, 2]]                      # 3: 32 -> 64
  - [-1, 1, C3k2, [256, False, 0.5]]                # 4: P3，64通道

  - [-1, 1, Conv, [512, 3, 2]]                      # 5: 64 -> 128
  - [-1, 1, C3k2, [512, False, 0.5]]                # 6: P4，128通道

  - [-1, 1, Conv, [512, 3, 2]]                      # 7: 128 -> 128
  - [-1, 1, C3k2, [512, False, 0.5]]                # 8: 128 -> 128
  - [-1, 1, SPPF, [512, 5]]                         # 9: P5，128通道

# ---------------- Head ----------------
head:
  - [-1, 1, Conv, [256, 1, 1]]                      # 10: lateral5，128 -> 64
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 11
  - [[-1, 6], 1, Concat, [1]]                       # 12: 64 + 128 = 192
  - [-1, 1, C3k2, [256, False, 0.5]]                # 13: F4，64通道

  - [-1, 1, Conv, [128, 1, 1]]                      # 14: lateral4，64 -> 32
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]      # 15
  - [[-1, 4], 1, Concat, [1]]                       # 16: 32 + 64 = 96
  - [-1, 1, C3k2, [128, False, 0.5]]                # 17: F3，32通道

  - [[17, 13, 9], 1, SimpleHead, [nc]]              # 18: F3、F4、P5
```

先不要急着理解每一个数字。我们先看整体结构，整张“阵图”可以先分成四部分：

```text
nc / ch      → 基础参数
scales       → 缩放系数，控制模型大小
backbone     → 主体结构，提取多尺度特征
head         → 头部结构，FPN 特征融合 + 多尺度预测
```

接下来，我们从第一行开始逐段解读，您可以跟着解读的节奏，逐步把内容写入yolo-obb-mini.yaml文档。

---

### 5.1.4 手搓阵图

#### 5.1.4.1基础参数
先把下面内容录入 `yolo-obb-mini.yaml`：

```yaml
# 基础参数
nc: 2
ch: 3
```

其中：

- `nc: 2`：类别数（Number of Classes）。我们的虚拟数据集只有2类，所以这里写2。如果后续使用真实数据集，需要将这里修改为真实数据集的类别数，比如DOTA数据集有15类，这里需要修改为15；
- `ch: 3`：输入通道数。RGB图像是3通道，所以这里目前是3，如果你处理灰度图，改成1即可。

这两个参数属于整个模型都会使用的全局配置，后面的 `parse_model()` 会读取它们。

---

#### 5.1.4.2模型缩放系数

继续在 `yolo-obb-mini.yaml` 文件中输入：

```yaml
# 模型缩放系数 [depth_multiple, width_multiple, max_channels]
scales:
  n: [0.50, 0.25, 1024]
```

这里的 `n` 表示 **Nano**，也就是轻量版本。

同一张基础阵图，可以通过不同的缩放系数生成不同大小的模型。比如，YAML里写的通道数（如64、128、512）是"基准值"，实际通道数 = min(基准值, max_channels) * width_multiple，然后向上取8的整数倍。为什么要取8的整数倍，卖个关子，我们稍后再说。

那为什么需要缩放呢？因为同一张阵图，可以通过不同的缩放系数，生成nano / small / medium / large / xlarge等多种体型的模型。例如：

```yaml
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
```

不过本卷为了降低复杂度，我们只讨论模型的 **Nano 版**，**Nano 版**的缩放系数是 `n: [0.50, 0.25, 1024]`。

缩放系数 `[depth_multiple, width_multiple, max_channels]` 的具体含义如下：

| 参数 | 含义 | Nano 值 | 作用 |  示例  |
| --- | --- | ---: | --- | --- |
| `depth_multiple` | 深度倍率 | 0.50 | 控制模块重复次数 | C3k2 的 repeats 乘以 0.5，向上取整到至少 1|
| `width_multiple` | 宽度倍率 | 0.25 | 控制实际通道数 | 蓝图里写的基础通道数乘以 0.25，如 YAML 写 64 → 实际 16|
| `max_channels` | 最大通道数 | 1024 | 限制通道上限 | 通道数不超过此值|

`depth_multiple` 是怎么控制“深度”的呢？这里的“深度”主要体现在某些模块要重复多少次。先假设某个模块在基础阵图中要求重复4次，深度倍率depth_multiple = 0.50，那么实际重复次数大致会变成 `4 × 0.50 = 2`。也就是说，`depth_multiple` 越小，模型通常越浅。

后面看到 YAML 中的 `repeats`，就会知道这个系数具体作用在哪里。

`width_multiple` 又怎么控制“宽度”呢？这里的“宽度”主要指每一层设定多少个**输出通道数**。假设某一层在YAML中定义的基础输出通道数为 `128`，如果 `width_multiple = 0.25`，那么实际通道数大致会缩放为：`128 × 0.25 = 32`。

所以，可以简单的理解为：width_multiple 越小 → 每层通道越少 → 模型更轻；width_multiple 越大 → 每层通道越多 → 模型更宽。

`max_channels` 用于限制最大通道数，防止模型深层通道数无限增大。

因此，这三个参数可以简单理解为：

```text
depth_multiple → 控制“叠多少层”
width_multiple → 控制“每层有多少通道”
max_channels   → 控制“最多允许有多少通道”
```

同一张蓝图，不同的缩放系数，生成不同大小的模型——这就是阵图的威力。

---

#### 5.1.4.3模型主体阵图：Backbone

接下来继续写 Backbone：

```yaml
backbone:
  # [from, repeats, module, args]

  - [-1, 1, Conv, [64, 3, 2]]                       # 0:  3 -> 16
  - [-1, 1, Conv, [128, 3, 2]]                      # 1: 16 -> 32
  - [-1, 1, C3k2, [128, False, 0.5]]                # 2: 32 -> 32

  - [-1, 1, Conv, [256, 3, 2]]                      # 3: 32 -> 64
  - [-1, 1, C3k2, [256, False, 0.5]]                # 4: P3，64通道

  - [-1, 1, Conv, [512, 3, 2]]                      # 5: 64 -> 128
  - [-1, 1, C3k2, [512, False, 0.5]]                # 6: P4，128通道

  - [-1, 1, Conv, [512, 3, 2]]                      # 7: 128 -> 128
  - [-1, 1, C3k2, [512, False, 0.5]]                # 8: 128 -> 128
  - [-1, 1, SPPF, [512, 5]]                         # 9: P5，128通道
```

这就是整张阵图最核心的格式。

`[from, repeats, module, args]` 中的四个参数分别表示：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `from` | 输入来自哪一层 | `-1` 表示上一层 |
| `repeats` | 模块重复次数 | `2` 表示基础配置重复 2 次 |
| `module` | 使用哪个模块 | `Conv`、`C3k2`、`SPPF`等 |
| `args` | `module`参数指定的模块的参数 |例如 `[64, 3, 2]` ，数字与选用模块的初始化参数基本对应|

##### 第 0 层：`[-1, 1, Conv, [64, 3, 2]]`

拆开来看：

```text
from    = -1           # 输入来自上一层。对于第 0 层来说，就是原始图像；
repeats = 1            # 只使用 1 个 `Conv`
module  = Conv         # 使用我们在 `nn/conv.py` 中实现的 `Conv`
args    = [64, 3, 2]   # 基准输出通道数 64，卷积核 3，stride=2。
```

注意，这里的 `64` 是阵图中的**基准通道数**。因为当前使用的是 Nano 版本，`width_multiple = 0.25`，所以实际输出通道会缩放为：`64 × 0.25 = 16`，这与第4卷中的代码 `self.stem = Conv(3, 16, 3, 2)` 是完全对应的。stride=2，空间尺寸减半。如果输入是 `[3, 256, 256]`，经过这一层后得到的张量的性质将是 `[16, 128, 128]`。

也许你会有一个疑问，参数部分写一个 `[64, 3, 2]`，怎么就知道64对应基准输出通道数，3对应卷积核，2对应stride？

这个其实是由模块的构造函数和 `parse_model()` 的实现逻辑来决定的。

通常情况下，YAML 中 `args` 里的参数顺序，会按照对应模块构造函数的参数顺序来填写，其中能够由 `parse_model()` 根据前面网络结构自动计算出来的参数，就不需要在 YAML 中手工写。

我们之前实现的Conv模块，其构造函数的定义是：
```python
class Conv(nn.Module):
    """标准卷积: Conv2d + BatchNorm2d + SiLU"""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        ……
        ……
```
由于输入张量通道数c1根据上一层输出自动计算，因此64对c2，3对应k，2对应s，如果有更多的参数，则依此类推。


##### 第 1 层：`[-1, 1, Conv, [128, 3, 2]]`

这一层仍然接上一层。基准输出通道数 `128`，Nano 版实际通道数：`128 × 0.25 = 32`，`stride=2`所以特征图尺寸再减半。

##### 第 3 层：`[-1, 1, C3k2, [128, False, 0.5]]`

这一层开始出现 `C3k2`：

```text
from = -1
repeats = 1
module = C3k2
args = [128, False, 0.5]
```

基础重复次数repeats是 1，而 Nano 版本 `depth_multiple = 0.50`，因此实际重复次数会根据深度缩放规则调整，最终至少保留 1 次。参数args表示基础输出通道数128(经过 `width_multiple=0.25` 缩放后=32), c3k=False, e=0.5。

##### 后续 Backbone

后面的规律基本一样：

```text
Conv        → 下采样
C3k2        → 特征提取
Conv        → 再下采样
C3k2        → 再提取
...
SPPF        → 扩大最深层特征的感受野
```

经过 Nano 的宽度缩放后，各关键层实际输出通道与第4卷中保持一致：

```text
第4层 → P3：64 通道
第6层 → P4：128 通道
第9层 → P5：128 通道
```

注意模型结构中 `C3k2` 模块的 `c3k` 参数始终是 `False` ，这是因为我们目前实现的 `C3k2` 模块仅支持`c3k=False` 。YOLO官方源码中实现了 `c3k=True` 的情况，所以如果你看YOLO官方的YAML，会发现部分 `C3k2` 模块中用了 `True` 参数。

---

#### 5.1.4.4检测头Head

继续写Head部分：

```yaml
head:
  - [-1, 1, Conv, [256, 1, 1]]                   # 10
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 6], 1, Concat, [1]]                    # 12
  - [-1, 1, C3k2, [256, False, 0.5]]             # 13

  - [-1, 1, Conv, [128, 1, 1]]                   # 14
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 15
  - [[-1, 4], 1, Concat, [1]]                    # 16
  - [-1, 1, C3k2, [128, False, 0.5]]             # 17

  - [[17, 13, 9], 1, SimpleHead, [nc]]           # 18
```

仔细分析Head部分的阵图，会发现这个阵图与第4卷YOLOLikeDetector中的FPN和检测头的代码是对应的：

```python

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

```

细心的读者可能会发现YAML中的通道数看起来比第 4 卷代码中设定的通道数大，这是因为YAML中写的是**基准输出通道数**。经过 Nano 的缩放系数 `width_multiple = 0.25`缩放后，实际通道数仍然是第 4 卷中的一致。

相比Backbone的阵图，Head 中出现了几个变化，一个是`from` 开始从 `-1` 变成列表：

```text
[-1, 6]
[-1, 4]
[17, 13, 9]
```
另一个是模块部分出现了几个新面孔：

```text
nn.Upsample
Concat
SimpleHead
```
我们分别来说明。

##### 1、暗线勾连：`from` 参数如何描述跨层跳接

Backbone 中，大多数层都是 `from = -1`，表示当前层直接使用上一层的输出。数据像流水一样向前传，
这是最简单的线性连接。

Head 中出现 `from = [-1, 6]`，表示这一层同时读取两份特征：

- `-1`：上一层，对于我们刚刚写的Head部分的阵法而言，也就是第 11 层 Upsample 的输出；
- `6`：Backbone 第 6 层的输出，也就是 P4 特征。

然后交给 `Concat` 沿通道维拼接。这就是我们之前讲过的 FPN（Feature Pyramid Network，特征金字塔网络）的核心操作——自顶向下融合。让高层的语义信息（"知道这是一架飞机"）和低层的细节信息（"轮廓的精确边缘"）汇聚在一起。

类似的连接还有：

| 层 | `yaml` | 含义 |
| --- | --- | --- |
| 16 | `[[-1, 4], 1, Concat, [1]]` | 上一层 Upsample 输出 + Backbone 第4层 P3 特征 |
| 18 | `[[17, 13, 9], 1, SimpleHead, [nc]]` | 同时读取 F3 / F4 / P5 三个尺度的特征 |

所以，`from` 本质上就是在描述**这一层的输入，到底从哪里来。**

可以看到，如果仍然采用第 4 章的硬编码方式，就需要手工管理很多中间变量，例如 `p3`、`p4`、`p5`、`up5`、`f4`、`up4`、`f3`。

现在，这些连接关系都被明确写进 YAML 的 `from` 中。以后 `parse_model()` 只需要按照 `from` 去找到对应层的输出，就能自动完成这些跨层连接。

这就是“阵图”的真正价值。

---

##### 2、新增模块

###### （1）`Concat`模块

为了支持阵图中的  `[[-1, 6], 1, Concat, [1]]` 部分，需要新增一个 `Concat` 模块。

打开 `ch05/nn/block.py`文件，在其中加入：

```python
class Concat(nn.Module):
    """
    沿指定维度拼接多个张量。
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
```

这个类本质上只是对 `torch.cat()` 做了一层封装，功能跟 `torch.cat()` 是一样的。

为什么要封装呢？因为 `parse_model()` 后面需要把 YAML 中的每一层都实例化成一个 `nn.Module`，而 `torch.cat` 本身只是普通函数，不是一个模块。封装以后，`Concat` 就可以和 `Conv`、`C3k2`、`SPPF` 一样，直接写进 YAML。

这里 `dimension=1`表示沿通道维拼接。对于 `[B, C, H, W]` 格式的特征图，`dim=1` 就是通道维，也就是 `[B, C, H, W]` 中的 `C` 这个维度。

例如如下两个张量：

```text
[B, 64, 16, 16]
[B, 128, 16, 16]
```

经过 `Concat` 拼接后得到：

```text
[B, 192, 16, 16]
```

其中的新的通道数192就是原始通道64和128相加得到的。

---

###### （2） `SimpleHead` 模块

YAML 的最后一层是 `[[15, 18, 21], 1, SimpleHead, [nc]]`，这里的 `SimpleHead` 其实不是突然增加的新检测结构。回忆第 4 卷，我们已经给 F3、F4、P5 分别设置了三个临时检测头：

```python
self.head3 = nn.Conv2d(32, self.out_channels, 1)
self.head4 = nn.Conv2d(64, self.out_channels, 1)
self.head5 = nn.Conv2d(128, self.out_channels, 1)
```

并在 `forward()` 最后写：

```python
return [
    self.head3(f3),
    self.head4(f4),
    self.head5(p5),
]
```

也就是说，第 4 卷其实已经有了一个“极简多尺度检测头”，只是当时三个 `1×1 Conv` 直接写在 `model.py` 中，没有单独封装。

到了这一卷，我们希望 YAML 中可以直接写 `SimpleHead` ，那它就必须和 `Conv`、`C3k2` 一样，要封装成一个可以被 `parse_model()` 创建的 `nn.Module`。

因此，我们需要把第 4 卷已经存在的三个 `1×1 Conv`，封装成一个独立的 `SimpleHead` 模块。

第 4 卷的代码中，检测头只是模型内部的几行代码。现在检测头需要封装成为一个独立模块，再继续放在 `model.py` 中就不合适了，因此我们在 `nn/` 目录下新增 `head.py `文件用来保存检测头模块，创建完成后，`ch05/nn/`目录的结构如下：

```text
ch05/
└── nn/
    ├── conv.py
    ├── block.py
    └── head.py      ← 新增
```

在 `head.py` 文件中输入：

```python
import torch.nn as nn

class SimpleHead(nn.Module):
    """
    极简多尺度检测头。

    参数：
        nc: 类别数
        ch: 各尺度输入通道数

    输出：
        list of [B, nc+8, Hi, Wi]
    """

    def __init__(self, nc=2, ch=()):
        super().__init__()

        self.nc = nc
        self.nl = len(ch)
        self.out_channels = nc + 8

        self.heads = nn.ModuleList(
            nn.Conv2d(c, self.out_channels, 1) for c in ch
        )

    def forward(self, x):
        return [head(xi) for head, xi in zip(self.heads, x)]
```

其中：参数 `ch=()` 记录三个尺度输入特征的通道数，例如 `parse_model()` 引擎根据阵图中的 `from = [17, 13, 9]`，可以自动得到 `ch = [32, 64, 128]`。

通过 `nn.Conv2d(c, self.out_channels, 1) for c in ch`这个循环，`SimpleHead` 会为每个尺度分别创建一个输入通道数为 `c` ( `c` 依次是32，64，128)，输出通道数为 `self.out_channels` 的 `1×1 Conv`。

现在大家应该已经非常熟悉python语言中这种循环的写法了。

`SimpleHead` 类的 `forward()` 方法接收 `[F3, F4, P5]`三个张量。通过zip()方法将检测头和对应的张量配对处理，然后分别输出我们在第4卷就熟悉的三个预测张量。

```text
F3 → [B, 10, 32, 32]
F4 → [B, 10, 16, 16]
P5 → [B, 10,  8,  8]
```

这里的 `SimpleHead` 仍然只是现阶段的占位检测头，后面会替换成真正的 OBB Head。但从这一章开始，它已经和 `Conv`、`C3k2`、`SPPF` 一样，可以被写进 YAML 阵图中。

两个新增的模块写完了，但还有非常重要的一件事没有做。

我们需要在 `nn\__init__.py` 文件中增加对新模块的导出，否则我们在python代码中，就没有办法像其他模块一样通过import导入。

在 `nn\__init__.py` 文件增加新模块的导出代码，由读者自行添加吧，也可以小小的检验一下自己的动手能力。

---

到这里，我们已经“画”出了第一张完整的模型阵图 `yolo-obb-mini.yaml`，其中最关键的格式是：

```text
[from, repeats, module, args]
```

四个位置分别负责：

```text
from     → 输入从哪里来
repeats  → 模块重复多少次
module   → 使用哪个网络模块
args     → 模块需要哪些参数
```

同时，我们也认识了 `scales`：它通过深度倍率和宽度倍率，让同一张阵图可以生成 `n / s / m / l / x` 等不同大小的模型。

现在，“阵图”已经画好了。但它还只是 YAML 中的一堆字符。

下一步，我们要真正打造阵法引擎 `parse_model()`，让程序能够读取 YAML，自动创建每一层、计算通道数，并按照 `from` 把所有阵法连接起来。

