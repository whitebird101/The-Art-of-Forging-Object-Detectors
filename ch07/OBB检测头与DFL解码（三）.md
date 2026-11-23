# 重塑视界 ------ OBB检测头与DFL解码（三）



## 7.5 构建检测头：从 Detect 到 OBB

前面已经准备好了 DFL 模块、锚点生成和两种框解码函数。接下来，我们要把这些工具组合起来，实现一个完整的旋转目标框（OBB） 检测头。

### 7.5.1 检测头的网络结构

在编写具体代码之前，先梳理一下检测头需要完成哪些工作，思路理清楚，写代码或者读代码，才能得心应手。

首先，网络需要根据输入的特征图预测目标的类别；其次，需要预测目标框的位置和大小；对于旋转框，还需要预测目标的角度。在前面的 `SimpleHead` 中，我们用一个1x1卷积同时输出分类和定位信息（之前还没有涉及到角度），这就好比让同一个人同时回答两个完全不同的问题：

*   "这是什么？"（分类）—— 需要关注纹理、颜色、形状等语义特征
*   "它在哪里？"（定位）—— 需要关注边缘、角点等空间特征

这两类任务的关注重点有很大差异。分类偏好高层语义（"看起来像飞机"），定位偏好低层几何（"边缘在这里"）。把它们塞进同一组卷积核，就像让一个人同时写论文和做体操——两件事都做不好。

既然需求冲突，塞到一起都做不好，那就干脆分开，给每个任务独立的卷积通道。各走各的经脉，互不干扰。

因为有分类、定位、角度三个不同的预测任务，我们给旋转目标框检测头设计三个独立分支：

*   **回归分支** `cv2`：预测目标框四个方向的距离分布。
*   **分类分支** `cv3`：预测目标属于各个类别的分数。
*   **角度分支** `cv4`：预测旋转目标框的角度。

说明一下，为了保持术语的一致性和专业性，边界框定位分支，后续我们将其称之为“回归分支”。但不要被这个名称唬住了。

以 OBB 检测头为例，看一下检测头内部的具体结构：

<p align="center">
  <img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f3a5c309f1f84c26928dcf81b0acd416~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5rGf5ram6Iif:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzI5NTY4NzI5NTY0NzQ4MyJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1796002283&x-orig-sign=irQQiLXEYjCgnY0XG9P%2FsYJ%2ByaI%3D" width="85%" alt="">
</p>

如图所示，每个分支都由三层卷积组成：

*   前两层使用封装好的 `Conv`，用于进一步处理输入特征；
*   最后一层使用普通的 `nn.Conv2d`，将特征映射为任务需要的原始预测分数。

也许有人会问，为什么三个分支的最后一层都使用普通的 `nn.Conv2d`，而不是封装好的 `Conv`？

这是因为它们输出的原始预测分数，后续还需要分别经过 DFL、`sigmoid` 或角度映射等处理。如果在最后一层加入 BN 和 SiLU，就会改变这些原始分数的数值，不符合当前的实现设计。

图中展示的是单个尺度的分支结构。实际代码会针对 P3、P4、P5 三个尺度分别建立一套独立的卷积模块，结构相同，但参数各不相同。

有一点要特别注意：**推理和训练时对数据的处理流程是不同的**。推理时，各尺度的预测结果经过整理和拼接，进入后续处理流程；而训练阶段，检测头需要保留各尺度的回归与分类预测结果，并将角度预测作为独立结果返回，供后续损失计算时使用。

仔细分析这三个分支，会发现回归、分类以及相关的推理流程中，有许多功能可以复用。例如，回归分支负责预测四个方向的距离分布，分类分支负责预测类别，DFL模块负责将离散分布转换为连续距离。这些功能无论是进行水平框检测、旋转框检测，甚至其他类似类型的检测，都可能用到，并非旋转框所独有的。

因此，在具体实现时，我们**将可复用的检测功能与旋转框特有的功能分开组织**。具体来说，可以先实现一个 `Detect` 模块，在其中组织回归分支 `cv2`、分类分支 `cv3` 以及默认的水平框预测与解码流程。随后，通过继承构建 `OBB` 模块，复用`Detect` 模块已有的功能，并增加角度分支 `cv4`，并将默认的框解码方式替换为旋转框解码方式。

通过拆分代码职责，可以让 OBB 的实现更加清晰，在后续扩展的时候，也可以避免重复编写已经完成的功能。

接下来，我们先编写 `Detect`，再逐步扩展出完整的 `OBB` 检测头。

## 7.5.2 基础检测头 `Detect`

打开 `ch07/nn/head.py`，先确认文件顶部已经导入需要的模块：

    import math
    import torch
    import torch.nn as nn

    from .conv import Conv
    from .block import DFL

##### 7.5.2.1定义属性并构建两个预测分支

在 `head.py` 中添加 `Detect` 类的初始化方法：

    class Detect(nn.Module):
        """基础水平框检测头。"""

        def __init__(self, nc=15, ch=()):
            super().__init__()

            
            self.nc = nc       # 类别数量        
            self.nl = len(ch)  # 检测尺度数量，例如 P3、P4、P5 共三个尺度        
            self.reg_max = 16  # 每个方向使用的离散候选值数量

            # 每个位置的原始输出通道数：回归分数 + 类别分数。注意：这里不包含 OBB 的角度输出
            self.no = nc + 4 * self.reg_max        
            self.stride = torch.zeros(self.nl) # 各尺度对应的步长，完整模型构建时再设置为实际值

            # 推理阶段使用的锚点缓存
            self.dynamic = False            # True 表示每次推理都重新生成锚点
            self.shape = None               # 记录上一次生成锚点时的特征图形状
            self.anchors = torch.empty(0)
            self.strides = torch.empty(0)

            # 回归分支和分类分支的中间通道数
            c2 = max(16, ch[0] // 4, self.reg_max * 4)
            c3 = max(ch[0], min(self.nc, 100))

            # 回归分支：每个尺度独立设置一套卷积权重
            # 输出 4 * reg_max 个通道，即四个方向各 16 个原始分数
            self.cv2 = nn.ModuleList([
                nn.Sequential(
                    Conv(c, c2, 3),                     # 第 1 层：3×3 卷积
                    Conv(c2, c2, 3),                    # 第 2 层：3×3 卷积
                    nn.Conv2d(c2, 4 * self.reg_max, 1)  # 第 3 层：1×1 输出层
                )  for c in ch ])

            # 分类分支：每个尺度独立设置一套卷积权重
            # 最终输出 nc 个类别原始分数
            self.cv3 = nn.ModuleList([
                nn.Sequential(
                    Conv(c, c3, 3),             # 第 1 层：3×3 卷积
                    Conv(c3, c3, 3),            # 第 2 层：3×3 卷积
                    nn.Conv2d(c3, self.nc, 1)   # 第 3 层：1×1 输出层
                ) for c in ch ])

            # 将四个方向的离散分布转换为四个连续距离
            # reg_max > 1 时使用 DFL；否则使用恒等映射
            self.dfl = DFL(c1=self.reg_max) if self.reg_max > 1 else nn.Identity())

这段代码主要完成三件事：保存检测头需要的基本属性、构建回归和分类两个分支，以及准备 DFL 模块。

代码里有几个点需要解释一下：

###### （1）`self.no` 为什么等于 `nc + 4 * reg_max`  ？

在当前实现中，检测头暂时不预测角度，因此每个特征图位置需要输出两部分数据：

*   回归分支：4 个方向，每个方向 16 个原始分数，共 64 个通道。
*   分类分支：`nc` 个类别原始分数。

因此每个位置的原始输出通道数 `self.no` 就等于 `nc + 4 * self.reg_max`。

如果当前任务使用 `DOTA` 数据集，这个数据集有 `15` 个类别，那么每个位置的原始输出通道数就是：`64 + 15 = 79`，这里的 `79` 是送入后续处理之前的原始输出通道数。它既不是最终边界框的参数数量，也不包含 OBB 的角度通道。

###### （2）为什么要使用 `nn.ModuleList` ？

回归分支和分类分支都采用了这样的结构：

```python
self.cv2 = nn.ModuleList([nn.Sequential(...) for c in ch ])
```

这里涉及 PyTorch 中两个不同的模块。`nn.Sequential` 我们之前大量使用过，它用于将多个网络层按顺序组合起来。输入特征会依次经过这三层，前一层的输出自动成为后一层的输入。 `nn.ModuleList` 则用于保存并注册多个子模块。假设 `ch = (64, 128, 256)`，`for c in ch`循环就会创建三套独立的回归分支保存在`nn.ModuleList`中，分别对应 P3、P4、P5 三个尺度。

为什么不直接使用普通的 Python 列表？

因为 `nn.ModuleList` 会将其中的网络层注册为当前模型的子模块。这样，PyTorch 才能正确追踪它们的参数，使其参与模型的参数统计、优化器更新、设备迁移以及模型权重的保存和加载。

如果只是把这些层放进普通 Python 列表，PyTorch 就不会自动将列表中的模块注册到模型中，容易造成参数没有被正确管理的问题。

这里还要注意，`nn.ModuleList` 负责保存和注册模块，但本身不会自动执行这些模块。后面我们仍然需要在 `forward` 方法中通过 `self.cv2[i](x[i])`，明确调用对应尺度的分支。

###### （3）`c2` 和 `c3` 是什么？

`c2` 和 `c3` 是回归分支和分类分支的中间通道数，用于平衡特征提炼能力与计算开销。

    c2 = max(16, ch[0] // 4, self.reg_max * 4)
    c3 = max(ch[0], min(self.nc, 100))

其中，`ch[0]` 表示第一个输入尺度的通道数，通常对应较高分辨率的 P3 特征图。 `c2` 综合考虑最低通道数、输入通道数和回归输出所需的通道规模。`c3` 根据输入通道数和类别数量确定中间宽度，并对类别数量设置上限，避免类别数较大时中间层过宽。

这些数值参考了Ultralytic中的设置，是一个经验值，不是所有检测头都必须使用的固定公式。

###### （4）为什么要用 `nn.Identity()`？

这是一个值得说明的小设计。`nn.Identity()` 是 PyTorch 提供的恒等映射模块。它不进行任何计算，只把输入原样返回。例如：

    x = torch.tensor([1.0, 2.0, 3.0])
    y = nn.Identity()(x)

    # y 与 x 的数值完全相同

使用 `nn.Identity()` 后，无论当前配置是否需要 DFL，后续代码都可以统一写成：`distance = self.dfl(box)`，当 `reg_max=16` 时，执行真正的 DFL 积分；当 `reg_max=1` 时，直接返回输入，不需要再写额外的条件判断。

这就是恒等映射在网络模块设计中的一个实用之处：**在不需要某项处理时，仍然可以保持统一的调用接口。**

当前使用 `reg_max=16`，因此实际执行的是 DFL 分支。

##### 7.5.2.2  `forward()`：前向传播方法

完成初始化后，接下来编写 `forward()`方法。`forward()`方法让输入的各尺度特征图分别经过回归分支和分类分支，并组织好输出。

在 `Detect` 类中继续添加：

        def forward(self, x):
            # 保存各尺度的预测结果
            outputs = []

            # 每个尺度分别进行框回归和类别预测
            for i in range(self.nl):
                box = self.cv2[i](x[i])  # 当前尺度的回归原始分数
                cls = self.cv3[i](x[i])  # 当前尺度的类别原始分数

                # 沿通道维拼接回归分数和类别分数
                outputs.append(torch.cat((box, cls), dim=1))

            # 训练阶段：保留各尺度的原始预测结果
            if self.training:
                return outputs

            # 推理阶段：进一步解码，得到具体的框坐标和类别概率
            return self._inference(outputs)

这里的 `x` 不是原始输入的图片张量，而是 Backbone 和 Neck 输出的多尺度特征图列表。例如：

```text
x[0]：P3 特征图，形状为 (B, 64, 80, 80)
x[1]：P4 特征图，形状为 (B, 128, 40, 40)
x[2]：P5 特征图，形状为 (B, 256, 20, 20)
```

其中，`B` 是批次大小。

###### （1）理解分支调用：从 `ModuleList` 到前向传播

先回顾一下初始化时的分支构建代码：

    self.cv2 = nn.ModuleList([
        nn.Sequential(
            Conv(c, c2, 3),
            Conv(c2, c2, 3),
            nn.Conv2d(c2, 4 * self.reg_max, 1)
        ) for c in ch])

这里的 `self.cv2` 是一个 `nn.ModuleList`，里面保存的不是单独的卷积层，而是多个 `nn.Sequential` 模块。每个 `nn.Sequential` 都包含2个 `Conv` 和 1个 `nn.Conv2d` 共三层网络，它们按照定义的顺序连接起来，组成一套完整的回归分支。

假设 `ch = (64, 128, 256)`，那么 `self.cv2` 中就有三套回归分支，分别对应 P3、P4、P5 三个尺度。分类分支 `self.cv3` 也是同样的组织方式。

接下来，看前向传播中的核心循环：

```python
for i in range(self.nl):
    box = self.cv2[i](x[i])
    cls = self.cv3[i](x[i])
```

其中，`i` 从 `0` 开始，依次取 `0、1、2`。每次循环都完成一个尺度的预测。以第1次循环为例：

`box = self.cv2[0](x[0])` 这行代码可以分成两步理解。

**第一步：取出对应的回归分支。** `self.cv2[0]` 从 `ModuleList` 中取出第一个 `nn.Sequential`，也就是专门处理 P3 特征图的回归分支。

**第二步：将特征图输入这个分支。** 括号中的 `x[0]` 是 P3 特征图。由于 `nn.Sequential` 是一个可调用的 PyTorch 模块，写成 `self.cv2[0](x[0])` 就会执行它的前向传播。输入特征图 `x[0]` 依次经过内部的三层网络，输出形状为 `(B, 64, 80, 80)`的回归预测结果。这里的 `64` 个输出通道对应四个方向各自的 `16` 个原始分数。

同理，`cls = self.cv3[0](x[0])`会将同一张 P3 特征图 `x[0]` 输入第一个分类分支，经过它内部的三层网络，得到形状为 `(B, nc, 80, 80)` 的类别预测结果。

第二次循环时，`i=1`，代码就会使用 `self.cv2[1]` 和 `self.cv3[1]` 处理 `x[1]`，也就是 P4 特征图。第三次循环则对应 P5。

总结一下，这段代码的核心逻辑是：**通过索引** `i` **，从** `ModuleList` **中取出对应尺度的** `nn.Sequential` **分支，再将同一尺度的特征图输入该分支，执行完整的前向传播得到预测结果。**

需要注意，`ModuleList` 只是负责保存和管理这些分支，并不会自动执行它们。真正触发前向传播的是 `self.cv2[i](x[i])` 这样的调用。

再看 ` outputs.append(torch.cat((box, cls), dim=1))`，因为两个分支的输出空间尺寸相同，所以可以沿着通道维拼接 `torch.cat((box, cls), dim=1)`，例如，P3 的回归分支输出形状为 `(B, 64, 80, 80)`，分类分支输出形状为 `(B, 15, 80, 80)`，拼接后就是：`(B, 79, 80, 80)`，每个空间位置都对应一组回归原始分数和类别原始分数。

###### （2）为什么训练和推理要分开处理？

训练和推理要分开处理，是因为两者需要的结果不同。

训练时，后续损失函数需要使用各尺度的原始预测结果，因此这里直接返回 `outputs`，不提前执行 DFL 和边界框解码。

推理时，我们则希望得到实际的目标框坐标和类别概率，所以还需要调用 `_inference()` 方法进行解码。`_inference()` 方法是我们自定义的解码方法，稍后实现。

这里的 `self.training` 是 PyTorch 模块的训练状态标记，可以通过 `model.train()` 和 `model.eval()` 切换。需要注意，`model.eval()` 只切换模块的行为，并不自动关闭梯度计算；实际推理时通常还会配合 `torch.no_grad()` 使用。

接下来，我们实现推理阶段的解码方法 `_inference()`。

##### 3. 为解码过程预留接口

在编写 `_inference()` 之前，先解决一个设计问题。

当前的 `Detect` 检测头使用 `dist2bbox()` 将四个距离解码为水平框。但后面实现 OBB 时，需要通过 `dist2rbox()` 解码为旋转框。如果直接把 `dist2bbox()` 写死在推理代码中，那么构建 `OBB` 检测头时就需要修改或复制整段推理流程。

为了避免这种情况，我们先在 `Detect` 中定义一个独立的解码方法：

        def decode_bboxes(self, bboxes, anchors, xywh=True):
            """
            """将 DFL 输出的四个距离解码为水平框 (cx, cy, w, h)。
            
            bboxes：DFL 输出的四个距离，形状为 (B, 4, A)
            anchors：锚点坐标，形状为 (1, 2, A)
            xywh：True 返回中心点和宽高；False 返回两个角点
            """
            return dist2bbox(bboxes, anchors, xywh=xywh, dim=1)

这里的 `bboxes` 实际上是 DFL 输出的四个距离，可以将它理解为待解码的距离张量。`dim=1` 是因为当前推理流程中的距离张量形状为 `(B, 4, A)`，四个距离位于第 1 维，也就是通道维。

这个方法目前只负责调用 `dist2bbox()`，看起来似乎多绕了一步，但它为后续继承留下了接口：基础类使用水平框解码，子类可以重写同一个方法，使用不同的解码方式。

下面编写的 `_inference()` 就会调用这个接口，而不直接指定具体的解码函数。

##### 4. 编写`_inference()`解码函数：从原始分数得到水平框

继续在 `Detect` 类中添加：

        def _inference(self, x):
            # 记录第一个尺度的输出形状，用于判断是否需要更新锚点缓存
            shape = x[0].shape

            # 将各尺度的空间位置展平，并沿锚点维拼接
            # 每个尺度：(B, no, H, W) -> (B, no, H*W)， 拼接后：  (B, no, A)，A 为所有尺度的锚点总数
            x_cat = torch.cat([ xi.reshape(shape[0], self.no, -1) for xi in x], dim=2)

            # 特征图形状变化时重新生成锚点和对应的步长
            if self.dynamic or self.shape != shape:
                anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)

                # (A, 2) -> (2, A)，方便与 (B, 2, A) 的张量运算
                self.anchors = anchor_points.transpose(0, 1)

                # (A, 1) -> (1, A)，保存每个锚点对应的步长
                self.strides = stride_tensor.transpose(0, 1)

                # 记录当前形状，下次推理时可以复用缓存
                self.shape = shape

            # 拆分回归分数和类别分数，box: (B, 4*reg_max, A)， cls: (B, nc, A)
            box, cls = x_cat.split( (4 * self.reg_max, self.nc), dim=1)

            # 回归原始分数 -> 四个连续距离 -> 水平框
            # DFL 输出形状：(B, 4, A)
            # 解码后得到：(B, 4, A)
            # 再乘以步长，将特征图坐标转换为原图像素坐标
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

            # 类别原始分数经过 sigmoid 转换为概率
            # 最终输出：(B, 4 + nc, A)
            return torch.cat((dbox, cls.sigmoid()), dim=1)

这段代码比前面的 `forward()` 长一些，可以分成四个步骤来理解。

**第一步：将多个尺度的预测结果整理成统一的锚点序列。**

前面的 `outputs` 中，每个尺度的数据都保留着二维空间结构。例如，P3 的形状为 `(B, 79, 80, 80)`。推理时，我们不需要按二维网格逐行处理，而是把每个空间位置看作一个锚点，将 `80×80` 个位置展开成一个长度为 `6400` 的序列，代码是 `xi.reshape(shape[0], self.no, -1)`，这里的 `-1` 表示让 PyTorch 自动计算这一维的长度。对于 P3，它就是 `80×80=6400`。随后，`torch.cat(..., dim=2)`，沿锚点所在的维度拼接 P3、P4、P5 的预测结果。这样，不同尺度的预测就被整理成了统一的序列，便于后续批量解码。

**第二步：生成并缓存锚点。**

`make_anchors()` 根据各尺度的特征图尺寸，生成每个格子的中心坐标，同时记录对应的 `stride`。

为什么要缓存？因为在连续推理时，如果输入尺寸和特征图形状没有变化，各尺度的锚点位置也不会变化。重复生成这些坐标没有必要，因此可以保存起来，下次直接复用。

*   `self.dynamic=True`：每次都重新生成锚点。
*   `self.shape != shape`：当前形状与上一次不同，需要重新生成。
*   其他情况：继续使用缓存。

这里的 `self.stride` 需要根据完整模型设置为对应的值，例如 `[8, 16, 32]`。

**第三步：将回归原始分数转换为边界框。**

拼接后的数据中，前 `4×16=64` 个通道属于回归分支，剩余 `nc` 个通道属于分类分支。`split()` 沿通道维将它们重新分开。回归部分接着经过 `self.dfl(box)`，DFL 对四组离散分数分别计算数学期望，得到 `left`、`top`、`right`、`bottom` 四个连续距离。

然后调用 `self.decode_bboxes()` 方法，将距离和锚点交给解码方法，计算水平框的中心坐标和宽高。

为什么还要乘以 `self.strides`？ 因为锚点和 DFL 输出的距离都处于特征图坐标尺度。例如，在步长为 8 的特征图上，1 个格子的距离对应原图中的 8 个像素。因此，乘以对应的步长，才能把解码结果转换到原图像素坐标。

**第四步：输出边界框和类别概率。**

回归分支得到的是 4 个框参数，分类分支经过 `sigmoid` 后得到 `nc` 个类别概率。两部分沿通道维拼接，形成最终的水平框推理输出。

### 5. 添加偏置初始化

检测头的最后一层除了卷积权重，还有偏置参数。需要对回归分支和分类分支的偏置进行初始化。

继续在 `Detect` 类中添加以下代码：

        def bias_init(self):
            """初始化回归分支和分类分支最后一层的偏置。"""

            # 初始化时不需要计算梯度
            with torch.no_grad():
                for box_branch, cls_branch, stride in zip(
                    self.cv2,
                    self.cv3,
                    self.stride
                ):
                    # 回归分支最后一层的偏置初始化为 1
                    box_branch[-1].bias.fill_(1.0)

                    # 分类分支参考了Ultralytic中的初始设置策略
                    cls_branch[-1].bias[:self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

这里的 `box_branch[-1]` 和 `cls_branch[-1]` 表示对应 `nn.Sequential` 中的最后一层，也就是输出原始预测分数的 `1×1` 卷积。

为什么要初始化偏置？网络刚开始训练时，参数通常还没有形成有效的预测能力。合理的偏置初始化可以为训练提供一个初始状态，尤其是类别分支，可以通过先验设置控制初始类别分数的水平。

需要注意两点。

第一，回归偏置设置为 `1.0`，并不意味着经过 DFL 后的距离就一定等于 1。最终距离由一整组原始分数经过 `softmax` 和数学期望计算后得到。

第二，分类偏置公式中的 `640` 对应输入尺寸，`stride` 则是当前尺度的实际步长。调用 `bias_init()` 之前，必须先设置正确的 `self.stride`，不能直接使用初始化时的全零占位值。

这套初始化策略参考了Ultralytic的实现，需要说明的是，并不是所有检测头都必须采用相同的偏置设置。

## 7.5.3 派生 OBB 旋转框检测头

到这里， `Detect` 检测头已经具备回归、分类、DFL 和水平框推理解码能力。

现在可以通过继承的方式实现 `OBB` 检测头了。具体思路就是：复用 `Detect` 检测头已经完成的功能，在此基础上增加角度分支，并替换框解码方法。

### 1. 增加角度预测分支

在 `Detect` 类后面添加 `OBB` 类：

    class OBB(Detect):
        """旋转框检测头：继承 Detect，并增加角度预测。"""

        def __init__(self, nc=80, ne=1, ch=()):
            # 初始化父类的回归分支、分类分支和 DFL
            super().__init__(nc=nc, ch=ch)

            # 额外预测的通道数，当前为 1 个角度
            self.ne = ne

            # 角度分支的中间通道数
            c4 = max(ch[0] // 4, self.ne)

            # 每个尺度各有一套独立的角度预测分支
            self.cv4 = nn.ModuleList([
                nn.Sequential(
                    Conv(c, c4, 3),             # 第 1 层：3×3 卷积
                    Conv(c4, c4, 3),            # 第 2 层：3×3 卷积
                    nn.Conv2d(c4, self.ne, 1)   # 第 3 层：输出角度原始分数
                ) for c in ch])

这里最关键的是 `super().__init__(nc=nc, ch=ch)`，它调用父类 `Detect` 的初始化方法。因此，`OBB`检测头不需要再重复定义 `cv2`、`cv3`、`self.dfl` 等内容，而是在父类已有的结构上增加 `cv4`即可。

角度分支的结构与前面的两个分支类似，但最后一层只输出 `ne=1` 个通道，因为只需要预测一个角度值。

### 2. 编写 OBB 的前向传播

角度分支同样需要对 P3、P4、P5 分别进行预测，再将各尺度的结果展平并拼接。

继续在 `OBB` 类中添加：

        def forward(self, x):
            # 批次大小
            bs = x[0].shape[0]

            # 各尺度分别预测角度，并将空间位置展平
            # 每个尺度：(B, ne, H, W) -> (B, ne, H*W)
            # 拼接后：(B, ne, A)
            angle = torch.cat(
                [
                    self.cv4[i](x[i]).reshape(bs, self.ne, -1)
                    for i in range(self.nl)
                ], dim=2 )

            # 将原始分数映射到约定的旋转角度范围
            angle = (angle.sigmoid() - 0.25) * math.pi

            # 推理时保存角度，供旋转框解码方法使用
            if not self.training:
                self.angle = angle

            # 父类负责回归分支、分类分支及其后续处理
            outputs = super().forward(x)

            # 训练时：原始预测和角度分别返回
            if self.training:
                return outputs, angle

            # 推理时：将角度追加到父类的输出中
            return torch.cat((outputs, angle), dim=1)

这里有两个需要特别说明的地方。

**第一，为什么在调用父类的** `forward() `**之前先计算角度？**

简单的说就是父类的 `forward()` 方法要用到这个角度，所以必须先算出来。

在推理阶段，父类的 `forward()` 会继续调用 `_inference()`，而 `_inference()` 又会调用 `self.decode_bboxes()`。后面我们将重写这个方法，他会使用 `self.angle` 进行旋转框解码。因此，必须先计算角度并保存，父类进入推理解码流程时，才能使用正确的角度。

**第二，为什么训练时返回两个结果，推理时却返回一个张量？**

训练时，父类会返回各尺度的原始预测结果，角度作为独立结果返回，供后续损失计算使用，所以返回代码是：`return outputs, angle`。推理时，父类已经完成了水平框解码流程中的公共部分，返回框参数和类别概率。OBB检测头再将角度追加到输出张量中即可。按照当前代码，最终通道顺序是 `(cx, cy, w, h, 类别分数..., angle)`，也就是说，角度已经被融合进最后一个通道了。

##### 3. 重写解码方法：从水平框变成旋转框

前面在 `Detect` 中定义的 `decode_bboxes()` 默认调用 `dist2bbox()`。现在需要在 `OBB` 中重写这个方法：

        def decode_bboxes(self, bboxes, anchors):
            """结合角度，将四个距离解码为旋转框中心和宽高。"""

            return dist2rbox(
                bboxes,
                self.angle,   # 当前所有锚点对应的预测角度
                anchors,
                dim=1)

为什么这样就可以改变解码方式？这个其实是面向对象编程中的知识了。

父类的 `_inference()` 调用的是`self.decode_bboxes(...)`，而不是直接调用`dist2bbox(...)`，
当实际对象是 `Detect` 时，调用的是父类的 `decode_bboxes()`，使用 `dist2bbox()` 解码水平框。当实际对象是 `OBB` 时，Python 会调用子类重写后的 `decode_bboxes()`，使用 `dist2rbox()` 解码旋转框。

这就是方法重写。父类负责实现通用的推理流程，子类只需替换有区别的部分，而不必复制整个`_inference()`。

还有一个细节值得注意：`dist2rbox()` 返回的是 `(cx, cy, w, h)`，不包含角度本身。角度由 `cv4` 单独预测，随后才被追加到最终输出中。

## 7.5.4 在 `main.py` 中验证检测头

到这里，`Detect` 和 `OBB` 的主要代码已经完成。别忘了在 `nn\__init__.py` 中增加对这两个模块的导出。

最后需要检查两件事：训练模式下是否能返回预期的原始预测结果，以及推理模式下是否能得到正确形状的输出。这些验证代码统一放在 `ch07/main.py` 中。通过独立的验证函数分别检查两个检测头，再由 `main()` 统一调用。

###### 1. 准备虚拟特征图

在 `main.py` 中添加：

    import torch

    from head import Detect, OBB


    def make_demo_feats():
        """构造与三个检测尺度相对应的虚拟特征图。"""

        return [
            torch.randn(1, 64, 80, 80),   # P3，stride=8
            torch.randn(1, 128, 40, 40),  # P4，stride=16
            torch.randn(1, 256, 20, 20),  # P5，stride=32
        ]

这里模拟的是输入尺寸为 `640×640` 时的三个特征图尺度。由于当前只验证检测头，所以不需要真正运行 Backbone 和 Neck，直接构造符合要求的特征图即可。

###### 2. 验证 `Detect` 检测头

继续添加独立的验证函数：

    def validate_detect():
        """验证 Detect 的训练输出和推理输出形状。"""

        head = Detect(nc=15, ch=(64, 128, 256))

        # 独立测试时手动设置三个尺度的步长
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        head.bias_init()  # stride 已经设置，现可初始化回归与分类输出层的偏置

        feats = make_demo_feats()

        # ---------- 检查训练模式 ----------
        head.train()
        raw_outputs = head(feats)

        expected_shapes = [
            (1, 79, 80, 80),
            (1, 79, 40, 40),
            (1, 79, 20, 20),
        ]

        assert len(raw_outputs) == 3
        assert [tuple(t.shape) for t in raw_outputs] == expected_shapes

        print("Detect 训练输出：")
        for i, out in enumerate(raw_outputs):
            print(f"  P{i + 3}: {tuple(out.shape)}")

        # ---------- 检查推理模式 ----------
        head.eval()

        with torch.no_grad():
            pred = head(feats)

        assert tuple(pred.shape) == (1, 19, 8400)
        assert torch.isfinite(pred).all()

        print("Detect 推理输出：", tuple(pred.shape))

这里检查了两种不同的输出。训练模式下，三个尺度分别输出 79 个通道，即 64 个回归原始分数和 15 个类别原始分数。

推理模式下，所有尺度的空间位置被合并为 8400 个候选位置，输出变成 `(1, 19, 8400)`，对应 4 个水平框参数和 15 个类别概率。

###### 3. 验证旋转框检测头 `OBB`

在 `main.py` 中继续添加：

    def validate_obb():
        """验证 OBB 的训练输出和推理输出形状。"""

        head = OBB(nc=15, ch=(64, 128, 256))
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        head.bias_init()  # 调用继承的初始化方法，初始化回归与分类分支

        feats = make_demo_feats()

        # ---------- 检查训练模式 ----------
        head.train()
        raw_outputs, angle = head(feats)

        # 父类返回三个尺度的原始预测
        assert len(raw_outputs) == 3
        assert [tuple(t.shape) for t in raw_outputs] == [
            (1, 79, 80, 80),
            (1, 79, 40, 40),
            (1, 79, 20, 20),
        ]

        # 角度分支将三个尺度的空间位置拼接在一起
        assert tuple(angle.shape) == (1, 1, 8400)

        print("OBB 训练输出：")
        print("  各尺度原始预测：", [tuple(t.shape) for t in raw_outputs])
        print("  角度预测：", tuple(angle.shape))

        # ---------- 检查推理模式 ----------
        head.eval()

        with torch.no_grad():
            pred = head(feats)

        assert tuple(pred.shape) == (1, 20, 8400)
        assert torch.isfinite(pred).all()

        print("OBB 推理输出：", tuple(pred.shape))

`OBB` 的训练输出与 `Detect` 不完全相同。除了三个尺度的原始预测之外，它还单独返回形状为 `(1, 1, 8400)` 的角度预测。推理时，输出形状为 `(1, 20, 8400)`，其中 20 个通道依次为 4 个框参数、15 个类别概率和 1 个角度。

###### 4. 在 `main()` 中统一运行

最后，在 `main.py` 文件末尾添加：

    def main():
        validate_detect()
        validate_obb()


    if __name__ == "__main__":
        main()

在 `ch07/` 目录下运行：`python main.py`，如果断言均通过，说明两个检测头的训练输出组织和推理解码链路已经满足当前的形状约定。

到这里，我们完成了从基础 `Detect` 到旋转框 `OBB` 的主要实现：父类负责复用回归、分类和推理流程，子类增加角度预测，并通过重写解码接口实现旋转框解码。

下一步是将它接入完整模型，敬请期待。
