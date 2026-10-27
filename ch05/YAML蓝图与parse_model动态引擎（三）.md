# YAML蓝图与parse_model动态引擎（三）

## 5.3 壳网络点火 ── 模型封装与动态验算

上一节，我们已经完成了 `parse_model()` 引擎。

YAML 中的一行行配置，终于可以被解析成真正的 PyTorch 模块。

但现在还差最后一步。

`parse_model()` 只负责**把模型的各层创建出来**，它并不知道这些层在前向传播时应该怎样按照 `from` 连接起来。尤其是 FPN 中还存在跨层连接，不能简单地从第 0 层一路顺序执行到最后一层。

所以这一节要完成两件事：

1. 用一个模型类把 YAML、`parse_model()` 和动态 `forward()` 封装起来；
2. 用虚拟输入验证：第 5 卷通过 YAML 构建出来的模型，是否仍然与第 4 卷的 `YOLOLikeDetector` 保持一致。


### 5.3.1 `nn.ModuleList` 与 `nn.Sequential`

在正式改造模型类之前，先厘清两个容易混淆的概念。

我们之前就已经知道，`nn.Sequential` 会按照顺序执行所有子模块，前一个模块的输出会自动传给下一个模块。

```python
seq = nn.Sequential(
    Conv(3, 16, 3, 2),
    Conv(16, 32, 3, 2)
)

out = seq(x)  # 自动: x → Conv1 → Conv2 → out
```


这种方式非常简洁，但如果网络中出现跳接、分支、`Concat` 等复杂连接，仅靠 `nn.Sequential` 的顺序处理就不够了。

相比 `nn.Sequential` ，`nn.ModuleList` 更像一个“模块容器”，它负责保存子模块，但不会自动规定这些模块应该怎样执行。

```python
mods = nn.ModuleList([
    Conv(3, 16, 3, 2),
    Conv(16, 32, 3, 2)
])

# 通过一个循环手动遍历:
for m in mods:
    x = m(x)
```

因此，`nn.ModuleList` 在执行方式上更加自由，可以由我们自己决定每一层的输入来自哪里。

那刚刚我们手搓的 `parse_model()` 引擎用了哪个？看上一节最后的返回值：

```python
return nn.Sequential(*layers), sorted(set(save))
```

也就是说，我们仍然使用 `nn.Sequential` 保存所有网络层。但是要注意：**这里主要把 `nn.Sequential` 当作一个按顺序保存网络层的容器，而不是直接依赖它的自动前向传播。**

后面真正执行模型时，我们会自己遍历模型中的每一层，再根据每一层记录的 `m.f` 决定输入从哪里来。

这也是为什么即使模型中存在 FPN 的跨层连接，仍然可以使用 `nn.Sequential` 来保存这些模块。

---

### 5.3.2 从 `YOLOLikeDetector` 到 `DetectionModel`

在第 4 卷中，我们的检测类是 `class YOLOLikeDetector(nn.Module)`，当时模型结构全部直接写在 `__init__()` 和 `forward()` 中，前向传播时也要手工维护。这是第 4 卷故意采用的写法，因为只有亲手把模型一层层搭出来，才能真正理解网络结构。

到了第 5 卷，情况发生了变化。现在模型结构已经写进 YAML 文档，每一层如何创建交给 parse_model()引擎，层与层之间如何连接由 from 决定，因此，`YOLOLikeDetector` 已经不需要再把具体网络结构写死在类里面了。

所以从这一卷开始，我们把这个类改名为：`DetectionModel`。`DetectionModel` 这个命名也更接近后面要对标的 Ultralytics 模型组织方式。

原本 `YOLOLikeDetector` 类的代码可以删除，因为我们要写一个新的类了。

请在 `model.py` 中继续加入下面的模型类。

如果文件顶部还没有导入 `deepcopy`，先补上：

```python
from copy import deepcopy
```

然后输入：

```python
class DetectionModel(nn.Module):
    
    def __init__(self, cfg="cfg/yolo-obb-mini.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        
        # 加载 YAML 配置
        self.yaml = (
                cfg
                if isinstance(cfg, dict)
                else yaml_model_load(cfg))
            
        # 输入通道数
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        
        # 允许覆盖类别数
        if nc is not None and nc != self.yaml["nc"]:
            print( f"覆盖 YAML nc={self.yaml['nc']} → nc={nc}")
            self.yaml["nc"] = nc
        
        # 解析 YAML， 构建模型
        self.model, self.save = parse_model(
                deepcopy(self.yaml),
                ch=ch,
                verbose=verbose)
        
        # 构建名称字典
        self.names = {
                i: f"{i}"
                for i in range(self.yaml["nc"])}

        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"\n模型总参数量: {total_params:,}")
            print(f"需要缓存的层: {self.save}")
```

这段代码不长，做的事情也比较清楚。

###### 1、读取 YAML

```python
self.yaml = (
    cfg
    if isinstance(cfg, dict)
    else yaml_model_load(cfg))
```

如果传入的 `cfg` 本身已经是一个 Python 字典，就直接使用。如果传入的是 `cfg/yolo-obb-mini.yaml` 这样的文件路径，就调用前面写好的 `yaml_model_load()` 把 YAML 文件读成字典。

###### 2、确定输入通道数

```python
ch = self.yaml["ch"] = self.yaml.get("ch", ch)
```

优先读取 YAML 中的 `ch` 参数，如果 YAML 中没有写，就使用构造函数传入的参数。对于 RGB 图像来说，输入通道数就是 3 .


###### 3、必要时覆盖类别数

```python
if nc is not None and nc != self.yaml["nc"]:
    print(f"覆盖 YAML nc={self.yaml['nc']} → nc={nc}")
    self.yaml["nc"] = nc
```

通常类别数直接由 YAML 中的 `nc` 参数决定。但如果创建模型时显式传入 `nc` 参数，就允许使用传入的 `nc` 覆盖 YAML 中的类别数。

###### 4、调用 `parse_model()`

真正的核心是：

```python
self.model, self.save = parse_model(
    deepcopy(self.yam),
    ch=ch,
    verbose=verbose)
```

`parse_model()` 会根据 YAML 一层一层创建网络，并返回两个结果：`self.model` 和 `self.save` ，上一节我们已经详细的讲过了。

这里使用 `deepcopy(self.yaml)`复制一个yaml，是为了给解析器一份独立的配置副本。

当前的 `parse_model()` 引擎已经尽量避免直接修改原始 YAML 数据，但保留一份独立副本仍然是一个比较稳妥的做法：以后解析器继续扩展时，即使内部需要调整某些配置，也不会影响 `self.yaml` 中保存的原始模型配置。

###### 5、保存类别名称

```python
self.names = {
    i: f"{i}"
    for i in range(self.yaml["nc"])}
```

当前还没有真正读取某个数据集的类别名称，所以这里先简单生成：

```python
{
    0: "0",
    1: "1"
}
```

后面使用真实数据集时再替换成真正的类别名称。

---

### 5.3.4 动态 `forward()`：让 `from` 真正运行起来

到现在为止，`parse_model()` 已经根据 YAML 把所有网络层创建出来了。

但模型的 `forward()` 不能直接写成 `return self.model(x)`。

因为这会让 `nn.Sequential` 简单地从头执行到尾，无法处理 `[[-1, 6], 1, Concat, [1]]` 这样的跨层连接。因此，我们还需要自己控制每一层的输入来源，让 YAML 中的 `from` 真正参与前向传播。

请在 `DetectionModel` 类中继续加入 `forward`方法：

```python
class DetectionModel(nn.Module):
    
    def __init__(self, cfg="cfg/yolo-obb-mini.yaml", ch=3, nc=None, verbose=True):
       ……
            
    def forward(self, x):        
        y = [] # 各层输出缓存
        for m in self.model:
            if m.f != -1:
                # 非线性连接: 从缓存中取输入
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f ]
            x = m(x)  # 执行当前层
            y.append(x if m.i in self.save else None) # 按需缓存
        return x
```

代码不长，但它**把 YAML 中静态** `from` **，变成模型运行时真正的数据流。**

下面逐步来看。

###### 1. 准备中间结果缓存

首先创建 `y = []`。

`y` 是一个中间结果缓存列表，用来保存各层前向计算得到的输出。对于后面还会再次使用的层，`y` 中保存它的输出张量；对于不需要再次使用的层，对应位置保存 `None`。因此 `y` 的索引始终和模型层号保持一致，例如 `y[6]` 表示第 6 层的输出缓存。

这样既可以通过层号找到需要的中间结果，又不需要在缓存列表中一直保留所有层的输出张量。

###### 2. 逐层执行模型

接下来：

```
for m in self.model:
```

依次遍历 `parse_model()` 创建出来的所有网络层。

前面在构建模型时，我们已经为每个模块记录了 `m.f`， `m.f` 就是这一层在 YAML 中对应的 `from`。

首先判断：

```
if m.f != -1:
```

如果 `m.f = -1`，表示当前层直接使用上一层输出。此时变量 `x` 本来就保存着上一层的结果，因此不用做任何额外处理，直接继续执行当前模块即可。

如果 `m.f != -1`，则说明当前层的输入不能简单地直接使用上一层，需要按照 `from` 指定的位置重新取得输入。

核心代码就是：

```python
x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f ]
```

这里分成两种情况。

如果 `m.f` 是一个整数，例如 `m.f = 6`，就直接执行 `x = y[6]`，取得之前缓存的第 6 层输出。

如果 `m.f` 是一个列表，例如 `m.f = [-1, 6]`，说明当前层需要多个输入。

此时列表中的每个位置分别处理：

-   `-1`：使用当前的 `x`，也就是上一层输出；
-   `6`：使用 `y[6]`，也就是之前缓存的第 6 层输出。

例如，假设上一层的输出是一个形状为 `[B, 64, 16, 16]` 的张量，而 `y[6]` 保存的是一个 `[B, 128, 16, 16]` 的张量，那么处理完成以后，`x` 会暂时变成：

```text
[
   [B, 64, 16, 16]    #上一层输出张量，对应m.f中的-1
   [B, 128, 16, 16]   #第6层输出张量，对应m.f中的6    
]
```

注意，这个列表中保存的不是层号 `-1` 和 `6`，而是根据这些层号找到的**真实特征张量**。

如果当前模块是 `Concat`，准备好输入 `x` 以后，再执行 `x = m(x)`，这个张量列表就会被直接传入`Concat`，完成特征拼接，也就是完成当前模块真正的前向计算。

###### 3. 按需保存当前层输出

当前层计算完成以后：

```python
y.append(x if m.i in self.save else None)
```

判断当前层的层号 `m.i` 是否位于 `self.save` 中。

如果在，说明后面的某一层还会通过 `from` 再次引用当前层，因此把当前输出保存到 `y`。

如果不在，就只在对应位置放入 `None`，不再额外保留这个输出的引用。当这个张量没有其他地方继续引用时，就可以被释放。

这里可以把 `m.f` 和 `self.save` 的关系简单理解为：

```text
m.f         → 当前层应该从哪里取输入

self.save   → 哪些层的输出不能提前丢掉
```

两者正好相互配合。

这也解释了为什么前面要构建 `save`：如果第 6 层以后还会被 `from` 引用，它的输出就必须提前保存下来；所以，`save` 的具体内容会随着 YAML 的连接方式变化，它并不是写死在模型中的。

到这里，动态 `forward()` 的核心逻辑就已经完整了：

```text
遍历当前层
    ↓
查看 from
    ↓
决定从哪里取得输入
    ↓
执行当前模块
    ↓
根据 save 决定是否缓存输出
    ↓
继续下一层
```

这套逻辑并不依赖某一张固定的 YAML。以后即使网络结构发生变化，只要仍然通过 `from` 描述层之间的连接，`forward()` 本身通常不需要跟着重新改写。

需要注意的是，YAML 中出现的 `Conv`、`C3k2`、`SPPF`、`Concat`、`SimpleHead` 等自定义模块，都必须提前导入 `model.py`，这样 `globals()[m_name]` 才能找到对应的 Python 类。相关导入代码前面已经完成，这里不再重复。

到这里，YAML、`parse_model()` 和动态 `forward()` 三部分就真正连接起来了：

```text
YAML
→ 描述使用什么模块、输入来自哪里

parse_model()
→ 把模块名称变成真正的 PyTorch 模块

动态 forward()
→ 根据 from 让这些模块按照 YAML 的连接关系真正运行
```

这也意味着，我们已经不再需要像第 4 卷那样，手工维护 `p3`、`p4`、`p5`、`up5`、`f4`、`f3` 等大量中间变量了。

#### 5.3.5 运行并验证 YAML 动态模型

###### 1、新的模型创建方式

模型已经能够从 YAML 自动创建，动态 `forward()` 也已经完成。

接下来只剩最后一步：**真正运行一次，看看从 YAML 构建出来的模型是否和第 4 卷的模型保持一致。**


第 4 卷中的数据集、DataLoader 和 `SimpleTrainer` 都可以继续沿用，我们只需要修改模型的导入和创建方式。

首先，在`ch05`的 `main.py` 文件中，把原来的模型导入：

```python
from model import YOLOLikeDetector
```

改成：

```python
from model import DetectionModel
```

第 4 卷创建模型时写的是：

```python
model = YOLOLikeDetector(num_classes=2)
```

现在改成：

```python
model = DetectionModel(
    "cfg/yolo-obb-mini.yaml",
    nc=2,
    verbose=True)
```

这看起来只是换了一行代码，但背后的模型构建方式已经完全不同。

#### 2、检查模型是否正确构建

创建 `DetectionModel` 时，我们设置了 `verbose=True`，因此在模型创建过程中，`parse_model()` 会逐层打印解析结果。

终端中会看到类似的输出：

```
  0      -1    1    ...    Conv       [3, 16, 3, 2]
  1      -1    1    ...    Conv       [16, 32, 3, 2]
  2      -1    1    ...    C3k2      [32, 32, 1, False, 0.5]
  ...
```

打印信息主要包括：

```
层号
from
repeats
参数量
模块类型
最终构造参数
```

这里可以检查**解析出来的模块类型、输入来源和通道参数，是否与 YAML 中的配置一致。**

#### 2. 检查前向传播输出

确认模型结构正常构建以后，可以再加入一段虚拟输入测试模型的数据处理流程是否正常：

```python
dummy = torch.randn(1, 3, 256, 256, device=device)
outs = model(dummy)

for i, o in enumerate(outs):
    stride = 256 // o.shape[-1]
    print(f"输出{i}: {list(o.shape)}  (stride={stride})")
```

这里创建了一个形状为 `[1, 3, 256, 256]` 的随机张量来表示输入，其中 `1` 表示 batch size，`3` 表示 RGB 三通道，`256×256` 是输入图像大小。

如果 YAML 动态构建和 `forward()` 都没有问题，模型仍然应该得到三个尺度的输出：

```
输出0: [1, 10, 32, 32]  (stride=8)
输出1: [1, 10, 16, 16]  (stride=16)
输出2: [1, 10,  8,  8]  (stride=32)
```

这三个尺度应该和第 4 卷保持一致。

只要模型结构能够正确解析，三个输出张量也与第 4 卷一致，就说明从硬编码模型到YAML动态模型这次升级已经成功完成。

---

## 本章小结

###### 从硬编码到阵图驱动

| 维度 | 第4卷：硬编码 | 第5卷：YAML 驱动 |
|---|---|---|
| 模型定义 | 在 Python 中逐层写 `self.xxx = ...` | 在 YAML 中写 `[from, repeats, module, args]` |
| 模型创建 | `YOLOLikeDetector` 自己创建每一层 | `parse_model()` 自动创建 |
| 模型连接 | 在 `forward()` 中手工写 `p3/p4/p5/f3/f4` | 由 YAML 的 `from` 描述 |
| 前向传播 | 手工管理每个中间变量 | `m.f + y + save` 动态取输入 |
| 修改结构 | 修改 Python 模型代码 | 主要修改 YAML |

这一卷真正改变的是：

```text
模型怎么定义
模型怎么创建
模型怎么连接
```

###### 炼器笔记

> *炼器之道，始于手打，成于阵图。*  


第 4 卷中，我们亲手写下每一层。第 5 卷中，同样的模型第一次从 YAML 蓝图中自动生长出来。

网络没有凭空变强，但构建方式已经发生了变化：

```text
手工搭建
    ↓
配置描述
    ↓
自动解析
    ↓
动态执行
```

---

# 练功房：动手实验

## 实验 1：修改模型宽度

复制一份当前 YAML 作为实验文件，然后把：

```yaml
scales:
  n: [0.50, 0.25, 1024]
```

改成：

```yaml
scales:
  n: [0.50, 0.50, 1024]
```

重新运行模型，观察：

- 每层实际输出通道数有什么变化？
- 总参数量有什么变化？
- 三个输出尺度的空间尺寸是否仍然是 `32×32 / 16×16 / 8×8`？

## 实验 2：观察 `depth` 如何影响 repeats

当前YAML 中所有 `C3k2` 的 `repeats` 都是 `1`。

复制一份实验 YAML，把其中某个 `C3k2` 临时改成 `[-1, 4, C3k2, [128, False, 0.5]]`。

当 `n: [0.50, 0.25, 1024]` 时，`4 × 0.50 → 2`。

当改成 `n: [1.00, 0.25, 1024]` 时，`4 × 1.00 → 4`。

观察模型结构打印中的 `n` 是否随之变化。

## 实验 3：追踪 `save`

可以暂时在 `DetectionModel.forward()` 中加入一行打印：

```python
def forward(self, x):
    y = []

    for m in self.model:
        if m.f != -1:
            x = (
                y[m.f]
                if isinstance(m.f, int)
                else [
                    x if j == -1 else y[j]
                    for j in m.f
                ]
            )

        x = m(x)

        print(
            f"Layer {m.i:2d} | "
            f"f={str(m.f):12s} | "
            f"cached={m.i in self.save}"
        )

        y.append(
            x if m.i in self.save else None
        )

    return x
```

运行一次 `model(dummy)`，观察：

```text
第4层
第6层
第9层
第13层
第17层
```

是否被标记为缓存。

这能非常直观地帮助你理解：

```text
from
save
动态 forward
```

三者之间的关系。

## 实验 4：新增一个最简单的模块

尝试新增一个 `Identity` 模块，它什么都不改变，只把输入原样返回。

在 `nn/block.py` 中添加：

```python
class Identity(nn.Module):
    def forward(self, x):
        return x
```

然后：

1. 在 `nn/__init__.py` 中导出 `Identity`；
2. 在 `model.py` 顶部导入 `Identity`；
3. 在实验 YAML 中插入 `[-1, 1, Identity, []]`。

由于 `Identity` 不改变通道数，当前 `parse_model()` 的默认分支已经能够处理它。

运行模型，检查：

```text
YAML
    ↓
反射找到 Identity
    ↓
parse_model() 创建模块
    ↓
forward() 正常执行
```

这个实验能比较直接地体会“YAML + 反射”的意义。

---

## 下章预告

> *至此，我们的法器已经从“手工锻打”升级到了“阵图自动组装”。但法器再精妙，也需要真正的灵矿来淬炼。*

阵图已成，反射引擎开始运转，YAML 中的一行行文字已经能够真正化为网络结构。但目前送进模型的仍然只是我们人为制造的虚拟数据。

接下来要面对的，就是旋转目标检测真正的数据基础——DOTA8。

它的标注格式是什么样？四个顶点如何描述一个旋转框？数据又应该怎样送进我们刚刚搭好的模型？

下一卷，我们开始处理真正的训练数据。
