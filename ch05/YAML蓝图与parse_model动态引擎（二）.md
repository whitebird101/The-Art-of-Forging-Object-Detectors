# 阵图法则：YAML 蓝图与 `parse_model` 动态引擎（二）

## 5.2 斡旋造化 ── `parse_model` 反射引擎

阵图已经画好了，现在我们要造一台能读懂阵图、自动组装法宝的引擎。

这台引擎就是 `parse_model()` 函数。

### 5.2.1 反射：从字符串到类的跳变

在正式看 `parse_model()` 代码之前，我们先要理解一个关键的 Python “魔法”——反射（Reflection）。

反射机制是现代编程语言最重要的特点之一，有句很古早的话是这么说的：反射反射，程序员的快乐。

什么是反射？简单说：**在运行时，用一段文字（字符串）去查找并调用对应的函数或类。**

举个生活中的例子：你走进一家大型超市，手里拿着一张购物清单，上面写着“牛奶”两个字。你把清单交给店员，店员根据“牛奶”这个名字，在货架上找到真正的牛奶商品，递给你。

从“牛奶”两个字符，变成你手里真正的一瓶“牛奶”，这过程对应到编程领域，就是从一个字符串，变成与字符串对应的类或者函数，这就是语言的反射机制。

在 Python 中，这个过程有几种实现方式。

###### 方式一：`eval()` —— 直接执行字符串

```python
# eval("Conv") 就像对 Python 说："请你帮我找到名叫 Conv 的东西"
cls = eval("Conv")        # cls 现在就是 Conv 类本身
layer = cls(3, 16, 3, 2) # 等价于 Conv(3, 16, 3, 2)
```

`eval("Conv")` 在当前环境中搜索名为 `Conv` 的对象——如果之前 `from nn import Conv`，那 `Conv` 就存在于当前命名空间，`eval` 找到后返回 `Conv` 类本身（不是字符串，是真正的类对象）。然后你可以像调用普通类一样，用 `cls(...)` 创建类的实例。

###### 方式二：`globals()` —— 从全局命名空间字典查找

```python
# globals() 返回当前模块所有全局变量的字典
# 类似于 {"Conv": <class Conv>, "C3k2": <class C3k2>, ...}
cls = globals()["Conv"]
layer = cls(3, 16, 3, 2)
```

`globals()` 返回一个字典，键是变量名（字符串），值是对应的 Python 对象。用 `globals()["Conv"]` 从字典中取出 `Conv` 类，在这个简单例子中可以达到和 `eval("Conv")` 相同的查找效果，但更加安全——`eval()` 可以执行任意表达式，谁也不知道传递给eval的是一个什么样的字符串，会不会是恶意构造的，而 `globals()` 只是查字典，不会额外执行字符串中的代码，相对于来说，就堵死了一个可能的风险点。

###### 方式三：`getattr()` —— 从特定对象的属性中查找

```python
import torch.nn as nn

# getattr(nn, "Upsample") 从 torch.nn 模块中查找 Upsample 类
cls = getattr(nn, "Upsample")
layer = cls(scale_factor=2, mode="nearest")
```

`getattr(对象, "属性名")` 从指定对象的属性中按名称查找。当我们需要引用 PyTorch 内置模块（如 `nn.Upsample`）时，用 `getattr(nn, "Upsample")` 最合适。

我们的 `parse_model()` 综合使用了方式二和方式三：自定义模块（Conv、C3k2、SPPF、Concat、SimpleHead 等）用 globals() 查找；PyTorch 内置模块（nn.Upsample 等）用 getattr() 查找。

这就是反射的威力：YAML 里写的是字符串 `"Conv"`，通过反射变成真正的 `Conv` 类，然后传入参数就能创建王璐模块实例。

需要注意，这里反射解决的是：

> **“模块名称 → Python 类”这一层映射。**

不同模块的参数组织方式和通道计算方式并不完全相同，后面的 `parse_model()` 仍然需要针对不同模块进行相应处理。

**为什么用反射而不是 if-else？**

你可能会问：直接写 `if m_name == "Conv": m = Conv` 不行吗？当然可以，但想象一下——Ultralytics YOLO 支持很多模块（Conv、DWConv、C2f、C3k2、SPPF、Detect、OBB、Segment、Pose……），如果每加一种模块，就要在“根据名称找到类”这一步增加一个 `elif`，代码会变成这样：

```python
if m_name == "Conv":
    m = Conv
elif m_name == "C2f":
    m = C2f
elif m_name == "C3k2":
    m = C3k2
elif m_name == "SPPF":
    m = SPPF
elif m_name == "Detect":
    m = Detect
elif m_name == "OBB":
    m = OBB
# ... 继续增加 elif ...
```

而用 `globals()` 查找自定义模块，只需要：

```python
m = globals()[m_name]
```

新增模块时，只需在文件顶部 import 一下，globals() 就自动包含它了。零修改、零维护。而使用if-else的话，新增一个模块，除了增加import外，还要修改一遍if判断代码，显然更加繁琐也更容易出错。

---

### 5.2.2 PyYAML库：YAML处理

反射机制解决了“字符串如何变成类”的问题。但在编写 `parse_model()` 之前，还有一个很自然的问题：我们上一节写好的 `yolo-obb-mini.yaml`，Python 是怎么把它读进来的？

这里需要使用一个新的第三方库 `PyYAML`，如果你的环境里没有安装的话，需要事先安装一下。

在代码中增加导入代码 `import yaml`，然后使用 `PyYAML` 库提供的 `yaml.safe_load()` 方法，把 YAML 文本转换为 Python 中的字典、列表、数字、字符串等对象。

例如 YAML：

```yaml
nc: 2
ch: 3
```

经过 `d = yaml.safe_load(f)` 加载后，会得到：

```python
{
    "nc": 2,
    "ch": 3
}
```

这里使用 `safe_load()`，而不是直接使用 `load()`，是因为 `safe_load()` 只解析常见的 YAML 基础数据类型，不会随意创建 YAML 中指定的 Python 对象，更适合读取模型配置文件。

到这里，两项准备工作就清楚了： `yaml.safe_load()` 把 YAML 文件读成 Python 数据；`globals() / getattr()` 把模块名称字符串找到对应的 Python 类。

接下来，就可以正式开始编写 `parse_model()` 引擎了。

---

### 5.2.3 依赖与辅助函数

#### 5.2.3.1 引入依赖

首先在 `ch05/model.py` 文件中，按以下修改需要导入的依赖：

```python
import ast
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from nn import Conv, Bottleneck, C2f, C3k2, SPPF, Concat, SimpleHead
```

要确保上一节新增的 `Concat` 和 `SimpleHead` 模块已经从 `nn/__init__.py` 中导出，否则这里无法直接导入。

#### 5.2.3.2 辅助函数：`make_divisible()`

在写 `parse_model()` 之前，先写一个小工具函数：

```python
def make_divisible(x, divisor):
    """将 x 调整为 divisor 的倍数（向上取整）。"""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor
```

`make_divisible` 函数把一个数字“对齐”到指定倍数。例如 `make_divisible(14, 8)`返回 16，因为：

```text
ceil(14 / 8) × 8
= 2 × 8
= 16
```

为什么需要对齐到 8 的倍数？因为GPU的张量计算引擎（CUDA Cores / Tensor Cores）在通道数为8或16的倍数时效率更高。不对齐的通道数会导致内存访问浪费、计算效率下降。这是一个工程细节，但在实际训练中影响不小。

---

#### 5.2.3.3  从文件到字典 ：`yaml_model_load()` 

接下来写读取yaml文档的函数yaml_model_load：

```python
def yaml_model_load(path):
    """加载 YAML 模型配置文件。"""

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    try:
        d["scale"] = re.search(r"yolo[v]?\d+([nslmx])", path.stem).group(1)
    except (AttributeError, TypeError):
        d["scale"] = "n"  # 默认 nano

    d["yaml_file"] = str(path)

    return d
```

`yaml_model_load()` 函数做三件事：

第一，读取 YAML 文件。通过 `d = yaml.safe_load(f)` 把 YAML 文本转换为 Python 字典d。

根据上一节最新的 `yolo-obb-mini.yaml`，转换后的字典大致是：

```python
{
    "nc": 2,
    "ch": 3,
    "scales": {
        "n": [0.50, 0.25, 1024]
    },

    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 1, "C3k2", [128, False, 0.5]],
        ...
    ],

    "head": [
        [-1, 1, "Conv", [256, 1, 1]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        ...
        [[17, 13, 9], 1, "SimpleHead", ["nc"]]
    ]
}
```

第二，猜测模型规模。从不带后缀的YAML文件的文件名中，通过正则表达式的匹配规则尝试提取 `n/s/m/l/x`。

python中的正则表达式的知识，我们这里就不详细介绍了，需要了解的请自行查阅相关资料。

我们的文件叫 `yolo-obb-mini.yaml` ，匹配不到标准的规模后缀，所以默认使用 `d["scale"] = "n"` ，也就是 Nano 版。

为什么匹配不到就能够默认是 `n` 呢？代码里为什么通过一个 `try...except` 能实现这个效果？

因为根据正则表达式的匹配规则，如果没有能够匹配上的话， `re.search(r"yolo[v]?\d+([nslmx])", path.stem)` 的输出是 `None` ，调用 `None.group(1)`，显然会抛出一个 `AttributeError`类型的错误，这个错误被 `catch`捕获，  就有了 `d["scale"] = "n" `。

第三，记录文件路径。通过代码 `d["yaml_file"] = str(path)` 将YAML文件的路径以字符串的形式保存在字典中，方便后续调试和追溯。

---

### 5.2.4 手搓 `parse_model()` 引擎


#### 5.2.4.1 `parse_model()` 函数签名与初始化

请在`model.py` 文件继续敲下以下代码：

```python
def parse_model(d, ch, verbose=True):
   """
    解析 YAML 字典，动态构建 PyTorch 模型。

    参数:
        d: YAML 配置字典
        ch: 输入通道数（通常为 3）

    返回:
        (nn.Sequential, list): 模型层序列和需要缓存输出的层索引列表
    """

    max_channels = float("inf")
    nc = d.get("nc", 2)
    scales = d.get("scales")
    depth, width = 1.0, 1.0
```

先来看这几个变量分别表示什么。

`max_channels = float("inf")`：先把最大通道数初始化为无穷大，也就是默认不限制通道数。后面如果 YAML 中定义了 `scales`，再用其中的实际值替换它。

`nc = d.get("nc", 2)`：从 YAML 字典中读取类别数 `nc`。如果没有找到，就默认使用 2。

`scales = d.get("scales")`：读取 YAML 中的模型缩放配置，并保存到 `scales` 中。

`depth, width = 1.0, 1.0`：先把深度倍率和宽度倍率都设置为 `1.0`，也就是默认不进行缩放。

由于 `parse_model`方法的代码比较长，为了更好的显示代码缩进关系，除了本部分需要录入的代码外，我会**将之前已经录入的代码以简写的方式显示出来**，请大家在录入代码的时候务必注意。

接下来，根据模型规模读取真正的缩放参数，请继续写入以下代码：

```python
def parse_model(d, ch, verbose=True):
    """
    …………
    """
    …………
    depth, width = 1.0, 1.0
    
    if scales:
        scale = d.get("scale", "n")
        depth, width, max_channels = scales[scale]
```

如果 YAML 中定义了 `scales`，首先通过：

```
scale = d.get("scale", "n")
```

确定当前使用哪个模型规模。这里 `d.get("scale", "n")`表示如果字典d中有 `scale`字段，就取这个字段对应的值（还记得吗，这个值来自于对yaml文件名的推测），如果没有，就默认为 `n`，相当于这里再次设置了一个默认值保险。

然后通过 `scales[scale]` 得到对应的缩放系数，我们这里就是 `[0.50, 0.25, 1024]`，最后按照顺序，把其中三个值分别赋给 `depth, width, max_channels`：

```text
depth = 0.50
width = 0.25
max_channels = 1024
```

然后打印表头，请继续输入以下代码，务必注意代码的缩进：

```python
def parse_model(d, ch, verbose=True):
    ……
    
    if scales:
        ……
        
    if verbose:
       print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}"
             f"{'module':<40}{'arguments':<30}")
```

如果用户传入verbose参数为True，也就是要求打印模型结构表的表头，这段代码输出一个表格头，后面每解析一层就打印一行，方便用户直观看到整个模型结构。

继续写入以下代码，初始化 `parse_model()` 引擎的几个核心变量：

```python
def parse_model(d, ch, verbose=True):
    ……
        
    if verbose:
         ……
         
    ch = [ch] # 每层输出通道数的列表, ch[i] = 第i层的输出通道
    layers, save, c2 = [], [], ch[-1]
    
    # 合并 backbone + head 的所有层定义
    all_layers = d["backbone"] + d["head"]
```

`ch = [ch]`：把初始通道数包装成列表。

后面每解析一层，就把该层的输出通道数追加到这个列表里。

`layers`：存放所有实例化后的模块。

`save`：存放需要被缓存的层索引，也就是后面会被其他层通过 `from` 再次引用的层。

`c2`：当前层的输出通道数，初始化为输入通道数。

`d["backbone"] + d["head"]` 通过字典叠加的方式，把 Backbone 和 Head 的层列表拼接起来，统一编号，为后续逐行遍历做好准备。

---

#### 5.2.4.2 引擎主循环

请继续输入以下代码，要特别注意代码缩进：

```python
def parse_model(d, ch, verbose=True):
    ……
    
    if verbose:
       ……
    ……

    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===
        # "nn.Upsample" → torch.nn.Upsample
        # "Conv"        → 我们 nn 包中的 Conv 类
        if m_name.startswith("nn."):
            m = getattr(nn, m_name[3:])
        else:
            # 从当前模块的全局命名空间查找（即上面 import 的类）
            m = globals()[m_name]
            
        args = list(args)
```

引擎的主循环`for i, (f, n, m_name, args) in enumerate(all_layers)` 中：

```text
i      → 层索引
f      → from
n      → repeats
m_name → module
args   → args
```

分别对应 YAML 中的 `[from, repeats, module, args]`。

##### 步骤 1：字符串 → Python 类（反射）

主循环中的以下代码，通过反射机制获取对应的类：

```python
        if m_name.startswith("nn."):
            m = getattr(nn, m_name[3:])
        else:
            m = globals()[m_name]
```

如果模块名以 `"nn."` 开头，例如 `nn.Upsample` ，说明它是 PyTorch 内置模块。通过 `m_name[3:]` 去掉前缀 `"nn."`，得到 `Upsample`，然后使用 `getattr(nn, "Upsample")`获取真正的 `nn.Upsample` 类。

否则，模块名不是以 `"nn."` 开头，说明它是我们自定义的模块，比如 `Conv、C3k2、SPPF、Concat、SimpleHead` 这些，就用 `globals()[m_name]`从当前文件的全局命名空间中获取对应类。

这完全就是 5.2.1 节讲的反射机制的实际应用。

此时，变量 `m` 已经不是字符串，而是真正的 Python 类对象了。

---

##### 步骤 2：解析参数中的字符串表达式

继续录入以下代码：

```python
def parse_model(d, ch, verbose=True):
    ……    

    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===           
        ……
        
        # === 步骤2: 解析参数中的字符串表达式 ===
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except (ValueError, SyntaxError):
                    pass
```

YAML 中的某些参数可能是字符串形式的变量名或表达式。比如检测头 `[[17, 13, 9], 1, SimpleHead, [nc]]`，
其中的 `nc` 被读取后是字符串 `"nc"`，但我们真正需要的是数值 2。

`locals()[a]`：在当前函数的局部变量中查找变量 `a` ，如果 `a`确实是当前函数的局部变量。例如：

```text
a = "nc"
nc = 2
```

`locals()[a]`就返回 2。

如果它不是局部变量名（`a in locals()` 返回false），就通过 `ast.literal_eval(a)`尝试把字符串解析为 Python 字面量，例如：

```text
"True"     → True
"[1,2,3]"  → [1,2,3]
```

这比 `eval()` 安全，因为 `ast.literal_eval()` 只允许解析有限的 Python 字面量，不能执行任意代码。

如果解析不了，例如 `"nearest"` 本身就是普通字符串，就保持原样。

---

##### 步骤 3：应用 depth 缩放

继续写入以下代码，注意缩进：

```python
def parse_model(d, ch, verbose=True):
    ……

    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===           
        ……
        
        # === 步骤2: 解析参数中的字符串表达式 ===
        for j, a in enumerate(args):
            ……

        # === 步骤3: 应用 depth 缩放（控制模块重复次数）===            
        n = n_ = max(round(n * depth), 1) if n > 1 else n
```

这一行有点密集，但意思是非常清晰的。

这里的 `n` 表示 YAML 中的 `repeats`，也就是模块的基础重复次数。

如果 YAML 写的 `repeats=1`，就直接保持 1。只有 `repeats > 1` 时，才会根据 `depth` 对重复次数进行缩放。

为什么要有个 `n_ `呢？

这是因为后面处理 `C3k2` 这类模块时，`n` 还会被修改。

`C3k2` 的构造函数本身就有一个 `n` 参数 `C3k2(c1, c2, n=1, ...)`，它用来控制模块内部子结构的重复次数。

因此，如果经过深度缩放以后得到 `n = 2`，后面会把这个 2 作为参数传给 `C3k2`，相当于 `C3k2(c1, c2, n=2, ...)`，既然重复次数已经交给 `C3k2` 内部处理，外层就只需要创建一个 `C3k2` 模块，因此后面还会把 `n` 设置 1，避免在外面再重复创建多个完整的 `C3k2`。

而 `n_` 不再修改，始终保存深度缩放后的重复次数，主要用于后面打印模型结构。

---

##### 步骤 4：计算模块输入输出通道数

前面我们已经知道了当前这一层要创建哪个模块，接下来还需要解决另一个核心问题**这个模块需要哪些参数？它的输入通道数和输出通道数分别是多少？**

这也是 `parse_model()` 中比较关键的一部分。

不同模块处理通道的方式并不一样。例如：

-   `Conv`、`C3k2` 等模块需要计算输入、输出通道；
-   `Concat` 会把多个输入的通道数相加；
-   `Upsample` 只改变特征图大小，不改变通道数；
-   `SimpleHead` 则需要同时知道 P3、P4、P5 三个尺度的输入通道数。

所以这里要根据模块类型分别处理。

请继续输入下面这段代码，**注意这些代码仍然位于** `for` **循环内部，缩进不要改变**：

```python
def parse_model(d, ch, verbose=True):
    ……
   
    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===           
        ……
        
        # === 步骤2: 解析参数中的字符串表达式 ===
        for j, a in enumerate(args):
            ……

        # === 步骤3: 应用 depth 缩放（控制模块重复次数）===            
        ……
        
        # === 步骤4: 根据模块类型计算通道数 ===
        if m in {Conv, Bottleneck, SPPF, C2f, C3k2}:
            c1 = ch[f] if isinstance(f, int) else ch[f[0]] # 输入通道
            # YAML中指定的输出通道
            c2 = args[0]
            # 应用 width 缩放 + max_channels 限制
            c2 = make_divisible(min(c2, max_channels) * width,8)

            args = [c1, c2, *args[1:]]
            
            # 对 C2f/C3k2 等块，将 repeats 插入构造参数
            if m in {C2f, C3k2}:
                args.insert(2, n)
                n = 1 # 不再外层重复，改为内部重复
                
        elif m is Concat:
            # Concat 的输出通道 = 所有输入层通道数之和
            c2 = sum(ch[x] for x in f)

        elif m is SimpleHead:
            # 检测头: 需要知道各输入层的通道数
            args.append([ch[x] for x in f])
            c2 = nc + 8

        elif m is nn.Upsample:
            # Upsample 不改变通道数
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]
```

从整体上看，这一大段代码其实只是在做一件事：**根据当前模块的类型，计算它需要的通道数，并把 YAML 中的参数整理成真正创建模块时需要的参数。** 下面再分别来看。

* * *

###### 1、`Conv`、`C3k2` 等特征模块

首先是 `if m in {Conv, Bottleneck, SPPF, C2f, C3k2}:`，这些模块有一个共同特点：它们都有明确的**输入通道数** `c1` **和输出通道数** `c2` ，可以用同一套逻辑来处理。 

对于这几个模块，我们先把这两个值计算出来。

输入通道数`c1` 看代码：`c1 = ch[f] if isinstance(f, int) else ch[f[0]]`。

这里的 `ch` 保存着前面每一层的输出通道数，例如：

```
ch[0] → 第0层输出通道数，比如16
ch[1] → 第1层输出通道数，比如64
ch[2] → 第2层输出通道数，比如128
...
```

而 `f` 就是 YAML 中的 `from`，它常见的是一个整数，比如“-1”，表示输入来自上一层，或者是一个列表，比如“[-1, 6]”，表示输入来自于还几个不同层。

当 `f` 是整数时，`c1` 就等于 `ch[f] `，如果 `f` 是 -1，那 `c1 = ch[-1]`，表示其输入通道数是上一层的输出通道数。也就是说**上一层输出多少通道，当前层就接收多少通道。**

当前 YAML 中，`Conv`、`C3k2`、`SPPF` 这些模块都是单输入模块，因此 `f` 实际上都是一个整数。

代码中保留了 `else ch[f[0]]`， 是为了增加代码的通用性。如果以后遇到列表形式的 `from`，至少能够取得其中第一个来源的通道数。

真正像 `Concat` 这样需要同时处理多个输入的模块，后面会单独处理。

* * *

输出通道数 `c2` 看代码 `c2 = args[0]`。

上一节已经讲过，YAML 中 args 是指定模块的参数，其顺序是对应模块构造函数的参数顺序。输入通道 `c1` 刚刚已经自动计算出来了，所以args中的第一个参数 `args[0]`，对应的就是模块的基准输出通道数 `c2`。

`c2 = args[0]`就是先把 YAML 中定义的**基准输出通道数**取出来。

注意，此时的 `c2` 还不是 Nano 模型真正使用的输出通道数，因为还有个缩放系数。

接下来，我们要调用make_divisible()方法：

```
c2 = make_divisible(
    min(c2, max_channels) * width,
    8
)
```

这一行就是把 YAML 中的基准通道数转换成当前模型真正使用的通道数。这段代码的计算逻辑很清晰，就不再详细介绍了。

现在，当前模块 `c1` 、`c2` 具体的值已经计算出来了。接下来自然是把它们加入到 `args` 中，形成真正的模块构造函数的参数：

```python
args = [c1, c2, *args[1:]]
```

举例例子，原来YAML中的 `args = [64, 3, 2]`，其中：

```text
64 → 基准输出通道
3  → 卷积核
2  → stride
```

经过处理后变成 `args = [3, 16, 3, 2]`，也就是“输入通道”、“输出通道”、“卷积核大小”、“stride”，就可以直接用于模块的构建了：

```python
Conv(*args)
```

等价于 `Conv(3, 16, 3, 2)`，这里体现了 YAML 动态建模中一个很重要的设计：

> **YAML 不需要手工填写输入通道数** `c1` **，因为** ` parse_model()  `**可以根据** `from` **自动推算。**

我们刚刚在讲 `n` 、 `n_` 的时候说，`C2f`、`C3k2`两个模块有点特殊的，这就要我们特殊情况特殊处理一下：

```python
if m in {C2f, C3k2}:
    args.insert(2, n)
    n = 1
```

 `C2f`、`C3k2` 和普通 `Conv` 有一点不同。以 `C3k2` 为例，它的构造函数本身就有：

```
C3k2(c1, c2, n=1, c3k=False, e=0.5, ...)
```

其中的 `n` 就是模块内部子结构的重复次数。因此，前面经过 `depth` 缩放后得到的 `n`，需要作为参数传进 `C3k2`。例如此时：

```
args = [32, 32, False, 0.5]
n = 2
```

就需要执行：

```
args.insert(2, n)
```

注意，索引 `2` 表示的是**第 3 个元素的位置**，也就是插在 `c1、c2` 后面。于是得到 `[32, 32, 2, False, 0.5]` ，最终构建的模块相当于：

```
C3k2(
    32,
    32,
    n=2,
    c3k=False,
    e=0.5
)
```

既然重复次数已经交给 `C3k2` 内部处理，外面就只需要创建一个 `C3k2` 模块，所以 `n` 要重新设置为 1，表示**外层不再重复创建整个** `C3k2` **。**

* * *

###### 2、`Concat`：多个输入通道直接相加

接下来处理`Concat` 模块：

```
elif m is Concat:
    c2 = sum(ch[x] for x in f)
```

`Concat` 和前面的模块不同，它不会重新生成一组新的特征通道，而是把多个特征沿通道维拼接起来，所以**Concat 的输出通道数 c2 = 所有输入特征的通道数之和。** 我们通过一个循环依次取出指定通道的输出通道数然后加起来。

###### 3、 `SimpleHead`：三个尺度的输入通道

接下来处理`SimpleHead`模块：

```
elif m is SimpleHead:
    args.append([ch[x] for x in f])
    c2 = nc + 8
```

`SimpleHead` 同样不是普通的单输入模块。YAML 最后一层 `[[17, 13, 9], 1, SimpleHead, [nc]]` 表示它同时接收三层的输入，因此 `[ch[x] for x in f]` 依次取得这三层的通道数，然后通过 `args.append()` 方法把这个列表插入到args中，形成新的args，比如原来 `args = [2]`， `args.append()` 之后就变成：

```
args = [2, [32, 64, 128]]
```

最终相当于：

```
SimpleHead(
    nc=2,
    ch=[32, 64, 128]
)
```

正好对应 `SimpleHead` 的三个临时预测头。另外 `c2 = nc + 8` 表示当前每个预测分支的输出通道数。


###### 4、`nn.Upsample`：通道数保持不变

对于上采样模块`nn.Upsample`：

```
elif m is nn.Upsample:
    c2 = ch[f] if isinstance(f, int) else ch[f[0]]
```

`Upsample` 只改变特征图的空间尺寸，不改变通道数。所以直接让 `输出通道数 = 输入通道数`即可。


###### 5、其他模块：暂时保持通道数不变

最后 `c2 = ch[f] if isinstance(f, int) else ch[f[0]]` 是一个兜底处理。对于当前没有单独定义通道计算规则的模块，暂时认为 `输出通道数 = 输入通道数`，

我们目前 YAML 中真正使用到的主要模块，都已经在前面分别进行了处理。

到这里，模块对应的构建参数，就都已经处理完成了。看起来代码分支很多，但核心目标始终只有一个：

> **在真正创建模块之前，先根据 YAML 中的网络连接关系，把这一层需要的参数准备完整。**

##### 步骤 5：完成当前层的构建

经过前面的处理，我们已经知道了：

-   当前这一层是什么模块；
-   它的输入来自哪里；
-   输入、输出通道数是多少；
-   最终需要传入哪些参数；
-   是否需要重复。

接下来，就可以真正创建这一层，并把这一层相关的信息记录下来备用了。

这一部分代码仍然位于 `for` 循环内部。为了避免被缩进弄乱，建议先把完整代码输入，再逐段理解：

```python
def parse_model(d, ch, verbose=True):
    ……
    
    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===
        ……        
        # === 步骤2: 解析参数中的字符串表达式 ===
        for j, a in enumerate(args):
            ……
        # === 步骤3: 应用 depth 缩放（控制模块重复次数）===           
        ……        
        # === 步骤4: 根据模块类型计算通道数 ===
        if m in {Conv, Bottleneck, SPPF, C2f, C3k2}:
            ……                
        elif m is Concat:
            ……
        else:
            ……
            
        # === 步骤5: 实例化模块 ===
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        # 记录当前层的信息
        t = m.__name__ # 类型名

        m_.np = sum(x.numel() for x in m_.parameters()) # 参数量
        m_.i = i     # 层索引
        m_.f = f     # 输入来源
        m_.type = t  # 类型字符串

        # 打印当前层信息
        if verbose:
            print(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f} "
                  f"{t:<40}{str(args):<30}")

        # 如果某层被其他层引用 (from != -1)，则需要缓存其输出
        save.extend( x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        # 保存当前模块
        layers.append(m_)

        # 记录当前层输出通道数
        if i == 0:
            ch = []

        ch.append(c2)
```

这段代码做的事情，可以概括成：

```
创建当前模块
    ↓
记录当前层的信息
    ↓
检查哪些层以后还要继续使用
    ↓
把当前层加入模型
    ↓
记录当前层的输出通道数
```

下面分别来看。

* * *

###### 1、实例化当前模块

```python
m_ = (
    nn.Sequential(*(m(*args) for _ in range(n)))
    if n > 1
    else m(*args)
)
```

终于到了“实例化”这一步。

`m(*args)` 表示用反射得到的类 `m` 和计算好的参数 `args` 创建模块实例。

例如：

```text
m = Conv
args = [3, 16, 3, 2]
```

那么 `m(*args)` 就等价于：

```python
Conv(3, 16, 3, 2)
```

如果 `n > 1`，说明需要在外部重复多个相同模块，就用 `nn.Sequential` 串联起来。

不过在我们当前的 Nano 教学模型中，经过深度缩放和 `C3k2` 的特殊处理后，实际外部 `n` 基本都是 1。


###### 2、记录当前层的信息

模块创建完成以后，继续：

```
t = m.__name__

m_.np = sum(x.numel() for x in m_.parameters())
m_.i = i
m_.f = f
m_.type = t
```

这几行代码是在给每个模块打上“身份标签”。

- `m_.i = i`：这是第几层。

- `m_.f = f`：这一层的输入来自哪里。

- `m_.np`：参数量，用于打印。

- `m_.type`：模块类型名称，用于打印。

后面的动态 `forward()` 会使用 `i` 和 `f` 来决定从哪里取输入、哪些层需要缓存。

`m_.parameters()` 可以取得模块中的所有参数，`x.numel()`表示某个参数张量中一共有多少个数值。把它们全部加起来，就是当前模块的总参数量。


###### 3、打印当前层信息

```python
if verbose:
    print(
        f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  "
        f"{t:<40}{str(args):<30}"
    )
```

把当前这一层的重要信息打印出来。

主要包括：

```
i       → 层号
f       → 输入来自哪里
n_      → 深度缩放后的 repeats
m_.np   → 参数量
t       → 模块名称
args    → 最终创建模块时使用的参数
```

这样整个模型解析完成以后，就可以在终端中看到一张完整的网络结构表。

###### 4、构建 `save list`

```
save.extend(
    x % i
    for x in ([f] if isinstance(f, int) else f)
    if x != -1
)
```

当某一层的 `from` 引用了之前的某一层（不是 `-1`），那被引用层的输出后面还要再次使用，所以我们需要提前记录**哪些层的输出以后还会再次使用。** 这就是 `save` 列表的作用。

例如 YAML 第 12 层是 `[[-1, 6], 1, Concat, [1]]`，所以第 6 层需要缓存。同理 `[[-1, 4], 1, Concat, [1]]`，需要缓存第 4 层， `[[17, 13, 9], 1, SimpleHead, [nc]]`，还会引用第 17、13、9 层。因此我们之前写的 YAML 最终需要缓存的关键层为： `[4, 6, 9, 13, 17]`。

`[f] if isinstance(f, int) else f` 这一行代码的作用是：如果 `f` 是整数，就先把它包装成列表，如果本来就是列表，那就直接使用。

接下来 `if x != -1` 表示不处理`-1` ，因为`-1` 表示“上一层”，不用额外缓存，当前变量天然保存着上一层输出。

`x % i` 主要用于处理负数形式的相对层索引。例如当前正在构建第10层，也就是 `i = 10`，如果 YAML 中写：`from = -3`，这表示 **从当前层向前数3层**，也就是第 7 层。

而 Python 中 `-3 % 10` 结果正好是7。因此 `x % i` 可以把这种负数形式的相对引用转换成真正的层索引。如果本来就是正数，则不会改变。

###### 5、保存当前模块

接下来 `layers.append(m_)` 把前面创建好的网络层保存到 `layers` 列表中。

整个 YAML 全部解析结束后，`layers`中就保存了所有网络模块。


###### 6、记录当前层的输出通道数

最后还要把当前层的输出通道数记录下来：

```
if i == 0:
    ch = []

ch.append(c2)
```

为什么需要不断维护 `ch` 呢？

因为后面的层还要通过 `ch[f]` 查询前面某一层的输出通道数。所以每创建完成一层，就必须把这一层的输出通道数 `c2` 加入 `ch`。

这里有一个看起来有些奇怪的处理：

```
if i == 0:
    ch = []
```

为什么第 0 层构建完成以后，要把 `ch` 清空一次？

回忆 `parse_model()` 刚开始时，我们写了： `ch = [ch]` ，如果输入是 RGB 图像：`ch = [3]`，这里的3表示原始输入图像的通道数。

在构建第 0 层时，这个 3 非常重要。因为 `c1 = ch[-1]`，于是第 0 层知道自己的输入是 3 通道。

但是，第 0 层构建完成以后，我们希望：

```
ch[0] → 第0层输出通道
ch[1] → 第1层输出通道
ch[2] → 第2层输出通道
...
```
如果不把最开始的 `[3]` 清除，就会变成：

```
ch[0] → 原始输入
ch[1] → 第0层输出
ch[2] → 第1层输出
...
```

这样层号就全部错开了一位。

所以第 0 层构建完成后需要执行：

```
if i == 0:
    ch = []
```

这实际上是在完成一次 `ch` **列表索引与模型层号的对齐。**


##### 步骤 6：循环结束：返回完整模型

前面的代码会在 `for` 循环中不断执行：

```
读取一层 YAML
    ↓
整理参数
    ↓
创建模块
    ↓
记录结构信息
    ↓
更新 save
    ↓
保存输出通道
    ↓
继续解析下一层
```

直到 YAML 中的所有层都构建完成。这时退出 `for` 循环，最后执行：

```
def parse_model(d, ch, verbose=True):
    ……
    
    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===
        ……
       
        # === 步骤5: 实例化模块 ===
        ……
       
    return nn.Sequential(*layers), sorted(set(save))
```

请把最后一行代码加入文件中，注意这一行已经**不在** `for` **循环内部**，请注意他的缩进。

`nn.Sequential(*layers)`  把 `layers` 中保存的所有模块组织成一个完整的模型层序列。

`sorted(set(save))` 完成对所有层的排序。

其中：`set(save)`去掉可能重复出现的层号；然后使用 `sorted(...)`方法按照从小到大的顺序排列。

因此，`parse_model()`引擎最终返回两个结果：

```
model, save = parse_model(...)
```

其中： `model` 表示根据 YAML 自动构建出来的模型层序列；`save` 表示动态 forward 时需要保留的中间层。

到这里，YAML 中写下的网络结构，已经真正变成了 PyTorch 模块。

`parse_model()`引擎构建完成。

---

### 5.2.5 通道推导的完整流程：以前三层为例

让我们用实际数据走一遍前三层的解析过程，帮你建立直觉。

```text
初始状态:
ch = [3]
depth = 0.50
width = 0.25
```

###### 第 0 层

```yaml
[-1, 1, Conv, [64, 3, 2]]
```

解析：

```text
反射:
"Conv" → Conv 类

depth缩放:
n = 1，不缩放

通道计算:
c1 = ch[-1] = 3

c2 = 64
   → min(64, 1024) × 0.25
   → 16
   → make_divisible(16, 8)
   → 16

args = [3, 16, 3, 2]
```

实例化 `Conv(3, 16, 3, 2)`

此时 `ch = [16]`

###### 第 1 层

```yaml
[-1, 1, Conv, [128, 3, 2]]
```

解析：

```text
c1 = ch[-1] = 16

c2 = 128
   → min(128, 1024) × 0.25
   → 32

args = [16, 32, 3, 2]
```

实例化 `Conv(16, 32, 3, 2)`

此时 `ch = [16, 32]`。

###### 第 2 层

```yaml
[-1, 1, C3k2, [128, False, 0.5]]
```

解析：

```text
depth缩放:
n = 1，不缩放

c1 = ch[-1] = 32

c2 = 128
   → min(128, 1024) × 0.25
   → 32

原始 args:
[128, False, 0.5]

补入 c1、实际 c2:
[32, 32, False, 0.5]

插入 C3k2 内部 repeats:
[32, 32, 1, False, 0.5]
```

实例化：

```python
C3k2(
    32,
    32,
    n=1,
    c3k=False,
    e=0.5
)
```

此时 `ch = [16, 32, 32]`

看到了吗？通道数开始一层层自动传递：

```text
3 → 16 → 32 → 32 → 64 → 64 → 128 → ...
```

这就是“通道推导”的核心思想。

下一节，我们就要启动这个“引擎”，从 `yaml` 文件自动构建模型。


---

## 附录
`parse_model` 函数的代码比较长，中间缩进很多，为了方便理解，讲解得时候代码切分的比较少碎，为了方便大家录入代码，把`parse_model` 函数以附录的形式放在最后，供大家参考。

`parse_model` 函数完整代码：

```python
def parse_model(d, ch, verbose=True):
    """
    解析 YAML 字典，动态构建 PyTorch 模型。

    参数:
        d: YAML 配置字典
        ch: 输入通道数（通常为 3）

    返回:
        (nn.Sequential, list): 模型层序列和需要缓存输出的层索引列表
    """

    max_channels = float("inf")
    nc = d.get("nc", 2)
    scales = d.get("scales")
    depth, width = 1.0, 1.0
    
    if scales:
        scale = d.get("scale", "n")
        depth, width, max_channels = scales[scale]
        
    if verbose:
       print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}"
             f"{'module':<40}{'arguments':<30}")
    ch = [ch] # 每层输出通道数的列表, ch[i] = 第i层的输出通道
    layers, save, c2 = [], [], ch[-1]
    
    # 合并 backbone + head 的所有层定义
    all_layers = d["backbone"] + d["head"]

    for i, (f, n, m_name, args) in enumerate(all_layers):
        # === 步骤1: 字符串 → Python 类 ===
        # "nn.Upsample" → torch.nn.Upsample
        # "Conv"        → 我们 nn 包中的 Conv 类
        if m_name.startswith("nn."):
            m = getattr(nn, m_name[3:])
        else:
            # 从当前模块的全局命名空间查找（即上面 import 的类）
            m = globals()[m_name]
            
        args = list(args)
        
        # === 步骤2: 解析参数中的字符串表达式 ===
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                except (ValueError, SyntaxError):
                    pass
        # === 步骤3: 应用 depth 缩放（控制模块重复次数）===            
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        
        # === 步骤4: 根据模块类型计算通道数 ===
        if m in {Conv, Bottleneck, SPPF, C2f, C3k2}:
            c1 = ch[f] if isinstance(f, int) else ch[f[0]] # 输入通道
            # YAML中指定的输出通道
            c2 = args[0]
            # 应用 width 缩放 + max_channels 限制
            c2 = make_divisible(min(c2, max_channels) * width,8)

            args = [c1, c2, *args[1:]]
            
            # 对 C2f/C3k2 等块，将 repeats 插入构造参数
            if m in {C2f, C3k2}:
                args.insert(2, n)
                n = 1 # 不再外层重复，改为内部重复
                
        elif m is Concat:
            # Concat 的输出通道 = 所有输入层通道数之和
            c2 = sum(ch[x] for x in f)

        elif m is SimpleHead:
            # 检测头: 需要知道各输入层的通道数
            args.append([ch[x] for x in f])
            c2 = nc + 8

        elif m is nn.Upsample:
            # Upsample 不改变通道数
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]

        else:
            c2 = ch[f] if isinstance(f, int) else ch[f[0]]
            
        # === 步骤5: 实例化模块 ===
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        # 记录当前层的信息
        t = m.__name__ # 类型名

        m_.np = sum(x.numel() for x in m_.parameters()) # 参数量
        m_.i = i     # 层索引
        m_.f = f     # 输入来源
        m_.type = t  # 类型字符串

        # 打印当前层信息
        if verbose:
            print(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f} "
                  f"{t:<40}{str(args):<30}")

        # 如果某层被其他层引用 (from != -1)，则需要缓存其输出
        save.extend( x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        # 保存当前模块
        layers.append(m_)

        # 记录当前层输出通道数
        if i == 0:
            ch = []

        ch.append(c2)    
    return nn.Sequential(*layers), sorted(set(save))

```

