---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 数据加载与预处理：把数据喂进模型

模型跑通之后，真实项目里最先出问题的地方往往不是算子，而是数据：形状对不上、类型对不上、
GPU 在等 CPU 读文件。这一篇把 Jittor 的数据管道讲清楚：

1. 怎么定义数据集、按批取数；
2. `len(loader)`、`shuffle`、`drop_last`、`num_workers` 的**真实**语义；
3. 预处理用 `jt.transform`，以及它和 `torchvision.transforms` 的差别。

全篇用内存里现造的假数据，不下载任何东西，CPU 上几十秒跑完。

## 1. 最小可跑的数据管道

一个数据集就是一个类：继承 `jt.dataset.Dataset`，实现 `__getitem__` 返回第 `index` 条样本，
再告诉它一共有多少条、怎么分批。

```{code-cell} ipython3
import time

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0

# 造一批假数据：2 维特征，标签由特征之和的正负决定
rng = np.random.default_rng(0)
N = 64
features = rng.normal(size=(N, 2)).astype("float32")
labels = (features.sum(axis=1) > 0).astype("float32").reshape(N, 1)


class Blobs(jt.dataset.Dataset):
    def __init__(self):
        super().__init__()            # 分批相关的属性都在这里
        self.set_attrs(total_len=N)   # 声明样本总数
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

两个要点：

* `super().__init__()` 必须调用。`batch_size`、`shuffle`、`num_workers` 这些属性都是它装的，
  不调用后面 `set_attrs` 会直接报 `AssertionError`。
* `set_attrs(total_len=N)` 是"一共有多少条样本"。迭代和 `len()` 都依赖它。

`__getitem__` 返回什么，批里就是什么。返回 **numpy 数组**最省事：Jittor 会把一批样本按字段
堆叠起来（`np.stack`），再转成 `jt.Var`。返回 `jt.Var` 也可以；但**不要返回 Python 的
`int`/`float` 标量**，堆叠那一步只接受数组。

## 2. 取批：`DataLoader`

```{code-cell} ipython3
loader = jt.dataset.DataLoader(Blobs(), batch_size=16)

print("样本总数:", loader.total_len)
print("批次数  :", len(loader))

for step, (x, y) in enumerate(loader):
    print("第 %d 批: x%s y%s" % (step, tuple(x.shape), tuple(y.shape)))
```

`jt.dataset.DataLoader(...)` 不是一个新的加载器对象，它是个**函数**：对你的数据集调用
`set_attrs`，然后**把同一个对象返回**。所以下面两行完全等价：

```{code-block} python
loader = jt.dataset.DataLoader(Blobs(), batch_size=16, shuffle=True)
loader = Blobs().set_attrs(batch_size=16, shuffle=True)
```

由此带来一个容易踩的点：**对同一个数据集调用两次 `DataLoader` 不会得到两个加载器**，
第二次改的还是那个对象。要两份配置，就建两个数据集实例。

`set_attrs` 能设的键（名字写错会立刻 `AssertionError`，不会静默忽略）：

| 键 | 含义 | 默认 |
| --- | --- | --- |
| `batch_size` | 每批多少条 | 16 |
| `shuffle` | 每个 epoch 打乱顺序 | `False` |
| `drop_last` | 丢掉最后不足一批的样本 | `False` |
| `num_workers` | 后台取数的进程数 | 0 |
| `total_len` | 样本总数 | `None` |
| `keep_numpy_array` | 批直接给 numpy，不转 `jt.Var` | `False` |
| `stop_grad` | 数据不参与求导 | `True` |
| `endless` | 取完不停止，反复循环 | `False` |

## 3. `len(loader)` 到底是几

这是最容易记错的一个数字。看四种组合：

```{code-cell} ipython3
ds = Blobs()
for batch_size, drop_last in [(16, False), (16, True), (10, False), (10, True)]:
    ds.set_attrs(batch_size=batch_size, drop_last=drop_last)
    print("batch_size=%-3d drop_last=%-5s -> len = %d"
          % (batch_size, drop_last, len(ds)))
```

64 条样本、每批 16 条，`len` 是 **4**——这是**批次数**，不是样本数。`drop_last=True` 会丢掉
装不满的那一批：每批 10 条时 64 条装成 6 整批 + 剩 4 条，所以 `len` 从 7 变成 6。

> `drop_last` 的默认值是 `False`：默认**会**留下最后一个小批。这和 PyTorch 一致，
> 但和这个类的文档字符串里写的不一致，以代码为准。

### 一个陷阱：自己定义了 `__len__`

PyTorch 的写法要求你实现 `__len__` 返回**样本数**。Jittor 里也允许，但它会**覆盖**基类的
`__len__`，于是 `len(loader)` 的含义就跟着变了：

```{code-cell} ipython3
class TorchStyleBlobs(jt.dataset.Dataset):
    def __init__(self):
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return N                       # PyTorch 习惯：返回样本数

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


torch_style = TorchStyleBlobs().set_attrs(batch_size=16)
print("len(loader)     :", len(torch_style), "  <- 你自己定义的：样本数")
print("total_len       :", torch_style.total_len, "（还没迭代过，是 None）")

for _ in torch_style:                  # 迭代一次，Jittor 才把 total_len 推出来
    break
print("迭代后 total_len:", torch_style.total_len)
print("真正的批次数    :", torch_style.__batch_len__())
```

两种写法都能正常迭代——Jittor 在第一次迭代时会把 `total_len` 从 `len(self)` 推出来。
区别只在 `len()` 的含义：

* **不定义 `__len__`**，用 `set_attrs(total_len=N)`：`len(loader)` = **批次数**。
* **定义 `__len__` 返回样本数**（PyTorch 习惯）：`len(loader)` = **样本数**，
  批次数要用 `loader.__batch_len__()`。

写进度条、算 epoch 数的时候，先确认自己在哪一种里。要批次数又不想记，就用 `__batch_len__()`，
它在两种写法下都对。

## 4. 打乱与并行取数

`shuffle=True` 时，**每次开始一个新的 `for ... in loader` 都会重新打乱**：

```{code-cell} ipython3
shuffled = jt.dataset.DataLoader(Blobs(), batch_size=16, shuffle=True)

def first_batch_labels(loader):
    for _, y in loader:
        return [int(v) for v in y.reshape(-1).numpy()]
    return []

epoch1 = first_batch_labels(shuffled)
epoch2 = first_batch_labels(shuffled)
print("第一轮第一批标签:", epoch1)
print("第二轮第一批标签:", epoch2)
print("两轮顺序相同?", epoch1 == epoch2)
```

`num_workers` 决定用几个**后台进程**去跑 `__getitem__`。它不改变数据内容和顺序，
只改变谁在干活：

```{code-cell} ipython3
serial = jt.dataset.DataLoader(Blobs(), batch_size=16, shuffle=False, num_workers=0)
parallel = jt.dataset.DataLoader(Blobs(), batch_size=16, shuffle=False, num_workers=2)

def batch_label_sums(loader):
    return [int(y.reshape(-1).sum()) for _, y in loader]

print("串行   :", batch_label_sums(serial))
print("两进程 :", batch_label_sums(parallel))
print("内容一致:", batch_label_sums(serial) == batch_label_sums(parallel))
```

几个实际约束：

* worker 进程是 **fork** 出来的（Jittor 显式指定了 `fork`，因为在 Python 3.14 上默认的
  `forkserver` 会试图 pickle 内部结构而直接失败）。fork 意味着**子进程里不能有未刷新的
  文件句柄状态**，也不要指望 `__init__` 里临时开的东西能带过去。
* 加了 worker 反而更慢是常事：进程启动有成本，而 `__getitem__` 太轻的时候，通信比计算贵。
  只有当单条样本足够重（读盘、解码图片、增广）时才划算。
* `pin_memory` 和 `persistent_workers` 这两个 PyTorch 参数**被接受但不起作用**。Jittor 的
  worker 本来就长期存活（相当于 `persistent_workers=True`），而批次走的是普通可分页内存，
  没有锁页带来的拷贝加速。传了不会报错，但别指望它有效果。

## 5. 预处理：`jt.transform`

`jt.transform` 提供的是一套**图像**变换。它和 `torchvision.transforms` 有三处必须知道的差别。
图像相关的类依赖 Pillow（`pip install pillow`，教程工具链里已经带上了）。

```{code-cell} ipython3
from PIL import Image
from jittor import transform

# 32x32 的彩色图，在内存里造，不读文件
picture = Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))

preprocess = transform.Compose([
    transform.Resize((16, 16)),          # 几何变换吃 PIL 图
    transform.ToTensor(),                # 转成 CHW float32、值域 [0, 1]
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

sample = preprocess(picture)
print("类型:", type(sample).__name__)     # 注意：numpy，不是 jt.Var
print("形状:", sample.shape)              # CHW，不是 HWC
print("范围: %.3f .. %.3f" % (sample.min(), sample.max()))
```

**差别一：`ToTensor()` 返回的是 numpy，不是 `jt.Var`。** 它做的是"PIL/numpy → CHW float32、
值域 [0, 1]"，仅此而已。真正的 `jt.Var` 是在 `DataLoader` 堆叠成批的那一步才产生的。
所以别在 `__getitem__` 里手动 `jt.array(...)`——交给加载器做。

**差别二：几个类改了名字。** 从 PyTorch 迁过来最容易在这里卡住：

| torchvision | jt.transform | 说明 |
| --- | --- | --- |
| `Normalize(mean, std)` | `ImageNormalize(mean, std)` | 名字不同 |
| `Grayscale(n)` | `Gray(num_output_channels)` | 名字不同 |
| `ToTensor()` | `ToTensor()` | 同名，但返回 numpy |
| `Resize` / `RandomCrop` / `CenterCrop` / `RandomHorizontalFlip` | 同名 | 几何类变换吃 PIL 图 |
| `RandomResizedCrop` / `ColorJitter` / `RandomRotation` | 同名 | 同上 |

**差别三：几何变换只认 PIL 图。** 传 numpy 进去它内部会先转成 PIL 再改回来。所以
`Compose` 里的顺序必须是：**先几何变换（PIL），再 `ToTensor`（变 numpy CHW），
最后 `ImageNormalize`**。把 `ToTensor` 放前面会让后面的 `Resize` 走一趟多余的转换。

### 变换放在哪里

放在 `__getitem__` 里。这样每个样本的预处理会落在取数的那一步——也就能被 `num_workers` 并行掉：

```{code-cell} ipython3
N_IMG = 8
pictures = [Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))
            for _ in range(N_IMG)]


class ImageBlobs(jt.dataset.Dataset):
    def __init__(self):
        super().__init__()
        self.set_attrs(total_len=N_IMG)
        self.pictures = pictures
        self.labels = labels[:N_IMG]
        self.preprocess = transform.Compose([
            transform.Resize((16, 16)),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        return self.preprocess(self.pictures[index]), self.labels[index]


image_loader = jt.dataset.DataLoader(ImageBlobs(), batch_size=4)
for x, y in image_loader:
    print("一批图像:", tuple(x.shape), x.dtype, " 标签:", tuple(y.shape))
```

## 6. 取数也会是瓶颈

`__getitem__` 里做的每一件事（读文件、解码、增广）都是**在 GPU 排队之前**发生的。它一慢，
GPU 就空转——而这段时间在算子画像里看不到，因为根本没有算子。

用一个刻意变慢的 `__getitem__` 来看多进程取数。先确认它**不改变数据**：

```{code-cell} ipython3
class SlowBlobs(jt.dataset.Dataset):
    def __init__(self, delay):
        super().__init__()
        self.set_attrs(total_len=N)
        self.delay = delay

    def __getitem__(self, index):
        time.sleep(self.delay)         # 模拟读盘 / 解码
        return features[index], labels[index]


def one_epoch(loader):
    return [int(y.reshape(-1).sum()) for _, y in loader]


serial = jt.dataset.DataLoader(SlowBlobs(0.02), batch_size=8, num_workers=0)
parallel = jt.dataset.DataLoader(SlowBlobs(0.02), batch_size=8, num_workers=2)

print("串行  :", one_epoch(serial))
print("两进程:", one_epoch(parallel))
print("内容一致:", one_epoch(serial) == one_epoch(parallel))
```

### 为什么这里**故意不给**加速比

多进程取数到底快多少，必须自己量；而**天真地量一定会得到错的结果**。两个坑：

* **进程启动的成本是固定的。** worker 要 fork、要建通信通道。把第一轮算进去，
  `num_workers=2` 在小数据集上甚至会**更慢**——"多进程没用"这个常见结论多半就是这么来的。
* **worker 会提前把后面的批装进缓冲区。** 所以紧接着的那一轮可能几乎瞬间返回：
  它不是变快了，而是上一轮已经替它干完了。这台机器上实测过一组数字——
  串行两轮 2.57 s，两进程两轮 0.64 s，算出来 **4.00x**。

最后一个数字就是最好的警示：**两个 worker 不可能有 4 倍**。越过理论上限的加速比不是好消息，
是测量错了的信号。预取把上一个测量窗口的时间挪到了下一个窗口里，而启动成本又被热身吃掉了。

正确的做法：**先热身一轮，然后按整段长时间求和**（不要取单轮最小值），并且拿这个
上限去校验结果——2 个 worker 最多 2 倍，超过了就说明哪里有预取的残留。
真实训练里数据集大、epoch 多，这两个坑都会被摊薄，但在小规模上量出来的数字**不能直接用**。

所以：

* worker 数不是越多越好，**先量一下再定**；
* 单条样本很轻时（比如就是在内存里切一刀），加 worker 只会更慢；
* worker 的存活期是整个数据集的生命周期，第一轮贵、之后便宜；
* 判断"是不是卡在取数上"的方法很直接：把 `__getitem__` 里的活儿临时去掉再测一遍。
  如果训练快了一大截，瓶颈就在取数，不在模型。

## 7. 检查清单

- `super().__init__()` 调了吗？没调的话 `set_attrs` 会报 `AssertionError`。
- `total_len` 设了吗，还是靠自定义 `__len__`？两种写法下 `len(loader)` 含义不同。
- `__getitem__` 返回的是数组吗？返回 Python 标量会在堆叠那一步失败。
- 需要打乱吗？`shuffle` 默认是 `False`，每个 epoch 重新开始迭代才会重新打乱。
- 最后一个小批要丢吗？`drop_last` 默认 `False`。
- 预处理放在 `__getitem__` 里了吗？放在外面就吃不到 `num_workers`。
- 取数够快吗？用"临时去掉 `__getitem__` 里的活儿再测一次"来判断。

下一篇看训练的中断与恢复：[检查点与断点续训](checkpointing.md)。
