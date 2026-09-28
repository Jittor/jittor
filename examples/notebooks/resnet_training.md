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

# 训练一个残差网络：一个能读的对照实验

图像分类的**标准训练流程**由几件固定的事组成：划分训练集与验证集、按 batch 迭代、
在每个 epoch 结束后**在验证集上评估**、区分训练模式和评估模式。

这一篇把这条流程完整走一遍，并且做一件比「跑通」更有价值的事：用**同一个数据集、
同样的步数、同样的随机种子**，只在**有没有残差连接**这一点上做对照，然后读两边的
验证准确率。

这一篇会：

* 用**离线合成数据**跑一次真正的分类训练（不需要下载任何东西）；
* 说明为什么**必须有验证集**——只看训练集准确率会被骗；
* 用一个对照实验给出残差连接的作用：深度 6 时验证准确率 **0.72 对 0.42**；
* 记录一个**量出来和教科书说法相反**的读数，并解释为什么不能照搬结论；
* 讲清 `state_dict`、`eval()` 与 BatchNorm 运行统计量之间的关系。

> 整篇在 CPU 上约 **2.5 分钟**，其中绝大部分是那两个模型的训练。
> 有显卡时可以改 `jt.flags.use_cuda = 1`。

```{code-cell} ipython3
import time

import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(0)

IMG = 16          # 图像边长
CLASSES = 6       # 类别数
N_TRAIN = 128
N_VAL = 64
CHANCE = 1.0 / CLASSES
```

## 1. 一个能离线的分类任务

教程要能离线跑，就不能依赖下载数据集。这里的做法是**合成**一个六分类任务：每类是一个
几何图案（横条、竖条、方块、十字、棋盘、对角线）加高斯噪声。它足够难——**不能靠单像素
判断**，必须真的学出卷积核；也足够小——CPU 上几十秒能跑完。

```{code-cell} ipython3
def make_dataset(n, seed):
    """返回 (图像, 标签)。图像形状 (n, 1, IMG, IMG)，像素范围约 [-1, 1]。"""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, CLASSES, n)
    x = np.full((n, IMG, IMG), -1.0, dtype="float32")
    for i, c in enumerate(y):
        if c == 0:                                   # 横条
            r = 5 + rng.randint(0, 3)
            x[i, r:r + 3, 2:14] = 1.0
        elif c == 1:                                 # 竖条
            col = 5 + rng.randint(0, 3)
            x[i, 2:14, col:col + 3] = 1.0
        elif c == 2:                                 # 方块
            r = 4 + rng.randint(0, 3)
            x[i, r:r + 6, r:r + 6] = 1.0
        elif c == 3:                                 # 十字
            x[i, 6:10, 2:14] = 1.0
            x[i, 2:14, 6:10] = 1.0
        elif c == 4:                                 # 棋盘
            x[i, ::2, ::2] = 1.0
        else:                                        # 对角线
            for k in range(IMG):
                x[i, k, k] = 1.0
    x += rng.normal(0, 0.15, x.shape).astype("float32")
    return x[:, None, :, :], y


x_train, y_train = make_dataset(N_TRAIN, seed=1)
x_val, y_val = make_dataset(N_VAL, seed=2)

print("训练集", x_train.shape, "标签分布", np.bincount(y_train, minlength=CLASSES))
print("验证集", x_val.shape, "标签分布", np.bincount(y_val, minlength=CLASSES))
print("随机猜的准确率 = 1/%d = %.3f" % (CLASSES, CHANCE))
```

**两个数据集用不同的种子生成**：验证集必须是模型没见过的样本，否则「验证准确率」只是
训练准确率的复读。这是标准流程里最容易被省掉、也最不能省的一步。

## 2. 残差块与网络

一个残差块的主分支是两组「卷积 + 批归一化」；**把输入直接加到输出上**，就是残差连接。
通道数变化时用一个 1×1 卷积把恒等分支投影到相同形状。

```{code-cell} ipython3
class Block(nn.Module):
    def __init__(self, channels, residual=True):
        super().__init__()
        self.residual = residual                      # 用来做对照实验
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(channels)

    def execute(self, x):
        h = nn.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.residual:
            h = h + x                                 # ← 残差连接
        return nn.relu(h)


class Net(nn.Module):
    def __init__(self, depth=6, channels=8, residual=True):
        super().__init__()
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(channels, residual) for _ in range(depth)])
        self.head = nn.Linear(channels, CLASSES)

    def execute(self, x):
        x = nn.relu(self.stem(x))
        x = self.blocks(x)
        return self.head(x.mean((2, 3)))              # 全局平均池化再分类
```

`residual` 这个开关就是整个对照实验的**唯一变量**。除了它，两个模型的结构、参数量、
数据、步数、种子全部相同。

## 3. 训练与评估循环

标准循环有两个容易写错的地方：

1. **评估时要切到 `eval()`**，训练时切回 `train()`。差别在 BatchNorm：训练时它用
   **当前 batch** 的统计量并更新运行统计量，评估时它用**累积的运行统计量**。忘记切换，
   验证结果会随 batch 组成抖动，而且会污染运行统计量。
2. **评估必须在 `no_grad` 语义下做**，否则会白白建图。

```{code-cell} ipython3
def evaluate(model, images, labels, batch=64):
    model.eval()                                      # ← 用运行统计量
    correct = 0
    for i in range(0, len(labels), batch):
        logits = model(jt.array(images[i:i + batch]))
        pred = logits.numpy().argmax(1)
        correct += int((pred == labels[i:i + batch]).sum())
    model.train()                                     # ← 切回训练模式
    return correct / len(labels)


def train(residual, depth=6, epochs=8, batch=32, lr=1e-2, seed=0):
    """训练一个模型，每个 epoch 后记录 (训练准确率, 验证准确率)。"""
    jt.set_global_seed(seed)
    model = Net(depth=depth, residual=residual)
    optimizer = nn.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        order = np.random.RandomState(epoch).permutation(N_TRAIN)
        for i in range(0, N_TRAIN, batch):
            idx = order[i:i + batch]
            logits = model(jt.array(x_train[idx]))
            loss = nn.cross_entropy(logits, jt.array(y_train[idx]))
            optimizer.step(loss)
        history.append((evaluate(model, x_train, y_train),
                        evaluate(model, x_val, y_val)))
    return model, history
```

**每个 epoch 都要在验证集上评估一次**，而不是只在最后测一次。原因在下一节的表里看得很
清楚：有个模型在中间**连续三个 epoch 卡住不动**，只看最终数字是发现不了的。

## 4. 对照实验：残差 vs 普通卷积

同一个网络，`residual=True` 与 `False` 各训一遍。

```{code-cell} ipython3
models, results = {}, {}
for residual in (True, False):
    t0 = time.time()
    models[residual], results[residual] = train(residual)
    print("residual=%-5s 训练用时 %5.1fs" % (residual, time.time() - t0))

print("\n%-6s %s" % ("epoch", " ".join("%6d" % e for e in range(8))))
for residual in (True, False):
    label = "残差  " if residual else "普通  "
    train_acc = " ".join("%6.2f" % h[0] for h in results[residual])
    val_acc = " ".join("%6.2f" % h[1] for h in results[residual])
    print("%s训练 %s" % (label, train_acc))
    print("%s验证 %s" % (label, val_acc))
print("\n随机猜 = %.3f" % CHANCE)
```

读这张表：

| | 最终训练准确率 | 最终验证准确率 | 过程 |
| --- | --- | --- | --- |
| **残差** | **0.80** | **0.72** | 稳步上升，第 4 个 epoch 就到 0.75 |
| **普通** | 0.56 | 0.42 | 第 2~4 个 epoch **卡在 0.31 不动**，之后才慢慢爬 |

- 加上残差连接，验证准确率从 **0.42 提到 0.72**。同样的数据、同样的步数、同样的种子，
  **唯一的变化是那一行加法**。
- 普通网络在中间三个 epoch 完全停滞——那不是「学得慢」，是**卡住了**。

```{code-cell} ipython3
res_only = results[True]
plain = results[False]
assert res_only[-1][1] > 0.6, "残差网络应该明显超过随机猜"
assert res_only[-1][1] > plain[-1][1] + 0.15, "残差应当明显优于普通卷积"
assert res_only[-1][1] > res_only[0][1], "残差网络的验证准确率应该上升"

frozen = [plain[e][1] for e in (1, 2, 3)]
assert len(set(frozen)) == 1, "普通网络应当出现连续多个 epoch 停滞"
print("验证准确率：残差 %.2f vs 普通 %.2f，差 %.2f"
      % (res_only[-1][1], plain[-1][1], res_only[-1][1] - plain[-1][1]))
print("普通网络在第 2~4 个 epoch 停滞在 %.2f（三个 epoch 完全不变）" % frozen[0])
```

## 5. 为什么残差有用——以及一个和预期相反的读数

最常见的解释是「残差连接缓解梯度消失」。**这个说法值得量一下再接受。**

```{code-cell} ipython3
def grad_norm(depth, residual):
    """loss 对输入的梯度模长：有多少信号能传回第一层。"""
    jt.set_global_seed(3)
    net = Net(depth=depth, residual=residual)
    x = jt.randn((8, 1, IMG, IMG))
    y = jt.randint(0, CLASSES, (8,)).astype("int32")
    loss = nn.cross_entropy(net(x), jt.array(y))
    return float(np.linalg.norm(jt.grad(loss, x).numpy()))


print("%-6s %-14s %-14s" % ("深度", "残差", "普通"))
for depth in (2, 6, 12, 20):
    print("%-6d %-14.3e %-14.3e"
          % (depth, grad_norm(depth, True), grad_norm(depth, False)))
```

**结果和「缓解梯度消失」的预期相反**：深度越大，**普通**网络的输入梯度反而越大（深度 20
时 `1.28e+01`，是残差网络的 47 倍）。所以在这个配置下，「残差让梯度更大」并不是它有效
的原因——**至少不是以梯度模长的方式**。

那到底为什么？读第 4 节那张表会更实在：

* 残差块只需要学**修正量**，恒等路径可以直接把信息送过去。深度堆得越多，这个好处越明显；
* 普通网络要把恒等映射也**学出来**，每一层都得准确拟合，堆叠起来就容易被卡住——
  表里那三个停滞的 epoch 就是它的样子。

这个例子值得记住的地方不是「残差是什么」，而是**不要直接把教科书的因果链搬进自己的
结论里**。梯度模长是一个可以量的量，量出来是反的，就得换一个解释。
（顺带一提：每个块里的 BatchNorm 也会显著改变梯度尺度，这正是它在这里盖过「消失」效应的
原因之一。）

## 6. 训练状态：`state_dict`、运行统计量与 `eval()`

`state_dict` 存取的不只是可训练参数，还包括 BatchNorm 的**运行统计量**。少了它，
恢复出来的模型在 `eval()` 下会给出不同的输出——而且**不会报错**。

```{code-cell} ipython3
trained = models[True]

trained.eval()
expected = trained(jt.array(x_val)).numpy()

state = {name: np.array(value.numpy(), copy=True)
         for name, value in trained.state_dict().items()}
print("state_dict 里有 %d 项，含 BatchNorm 的运行统计量：" % len(state))
print([k for k in state if "running" in k or "num_batches" in k][:6])

restored = Net(depth=6, residual=True)
restored.load_parameters(state)
restored.eval()
actual = restored(jt.array(x_val)).numpy()

assert np.allclose(actual, expected, atol=1e-5), "恢复后的输出应当逐位接近"
print("\n恢复后输出一致:", bool(np.allclose(actual, expected, atol=1e-5)))
print("验证准确率（恢复后）: %.2f" % evaluate(restored, x_val, y_val))
```

两件容易忘的事：

* **只存权重不存运行统计量**，恢复后在 `eval()` 下输出会变，而且没有任何报错；
* **忘记 `eval()`** 就评估，BatchNorm 会用当前 batch 的统计量，小 batch 下结果会剧烈抖动。

想连优化器状态一起续训，见 [检查点与断点续训](checkpointing.md)。

## 7. 同一件事，用 PyTorch API 再写一遍

图像分类的写法在两种前端里几乎逐字对应。**激活必须在创建任何张量之前**，所以这一节
自成一段。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)


class BlockT(tnn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = tnn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = tnn.BatchNorm2d(channels)
        self.conv2 = tnn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = tnn.BatchNorm2d(channels)

    def forward(self, x):
        h = tnn.functional.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return tnn.functional.relu(h + x)


net_t = tnn.Sequential(
    tnn.Conv2d(1, 8, 3, padding=1),
    BlockT(8),
    tnn.AdaptiveAvgPool2d(1),        # 全局平均池化，替代手写 .mean((2, 3))
)
head_t = tnn.Linear(8, CLASSES)

x_t = torch.tensor(x_train[:16])
y_t = torch.tensor(y_train[:16].astype("int64"))
optim_t = torch.optim.Adam(list(net_t.parameters()) + list(head_t.parameters()), lr=1e-2)

for _ in range(20):
    optim_t.zero_grad()
    loss_t = tnn.functional.cross_entropy(head_t(net_t(x_t).reshape(16, -1)), y_t)
    loss_t.backward()
    optim_t.step()

print("torch 侧损失: %.4f" % float(loss_t))
print("卷积权重的梯度拿到了吗:", net_t[0].weight.grad is not None)
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 前向方法名 | `execute` | `forward` |
| 全局平均池化 | `x.mean((2, 3))` | `nn.AdaptiveAvgPool2d(1)` |
| 训练一步 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |
| 标签 dtype | 整数即可 | `CrossEntropyLoss` 要 `int64` |

## 8. 检查清单

```{code-block} text
1. 训练集与验证集必须分开     —— 不同种子生成；否则「验证」只是复读训练准确率
2. 每个 epoch 都评估          —— 第 4 节那个停滞三个 epoch 的模型，只看最终值发现不了
3. 评估前切 eval()            —— BatchNorm 用运行统计量；忘了会抖动并污染统计量
4. state_dict 要连统计量一起存 —— 只存权重会让 eval() 下的输出静默改变
5. 对照实验只改一个变量       —— 结构、参数、数据、步数、种子全同，只动那一行加法
6. 有常数基线                —— 这里是「随机猜 = 0.167」；0.42 只有配上它才有意义
7. 教科书因果链要量过再引用    —— 第 5 节的梯度模长量出来是反的
```

第 2 条和第 6 条是通用纪律，不限于残差网络：**要能看到过程的形状，要有一个能对照的
常数基线。** [VAE](vae.md) 里「KL = 0 时采样图看着不错但其实塌了」、
[WGAN-GP](wgan_gp.md) 里「W 为负时不要相信它的数值」，说的是同一件事。

## 接下来

* [循环网络](rnn.md)：另一类结构选择，以及怎么用可测的任务验证它。
* [检查点与断点续训](checkpointing.md)：`state_dict` 之外还要存什么才能真的续训。
* [数据加载与预处理](data_pipeline.md)：把这里的合成数据换成真实数据集。
* 回到 [教程总目录](README.md)。
