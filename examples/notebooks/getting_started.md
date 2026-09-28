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

# 从零开始：装好、跑通、训练第一个模型

这一篇是**唯一的入门起点**。读完并亲手跑完里面的单元格，你就完成了从安装到训练的
完整一圈，也知道了接下来该读哪一篇。

不需要任何深度学习背景，只要会写 Python。**全程 CPU 就能跑完**，有显卡只是更快。

本篇的每个代码单元都是可以直接执行的；只有最后一节「换成真实数据」需要联网下载
MNIST，因此被标记为不参与仓库的离线冒烟测试。

## 0. 安装，然后自检

先装好解释器依赖，再装 Jittor 本体：

```{code-block} bash
# 需要 Python >= 3.7，以及一个 C++ 编译器（g++ >= 5.4 或 clang >= 8）
python -m pip install jittor

# 可选：如果你希望 Jittor 用 GPU，另外准备好 nvcc，不必现在做
# python -m pip install jittor -U     # 升级

# 装完立刻自检。它会真的编译一个算子并执行一次
python -m jittor.selftest
```

`selftest` 打印出结果且没有报错，就说明**即时编译（JIT）链路是通的**——这是 Jittor
和别的框架最不一样的地方：它的算子在你第一次用到时才编译。下面 Import 单元第一次
运行时也会做同样的事，比普通的 `import` 慢，这是正常的，第二次就快了。

```{code-cell} ipython3
import jittor as jt

print("jittor version:", jt.__version__)
print("backend in use:", "cuda" if jt.flags.use_cuda else "cpu")
```

`jt.flags.use_cuda` 是**全局**开关：有可用的加速器时 Jittor 会自动打开它。后面
[设备与驻留](device_placement.md) 会解释它和「每个张量各自带设备」的区别——这是
Jittor 与 PyTorch 最容易踩错的一处差异。

## 1. 第一个 Var

`Var` 是 Jittor 的基本数据类型，相当于其它框架里的张量。它长得像 NumPy 数组，
但有个关键不同：**计算是异步的**。写下一个表达式并不会立刻算出数值，它只是进入
计算图；等你真的需要数值时（`.numpy()` 或 `.sync()`），Jittor 才把整张图算掉。

```{code-cell} ipython3
a = jt.array([1, 2, 3], dtype="float32")
b = a * 2 + 1

print("a =", a)            # 打印 Var 本身，看到的是形状和类型
print("b =", b)
print("b.numpy() =", b.numpy())   # 取数值：这一步才会同步等待计算完成
```

`b` 的计算其实被推迟到了 `.numpy()`。异步是 Jittor 能做算子融合与调优的前提；代价是
「什么时候真正算」不由你写代码的顺序决定，需要明确的时机就显式 `sync()`。

```{code-cell} ipython3
b.sync()          # 强制此刻就把 b 算出来
print("after sync, b =", b.numpy())
```

算子的写法与 NumPy 一致，并且每个 `jt.xxx(x, ...)` 都有一个等价的 `x.xxx(...)`
方法写法，两者是同一个实现：

```{code-cell} ipython3
c = jt.array([4, 5, 6], dtype="float32")
print("jt.max(c)      =", jt.max(c).numpy())
print("c.max()        =", c.max().numpy())
print("c.add(a)       =", c.add(a).numpy())          # 等价于 jt.add(c, a)
```

## 2. 自动求导

Jittor 会为参与计算、且需要梯度的 Var 记录计算图。用 `jt.grad(y, x)` 求 `y` 对 `x`
的梯度——它返回的是一个 Var，同样需要 `.numpy()` 才拿到数值。

```{code-cell} ipython3
x = jt.array([2.0, 3.0], dtype="float32")
y = (x * x).sum()                  # y = x0^2 + x1^2

g = jt.grad(y, x)                  # dy/dx = 2x
print("y      =", y.numpy())
print("dy/dx  =", g.numpy(), "（期望 [4, 6]）")
```

手写 `jt.grad` 能让你看清机制，但训练时你不会这么写：优化器会替你调用它。下一节就是
训练循环的最小形态。

## 3. 定义并训练一个模型

模型继承 `jt.Module`，在 `__init__` 里声明层，在 `execute` 里描述前向计算。**方法名是
`execute` 而不是 `forward`**——这是 Jittor 与 PyTorch 的一个书写差异。

```{code-cell} ipython3
import numpy as np

jt.set_seed(0)
np.random.seed(0)

class Net(jt.Module):
    def __init__(self):
        self.fc1 = jt.nn.Linear(1, 16)   # 输入 1 维，隐藏层 16 维
        self.act = jt.nn.ReLU()
        self.fc2 = jt.nn.Linear(16, 1)   # 输出 1 维

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))

net = Net()
optim = jt.optim.SGD(net.parameters(), lr=0.1)
```

`jt.optim.SGD` 是新家；`jt.nn.SGD` 是同一个对象的兼容别名（`jt.nn.SGD is jt.optim.SGD`
为真），两种写法都对。

下面用一条已知的直线 `y = 3x + 2` 造数据，看模型能不能把参数学回来。这个例子不联网、
不用显卡，几十秒就能跑完。

```{code-cell} ipython3
def batches(steps=200, batch_size=32):
    for _ in range(steps):
        x = np.random.rand(batch_size, 1).astype("float32")
        yield jt.array(x), jt.array(3.0 * x + 2.0)   # y = 3x + 2

losses = []
for x, y in batches():
    pred = net(x)                       # 前向
    loss = jt.nn.mse_loss(pred, y)      # 损失
    optim.step(loss)                    # 梯度 + 更新，一步完成
    losses.append(float(loss.item()))

print("step   0 loss =", round(losses[0], 4))
print("step 199 loss =", round(losses[-1], 4))
assert losses[-1] < losses[0], "训练没有让损失下降，请检查上面的代码"
```

这里最值得注意的是 `optim.step(loss)`。PyTorch 里你需要写三行：

```{code-block} python
# PyTorch 的写法，供对照——Jittor 里不需要这样
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Jittor 把「清空梯度、反向传播、更新参数」合成了一次 `optim.step(loss)`：梯度属于
这一次计算图，用完即弃，所以没有 `zero_grad()`。这也是为什么 `step` 需要你把
**损失**传进去，而不是像 PyTorch 那样只要 `step()`。

最后确认一下参数确实学到了接近真实的斜率和截距：

```{code-cell} ipython3
w = net.fc1.weight.numpy().reshape(-1)
print("隐含层权重（前 4 个）:", np.round(w[:4], 3))
# 用训好的模型预测：x = 5 时答案应接近 3*5+2 = 17
print("预测 x=5 ->", float(net(jt.array([[5.0]], dtype="float32")).item()), "（期望约 17）")
```

## 4. 换成真实数据：MNIST

把上面的骨架换成手写数字识别，只多了「加载数据」和「评估准确率」两步。这一步需要
联网下载数据集，所以下面几个单元标了 `network`，**不参与仓库的离线冒烟测试**；
你可以在自己的 notebook 里从头执行。

![MNIST 手写数字示例](figs/mnist.png)

```{code-cell} ipython3
:tags: [network, skip-execution]

from jittor.dataset.mnist import MNIST
import jittor.transform as trans

batch_size = 64
train_loader = MNIST(train=True, transform=trans.Resize(28)).set_attrs(
    batch_size=batch_size, shuffle=True)
val_loader = MNIST(train=False, transform=trans.Resize(28)).set_attrs(
    batch_size=batch_size, shuffle=False)

for inputs, targets in val_loader:
    print("inputs :", inputs.shape, inputs.dtype)   # [N, 3, 28, 28]
    print("targets:", targets.shape, targets.dtype)  # [N, 1]
    break
```

卷积网络把上面的一层线性换成两层卷积加池化即可。注意这里的 MNIST 被统一成 3 通道，
所以第一层卷积的输入通道是 3。

```{code-cell} ipython3
:tags: [network, long-running, skip-execution]

class ConvNet(jt.Module):
    def __init__(self):
        self.conv1 = jt.nn.Conv(3, 32, 3, 1)
        self.conv2 = jt.nn.Conv(32, 64, 3, 1)
        self.bn = jt.nn.BatchNorm(64)
        self.pool = jt.nn.Pool(2, 2)
        self.relu = jt.nn.ReLU()
        self.fc1 = jt.nn.Linear(64 * 12 * 12, 256)
        self.fc2 = jt.nn.Linear(256, 10)   # 10 个数字

    def execute(self, x):
        x = self.relu(self.conv1(x))       # [N, 3, 28, 28] -> [N, 32, 26, 26]
        x = self.relu(self.bn(self.conv2(x)))  # -> [N, 64, 24, 24]
        x = self.pool(x)                   # -> [N, 64, 12, 12]
        x = jt.reshape(x, [x.shape[0], -1])
        return self.fc2(self.relu(self.fc1(x)))

model = ConvNet()
loss_fn = jt.nn.CrossEntropyLoss()
optim = jt.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

for epoch in range(2):
    model.train()
    for inputs, targets in train_loader:
        loss = loss_fn(model(inputs), targets)
        optim.step(loss)

    model.eval()
    correct = total = 0
    for inputs, targets in val_loader:
        pred, _ = model(inputs).argmax(dim=1)   # Jittor 返回 (索引, 取值)
        correct += int((pred.numpy() == targets.numpy().reshape(-1)).sum())
        total += inputs.shape[0]
    print("epoch %d  val acc = %.4f" % (epoch, correct / total))
```

`argmax(dim=1)` 取每行 10 个分数里最大的那个下标，就是预测的数字；`model.eval()` 与
`model.train()` 的切换会影响 BatchNorm 和 Dropout 的行为，与 PyTorch 一致。

**注意返回值的差异**：Jittor 的 `argmax(dim=...)` 返回的是 `(索引, 取值)` 二元组，
而 PyTorch 的 `torch.argmax` 只返回索引（返回二元组的是 `torch.max(dim=...)`）。所以
上面写的是 `pred, _ = ...`。这类「同名不同签名」的地方是迁移时最容易漏的。

有显卡的话，只要**在建立任何模型和变量之前**加一行，整个计算图就会搬到显卡上：

```{code-cell} ipython3
:tags: [cuda, skip-execution]

jt.flags.use_cuda = 1     # 全局生效；没有可用 GPU 时不要打开
print("now on:", "cuda" if jt.flags.use_cuda else "cpu")
```

## 5. 接下来读什么

你已经会：造 Var、求梯度、定义 `Module`、写训练循环。往后有两条线，按你的情况选。

* **继续用原生 Jittor**：先读 [算子与 Var](basics.md) 补齐细节，再读
  [设备与驻留](device_placement.md)（**建议在训练之前读**，否则性能问题会归因到
  错误的地方），然后 [模型定义与训练](example.md)。之后按学习路径进入
  [训练实战](README.md)。
* **手上已有 PyTorch 代码**：走 [用 PyTorch API 写 Jittor](torch_compat.md)，
  它讲怎么原地激活兼容层，让现有的 `torch` 代码不改一行跑在 Jittor 上。

完整的阅读顺序和「每篇教什么、要不要显卡」，都在
[教程总目录](README.md) 里。
