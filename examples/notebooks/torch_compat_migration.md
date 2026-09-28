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

# 把已有 PyTorch 脚本迁到 Jittor

上一篇讲了怎么激活兼容层。这一篇讲**迁移时真正会绊倒你的地方**：同样是激活之后，
哪些代码原样能跑，哪些必须动，以及为什么。

先看一个可以当场验证的结论：同一个计算用两套写法，数值一致。这能在你怀疑「是不是
兼容层算错了」时快速自证。

```{code-cell} ipython3
import numpy as np
import jittor as jt

jt.flags.use_cuda = 0

# 写法一：原生 Jittor
w_native = jt.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
x_native = jt.array(np.array([[1.0], [1.0]], dtype="float32"))
native_out = (w_native @ x_native).numpy()

# 写法二：激活兼容层后，用 PyTorch 的写法
from jittor.compat.shim import activate
activate()
import torch

w_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x_torch = torch.tensor([[1.0], [1.0]])
torch_out = (w_torch @ x_torch).numpy()

print("原生   :", native_out.ravel())
print("torch  :", torch_out.ravel())
assert np.allclose(native_out, torch_out), "两种写法结果不一致"
print("一致 ✅")
```

两套前端共享同一个 Var/Op 运行时，所以数值一致；但它们的 **Tensor、Parameter、Module
类型是分开的**：

```{code-cell} ipython3
print("torch.is_tensor(jt.array([1.0])) =", torch.is_tensor(jt.array([1.0])))
print("torch.__version__ =", torch.__version__, "（这是 Jittor 的实现，不是真 PyTorch）")
```

这就是「别混用」的含义——在同一个模型里一半用 `jt.nn`、一半用 `torch.nn`，类型对不上。

## 1. 先看哪些**不用改**

迁移前最容易犯的错是**改多了**。兼容层实现的是一整套常用 API，不是一层薄薄的垫片。
先把覆盖面确认一遍，省掉无谓的改写。

```{code-cell} ipython3
import torch.nn as tnn
import torch.optim as topt
import torch.nn.functional as F

for name in ("nn", "optim", "nn.functional", "utils.data", "autocast", "no_grad",
             "save", "load", "device", "float32", "manual_seed", "Tensor"):
    obj, ok = torch, True
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            ok = False
            break
    print("  %-16s %s" % (name, "有" if ok else "**没有**"))
```

下面这些东西**原样能跑**，一行都不用改。这段代码本身就是证据。

```{code-cell} ipython3
torch.manual_seed(0)
net = tnn.Sequential(tnn.Linear(8, 16), tnn.ReLU(), tnn.Linear(16, 3))
x = torch.randn(4, 8)

# no_grad
with torch.no_grad():
    y = net(x)
print("no_grad 生效（不需要梯度）:", not y.requires_grad)

# eval / train 切换
net.eval()
print("eval()  -> training =", net.training)
net.train()
print("train() -> training =", net.training)

# item / detach
print("item()   ->", round(float(y.sum().item()), 4))
print("detach() -> requires_grad =", y.detach().requires_grad)

# dtype / device
print("dtype    ->", y.dtype, " device ->", y.device)
```

DataLoader、损失函数、存盘也都在。

```{code-cell} ipython3
from torch.utils.data import DataLoader, TensorDataset

ds = TensorDataset(torch.randn(20, 8), torch.randint(0, 3, (20,)))
loader = DataLoader(ds, batch_size=8, shuffle=True)
batches = list(loader)
print("DataLoader: %d 个 batch，第一个形状 %s"
      % (len(batches), [tuple(t.shape) for t in batches[0]]))

crit = tnn.CrossEntropyLoss()
loss = crit(net(x), torch.randint(0, 3, (4,)))
loss.backward()
print("CrossEntropyLoss + backward: 损失 %.4f" % float(loss))

import io
buf = io.BytesIO()
torch.save(net.state_dict(), buf)
buf.seek(0)
try:
    back = torch.load(buf, weights_only=True)
except TypeError:
    buf.seek(0)
    back = torch.load(buf)
print("torch.save / torch.load 往返: %d 字节，键 %s"
      % (buf.getbuffer().nbytes if hasattr(buf, "getbuffer") else 0, list(back)[:2]))
```

一句话总结这一节：**要改的不是 API，而是那几处「语义不同」的地方。**

## 2. 必须改的五处

### 2.1 入口：必须先激活

```{code-block} python
import torch                          # ❌ 拿到的是真的 PyTorch
from jittor.compat.shim import activate
activate(); import torch              # ✅ 拿到的是 Jittor 的实现

# 或者整进程生效：
#   JITTOR_TORCH_SHIM=1 python train.py
```

`import jittor as torch` **不是**激活，它只是一个本地别名，底层还是原生 Jittor。

### 2.2 设备：全局标志 vs 每张量自带设备

这是最容易归因错误的一处。PyTorch 里设备跟着张量走（`x.cuda()`、`x.to("cuda")`）；
Jittor 用一个**全局标志**把整张计算图放到设备上。

```{code-block} python
# Jittor 原生：全局生效，且要在建变量之前决定
jt.flags.use_cuda = 1

# 兼容层里 .cuda()/.to(...) 仍然可用，按 PyTorch 的语义解释
model = model.cuda()
```

不先弄清这点，性能问题会被归到错误的位置。详见 [设备与驻留](device_placement.md)。

### 2.3 优化器：原生与兼容层语义不同

```{code-block} python
# 原生 Jittor：一步包含 清梯度 + 反向 + 更新，所以要把 loss 传进去
optim.step(loss)

# 兼容层（PyTorch 语义）：保持你原来的三行
opt.zero_grad()
loss.backward()
opt.step()
```

如果你把原生写法里的 `optim.step(loss)` 带进兼容层，或者反过来，都会出问题——先确认
当前用的是哪一套前端。

### 2.4 随机性：可复现要显式播种

```{code-block} python
jt.set_global_seed(0)   # 原生
torch.manual_seed(0)    # 兼容层
```

没有种子时，Jittor 的即时编译与算子调度会让两次运行的数值细节不同，对比实验必须播种。

### 2.5 同名不同签名：`argmax` 返回的是二元组

这一类差异不会在 import 时暴露，只会在你取返回值时出错。Jittor 原生的
`argmax(dim=...)` 返回 `(索引, 取值)`；PyTorch 的 `torch.argmax` 只返回索引，返回二元组
的是 `torch.max(dim=...)`。顺序也更接近后者。

```{code-cell} ipython3
row = jt.array([[1.0, 3.0, 2.0]])

idx, val = row.argmax(dim=1)          # 原生：先索引，后取值
print("原生 argmax  ->", idx.numpy(), val.numpy())

print("torch.argmax ->", torch.argmax(torch.tensor([[1.0, 3.0, 2.0]]), dim=1).numpy())

tv, ti = torch.max(torch.tensor([[1.0, 3.0, 2.0]]), dim=1)
print("torch.max    ->", tv.numpy(), ti.numpy(), "（先取值，后索引）")
```

所以在原生代码里写 `pred = logits.argmax(dim=1)` 会拿到一个元组，后面 `.numpy()` 就会
报 `'tuple' object has no attribute 'numpy'`；正确写法是 `pred, _ = ...`。这类地方只能
靠跑一遍发现，所以迁移后**先跑一次小规模的前向**，不要直接上大训练。

## 3. 跨前端的存档：名字要对上，报错方式不对称

一个很自然的期待是「用兼容层跑一遍，把权重存下来，再用原生代码读」。**这件事是可以
做的**，但有一处容易踩空。

先看好的一面：如果两边的**层命名一致**，权重能双向流动。`nn.Sequential` 在两个前端里
用的是同一套整数键名。

```{code-cell} ipython3
from jittor import nn as jnn

native_seq = jnn.Sequential(jnn.Linear(8, 16), jnn.ReLU(), jnn.Linear(16, 3))
torch_seq = tnn.Sequential(tnn.Linear(8, 16), tnn.ReLU(), tnn.Linear(16, 3))

print("原生  Sequential 的键:", list(native_seq.state_dict()))
print("torch Sequential 的键:", list(torch_seq.state_dict()))

# 把原生权重写成一个可辨认的值，再看能不能传过去
for v in native_seq.state_dict().values():
    v.update(jt.ones(v.shape) * 0.5)

torch_seq.load_state_dict(native_seq.state_dict())
w = torch_seq[0].weight.numpy()
print("\ntorch <- 原生: 成功，权重确实变成了 0.5:", bool(np.allclose(w, 0.5)))
```

反方向也一样能过。

```{code-cell} ipython3
# 把 torch 侧的权重设成 0.25。原生 Var 有 .update()，先写进一个原生模块再传过去，
# 避免直接改 torch 侧的对象。
src = jnn.Sequential(jnn.Linear(8, 16), jnn.ReLU(), jnn.Linear(16, 3))
for v in src.state_dict().values():
    v.update(jt.ones(v.shape) * 0.25)
torch_seq.load_state_dict(src.state_dict())

native_seq2 = jnn.Sequential(jnn.Linear(8, 16), jnn.ReLU(), jnn.Linear(16, 3))
native_seq2.load_state_dict(torch_seq.state_dict())      # 原生 <- torch
print("原生 <- torch: 权重变成了 0.25:",
      bool(np.allclose(native_seq2[0].weight.numpy(), 0.25)))
```

**但前提是键名一致。** 换成自定义 `Module`（键名是 `fc1.weight` 这样的属性名），
两边就对不上了。这时两个方向的**报错方式完全不同**：

```{code-cell} ipython3
class NativeMLP(jt.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = jt.nn.Linear(8, 16)
        self.fc2 = jt.nn.Linear(16, 3)

    def execute(self, x):
        return self.fc2(jt.nn.relu(self.fc1(x)))


plain = NativeMLP()
print("原生 MLP 的键:", list(plain.state_dict()))

# 方向一：torch 载入原生的键 —— 会抛异常（严格）
try:
    torch_seq.load_state_dict(plain.state_dict())
    print("torch <- 原生: 居然成功了")
except Exception as exc:
    print("\ntorch <- 原生: %s" % type(exc).__name__)
    print("  ", str(exc).splitlines()[0])
```

```{code-cell} ipython3
# 方向二：原生载入 torch 的键 —— 不抛异常，只写日志
before = plain.fc1.weight.numpy().copy()
result = plain.load_state_dict(torch_seq.state_dict())
after = plain.fc1.weight.numpy()
print("原生 <- torch: 返回值 =", result, "（没有异常）")
print("权重被替换了吗:", not bool(np.allclose(before, after)))
print("上面这个 False 就是问题所在：4 个键一个都没对上，调用却正常返回了")
```

两个方向的差别是关键：

| | 行为 | 后果 |
| --- | --- | --- |
| `torch.load_state_dict` | **抛异常**，列出 missing / unexpected 键 | 立刻发现 |
| 原生 `load_state_dict` | **不抛异常**，只写日志，返回 `None` | **静默地什么都没载入** |

上面第二个方向里，原生模型的 4 个键一个都没对上，但调用**正常返回**了。这正是
[检查点与断点续训](checkpointing.md) 第 4 节讲的静默失败——只是当它发生在**跨前端**
的时候更容易被忽略，因为你会先入为主地以为「都是 Jittor，应该能直接读」。

要判断到底载入了几个参数，唯一可靠的办法是**载入前后各取一次权重做比较**，而不是
看有没有报错。另外注意上面的键名差异**不是类型问题**——`nn.Sequential` 那条路径已经
证明了两种类型的权重是可以互相传的，对不上的只是**名字**。

## 4. 代价：兼容层不是免费的

兼容层是一层 Python 转发：每一次算子调用都要经过一次额外的分发，才能落到和原生写法
完全相同的核上。对小模型来说，这部分开销的**占比**会很明显。

```{code-cell} ipython3
import time

RNG = np.random.RandomState(0)
XA = RNG.randn(64, 8).astype("float32")
YA = RNG.randint(0, 3, 64)
XT, YT = torch.tensor(XA), torch.tensor(YA)


def native_train(steps, seed):
    jt.set_global_seed(seed)
    net = NativeMLP()
    optim = jt.nn.Adam(net.parameters(), lr=3e-2)
    t0 = time.time()
    for _ in range(steps):
        loss = jt.nn.cross_entropy(net(jt.array(XA)), jt.array(YA))
        optim.step(loss)
    return time.time() - t0, float(loss.item())


def torch_train(steps, seed):
    torch.manual_seed(seed)
    net = tnn.Sequential(tnn.Linear(8, 16), tnn.ReLU(), tnn.Linear(16, 3))
    optim = topt.Adam(net.parameters(), lr=3e-2)
    crit = tnn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(steps):
        optim.zero_grad()
        loss = crit(net(XT), YT)
        loss.backward()
        optim.step()
    return time.time() - t0, float(loss.item())


native_train(20, 0)              # 先各跑一轮，把即时编译和缓存焐热
torch_train(20, 0)

STEPS = 200
print("同一份数据、同一个结构、同样 200 步：\n")
print("%-4s %-8s %-9s %-10s" % ("轮", "前端", "秒", "最终损失"))
for rnd in range(2):
    t, l = native_train(STEPS, rnd)
    print("%-4d %-8s %-9.2f %-10.4f" % (rnd, "原生", t, l))
    t, l = torch_train(STEPS, rnd)
    print("%-4d %-8s %-9.2f %-10.4f" % (rnd, "torch", t, l))
```

在这个玩具模型上，兼容层大约是原生的 **2 倍出头**（实测 0.19s 对 0.44s，200 步），
而**最终损失是一样的**（数值一致是它的目标，速度不是）。两点要注意：

* 这是**小模型**的数字。开销基本是每次算子调用的固定转发成本，模型一大，它占的比例
  会迅速下降；不要把这个倍数直接套到真实训练上。
* 所以兼容层的定位是**迁移工具，不是性能路径**。想榨性能就按原生写法重写热点，或者
  用原生算子改写关键模块；想尽快让已有代码跑起来、先把结果对上，就用它。

## 5. 迁移检查表

| 检查项 | 动作 |
| --- | --- |
| 入口 | 确认在 import torch **之前**激活；不要混用别名激活 |
| 覆盖面 | 先跑一遍本文第 1 节的自检，`no_grad`/`eval`/`DataLoader` 等**不用改**，别白改 |
| 设备 | 用全局 `jt.flags.use_cuda`，或在兼容层里按 PyTorch 语义 `.cuda()` |
| 优化器 | 兼容层保持 `zero_grad/backward/step`；原生才是 `step(loss)` |
| 类型 | 不要在同一模型里混用 `jt.nn.*` 与 `torch.nn.*` |
| 随机性 | 对拍或复现实验前显式播种 |
| 同名签名 | `argmax(dim)` 等原生算子可能返回元组；迁移后先跑一次小规模前向 |
| 跨前端存档 | 键名必须一致；**原生侧载入失败会静默返回**，载入后要自己核对权重 |
| 速度 | 兼容层在小模型上约 2 倍开销；它是迁移工具，不是性能路径 |
| 数值 | 与官方 PyTorch 的差异量级见生态对拍报告，不要假设逐位相同 |

## 接下来

* [用 PyTorch API 写 Jittor](torch_compat.md)：激活方式、张量与训练的完整入门。
* [检查点与断点续训](checkpointing.md)：原生 `load_state_dict` 静默失败的完整机理。
* 生态库（transformers / diffusers / PEFT 等）的接入与验证结论，见
  [用 PyTorch API 写 Jittor 文档](https://github.com/Jittor/jittor/blob/master/docs/compatibility/torch.md)。
* 回到 [教程总目录](README.md)。
