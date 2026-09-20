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
类型是分开的**：`torch.is_tensor(jt.array([1.0]))` 是 `False`。这就是「别混用」的
含义——在同一个模型里一半用 `jt.nn`、一半用 `torch.nn`，类型对不上。

## 必须改的四处

### 1. 入口：必须先激活

```{code-block} python
import torch                          # ❌ 拿到的是真的 PyTorch
from jittor.compat.shim import activate
activate(); import torch              # ✅ 拿到的是 Jittor 的实现

# 或者整进程生效：
#   JITTOR_TORCH_SHIM=1 python train.py
```

`import jittor as torch` **不是**激活，它只是一个本地别名，底层还是原生 Jittor。

### 2. 设备：全局标志 vs 每张量自带设备

这是最容易归因错误的一处。PyTorch 里设备跟着张量走（`x.cuda()`、`x.to("cuda")`）；
Jittor 用一个**全局标志**把整张计算图放到设备上。

```{code-block} python
# Jittor 原生：全局生效，且要在建变量之前决定
jt.flags.use_cuda = 1

# 兼容层里 .cuda()/.to(...) 仍然可用，按 PyTorch 的语义解释
model = model.cuda()
```

不先弄清这点，性能问题会被归到错误的位置。详见 [设备与驻留](device_placement.md)。

### 3. 优化器：原生与兼容层语义不同

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

### 4. 随机性：可复现要显式播种

```{code-block} python
jt.set_seed(0)      # 原生
torch.manual_seed(0)  # 兼容层
```

没有种子时，Jittor 的即时编译与算子调度会让两次运行的数值细节不同，对比实验必须播种。

### 5. 同名不同签名：`argmax` 返回的是二元组

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

## 迁移检查表

| 检查项 | 动作 |
| --- | --- |
| 入口 | 确认在 import torch **之前**激活；不要混用别名激活 |
| 设备 | 用全局 `jt.flags.use_cuda`，或在兼容层里按 PyTorch 语义 `.cuda()` |
| 优化器 | 兼容层保持 `zero_grad/backward/step`；原生才是 `step(loss)` |
| 类型 | 不要在同一模型里混用 `jt.nn.*` 与 `torch.nn.*` |
| 随机性 | 对拍或复现实验前显式播种 |
| 同名签名 | `argmax(dim)` 等原生算子可能返回元组；迁移后先跑一次小规模前向 |
| 数值 | 与官方 PyTorch 的差异量级见生态对拍报告，不要假设逐位相同 |

## 接下来

* [用 PyTorch API 写 Jittor](torch_compat.md)：激活方式、张量与训练的完整入门。
* 生态库（transformers / diffusers / PEFT 等）的接入与验证结论，见
  [用 PyTorch API 写 Jittor 文档](https://github.com/Jittor/jittor/blob/master/docs/compatibility/torch.md)。
* 回到 [教程总目录](README.md)。
