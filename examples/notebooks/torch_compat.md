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

# 用 PyTorch API 写 Jittor

如果你手上已经有一份 PyTorch 代码，**不必先把它翻译成 Jittor 的写法**。Jittor 提供了
一层 Torch 兼容前端：激活之后，`import torch` 拿到的就是 Jittor 的实现，`.py` 文件里
的 `torch`、`torch.nn`、`torch.optim` 照常工作，算子跑在 Jittor 的运行时上。

这一篇讲清楚三件事：**怎么激活**、**激活之后什么变了**、**哪些写法必须改**。
全部内容 CPU 就能跑完，不需要显卡。

## 1. 激活：必须在 import torch 之前

兼容层是**显式启动**的，因为普通的 `import jittor as jt` 必须保持原生行为、不占用
`torch` 这个名字。

```{code-cell} ipython3
import jittor as jt

jt.flags.use_cuda = 0            # 本篇全程用 CPU；有显卡时可以去掉这一行

from jittor.compat.shim import activate
activate()                       # 第一次激活会部署并编译兼容层，稍慢

import torch
print("torch 来自:", type(torch).__module__ + "." + type(torch).__name__)
print("torch.__version__ =", torch.__version__)
```

激活有两种等价方式，选一种即可：

```{code-block} bash
# 方式一：在代码里激活（上面用的就是这种），适合 notebook 和单个脚本
#   from jittor.compat.shim import activate; activate()

# 方式二：把它装成当前环境里的 torch，之后所有进程都不用改代码
jittor-torch-shim              # 部署到当前环境
jittor-torch-shim --check      # 校验已部署的文件

# 或者用环境变量激活
JITTOR_TORCH_SHIM=1 python train.py
```

环境变量与部署包适合「一整个项目都要跑在 Jittor 上」；`activate()` 适合「只想让这一段
跑在 Jittor 上」。两者**在创建任何张量之前**选定即可，之后不要来回切换。

## 2. 张量、模型、自动求导

激活之后，你已经会的那套写法直接可用。

```{code-cell} ipython3
import torch.nn as nn

x = torch.randn(4, 3)
print("x:", tuple(x.shape), x.dtype)

lin = nn.Linear(3, 2)
print("前向:", tuple(lin(x).shape))

seq = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
print("Sequential:", tuple(seq(x).shape))
```

自动求导也是 PyTorch 的语义：`loss.backward()` 累积梯度，`opt.zero_grad()` 清空，
`opt.step()` 更新。**注意这里和原生 Jittor 不同**——原生写法是 `optim.step(loss)`
一步完成。

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

xt = torch.randn(8, 1)
loss = ((model(xt)) ** 2).mean()
loss.backward()

print("梯度已就位:", model[0].weight.grad is not None)
opt.zero_grad()
print("zero_grad 之后:", model[0].weight.grad is None or bool((model[0].weight.grad == 0).all()))
```

## 3. 训练一个模型

把上面拼起来就是一个完整的 PyTorch 训练循环——和你原来写的一模一样。

```{code-cell} ipython3
import torch

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.1)

losses = []
for step in range(200):
    xb = torch.rand(32, 1)
    yb = 3.0 * xb + 2.0                 # y = 3x + 2
    opt.zero_grad()
    loss = nn.functional.mse_loss(model(xb), yb)
    loss.backward()
    opt.step()
    losses.append(float(loss.item()))

print("step   0 loss =", round(losses[0], 4))
print("step 199 loss =", round(losses[-1], 4))
assert losses[-1] < losses[0], "训练没有让损失下降，请检查上面的代码"
```

能跑起来是一回事，**确实跑在加速器上**是另一回事：Jittor 在 import 时如果发现可用的
加速器，会把 `use_cuda` 置 1 并自动使用它。要确认真的用了设备，看进程的显存占用
（`nvidia-smi`），只有几十上百 MB 说明实际上还在 CPU 上。

## 4. 设备、混合精度与生态

下面几节需要显卡或联网，因此**不参与仓库的离线冒烟测试**；在你自己的环境里可以执行。

混合精度就是 PyTorch 的那套 API，`torch.amp` 的整个表面都在（`autocast`、
`GradScaler`、`custom_fwd`/`custom_bwd`，以及 `torch.cuda.amp`/`torch.cpu.amp` 子模块）：

```{code-cell} ipython3
:tags: [cuda, skip-execution]

model = model.cuda()
scaler = torch.cuda.amp.GradScaler()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = nn.functional.mse_loss(model(torch.rand(64, 1).cuda()), torch.rand(64, 1).cuda())
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
print("amp step ok:", float(loss.item()))
```

在昇腾 NPU 上**优先用 bf16**（cube 单元原生支持，矩阵乘比 fp32 快一个量级），fp16 配合
`GradScaler` 也可用。设备侧的通用规则见 [设备与驻留](device_placement.md)。

生态库同理——`transformers`、`diffusers`、`PEFT`、`accelerate`、`mmcv` 等按 PyTorch
写成的库，激活之后直接 import 即可，不用改它们一行代码：

```{code-cell} ipython3
:tags: [network, cuda, long-running, skip-execution]

# 以 transformers 为例：先激活（见第 1 节），再照常使用
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").cuda()
print(tok.decode(mdl.generate(**tok("你好", return_tensors="pt").to("cuda"),
                              max_new_tokens=16)[0], skip_special_tokens=True))
```

已收录进验证报告的生态与对拍结论见
[用 PyTorch API 写 Jittor](https://github.com/Jittor/jittor/blob/master/docs/compatibility/torch.md)。

## 5. 哪些写法必须改

激活解决的是**名字和语义**，不是所有差异。迁移时按这份清单检查：

| 你原来的写法 | 在 Jittor 兼容层下 |
| --- | --- |
| `import torch`（无激活） | 必须先 `activate()` 或设 `JITTOR_TORCH_SHIM=1`，否则 import 的是真的 PyTorch |
| `optimizer.step()` 前手写 `zero_grad()`/`backward()` | **保持原样**，兼容层就是 PyTorch 语义 |
| 混用原生 `jt.xxx` 与 `torch.xxx` | 尽量别混：两套前端的 Tensor/Module 类型是分开的 |
| 依赖 `torch.cuda` 的具体版本号 | 以 `torch.__version__` 与文档为准，不要假设与官方 PyTorch 逐位相同 |

原生写法的对照版见 [模型定义与训练](example.md)：那边是 `execute` + `optim.step(loss)`，
这边是 `forward` + `zero_grad/backward/step`。

## 接下来

* [把已有 PyTorch 脚本迁到 Jittor](torch_compat_migration.md)：同一份模型的两种写法并排对照，
  以及迁移时真正会绊倒你的几处。
* 回到 [教程总目录](README.md) 选择其它主题。
