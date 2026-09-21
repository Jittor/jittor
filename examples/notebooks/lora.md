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

# LoRA：只训练一个低秩增量

给一个大模型做全参数微调，显存的大头**不是权重本身，而是优化器状态**：Adam 要为每个
参数额外保存一阶和二阶动量。参数翻倍，优化器状态也翻倍。

LoRA 的做法是：**把预训练权重冻住，只在旁边训练一个低秩增量**。

$$W_{\text{微调}} = W + BA,\qquad A \in \mathbb{R}^{r\times d_{\text{in}}},\ B \in \mathbb{R}^{d_{\text{out}}\times r}$$

`r` 远小于两个维度时，`BA` 的元素数比 `W` 少几个数量级。这一篇从零实现 LoRA，
并**验证让它成立的三个性质**：

1. 训练开始前，适配器**严格等价于恒等**（不是近似）；
2. 冻结的权重在真实优化器步骤之后**一个数都没变**；
3. 训练好的适配器**可以合并回权重**，输出完全不变、推理零额外开销。

外加一条实测的**rank 阶梯**：到底需要多大的 `r`。

> 全篇 CPU 可跑，约 30 秒。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(0)
```

## 1. 适配器：从严格的恒等开始

`B` 初始化为**全零**，所以 `BA = 0`，适配器一开始不产生任何影响。这一点比看起来重要：
微调因此**严格从预训练模型出发**，而不是从它附近出发。少了这个性质，微调的第一步就会
把已经学好的东西破坏掉。

```{code-cell} ipython3
class LoRALinear(nn.Module):
    """把一个冻结的 Linear 包起来，旁边加一个可训练的低秩增量。"""

    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank

        # 冻结的权重不能收集梯度，否则优化器状态会把它算进去——省参数就白省了
        self.base.weight.stop_grad()
        if self.base.bias is not None:
            self.base.bias.stop_grad()

        self.lora_a = jt.randn(rank, base.in_features) * 0.01
        self.lora_b = jt.zeros((base.out_features, rank))      # ← 零初始化

    def execute(self, x):
        update = jt.matmul(jt.matmul(x, self.lora_a.transpose()),
                           self.lora_b.transpose())
        return self.base(x) + update * self.scaling


base = nn.Linear(64, 64)
adapted = LoRALinear(base, rank=4)
x = jt.randn(8, 64)

before = base(x).numpy()
after = adapted(x).numpy()
print("适配器初始是恒等:", bool(np.allclose(before, after, atol=1e-6)))
print("最大差异: %.2e" % np.abs(before - after).max())
assert np.allclose(before, after, atol=1e-6), "零初始化必须严格等价于恒等"
```

`alpha / rank` 这个缩放是原论文的约定：`alpha` 固定时，改 `r` 不会同时改变增量的
初始幅度。这里 `rank=4, alpha=8`，缩放系数是 2。

## 2. 谁在被训练

`stop_grad()` 是整件事省参数的关键。对比一下可训练参数量。

```{code-cell} ipython3
def trainable(module):
    """只挑出还会收集梯度的参数。"""
    return [p for p in module.parameters() if not p.is_stop_grad()]


full = int(np.prod(base.weight.shape)) + int(np.prod(base.bias.shape))
lora = sum(int(np.prod(p.shape)) for p in trainable(adapted))
print("全参数微调: %d 个参数" % full)
print("LoRA r=4  : %d 个参数  (%.1f%%)" % (lora, 100.0 * lora / full))
```

但**标志位正确不等于真的冻住了**。必须跑一步真实的优化器，再逐个比较数值。

```{code-cell} ipython3
before_weight = adapted.base.weight.numpy().copy()
before_bias = adapted.base.bias.numpy().copy()

optimizer = nn.Adam(trainable(adapted), lr=1e-2)
optimizer.step((adapted(x) ** 2).mean())

print("一步之后，冻结权重一个数都没变:",
      bool(np.array_equal(before_weight, adapted.base.weight.numpy())))
print("偏置同样没变:",
      bool(np.array_equal(before_bias, adapted.base.bias.numpy())))
print("适配器动了:",
      not bool(np.allclose(adapted.lora_b.numpy(), 0.0)))
assert np.array_equal(before_weight, adapted.base.weight.numpy())
assert not np.allclose(adapted.lora_b.numpy(), 0.0)
```

## 3. 微调一个小网络

造一个「换任务」的场景：先在源任务上预训练，再把它迁移到目标任务的**旋转版本**。
旋转是一个确实不同的目标函数，所以**冻结模型自己无法拟合它**——这就是需要微调的理由。

```{code-cell} ipython3
rng = np.random.RandomState(0)
DIM = 32

source_map = rng.randn(DIM, DIM).astype("float32") * 0.2
angle = np.pi / 4
rotation = np.eye(DIM, dtype="float32")
rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
target_map = (source_map @ rotation).astype("float32")


def make_batch(mapping, n=128):
    inputs = rng.randn(n, DIM).astype("float32")
    return jt.array(inputs), jt.array(inputs @ mapping)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, 64)
        self.fc2 = nn.Linear(64, DIM)

    def execute(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


pretrained = Net()
optimizer = nn.Adam(pretrained.parameters(), lr=5e-3)
for step in range(801):
    inputs, targets = make_batch(source_map)
    loss = ((pretrained(inputs) - targets) ** 2).mean()
    optimizer.step(loss)
print("源任务上预训练完成，loss %.4f" % float(loss.numpy().ravel()[0]))

# 留出一批评估样本，并记下预训练权重
eval_inputs, eval_targets = make_batch(target_map, n=512)
base_state = {k: np.array(v.numpy(), copy=True)
              for k, v in pretrained.state_dict().items()}
```

冻结模型在**新任务**上的表现，就是「不微调」的基线：

```{code-cell} ipython3
frozen = Net()
frozen.load_parameters(base_state)          # 冻结模型 = 不做任何适配
frozen_loss = float(((frozen(eval_inputs) - eval_targets) ** 2).mean().numpy().ravel()[0])
total = sum(int(np.prod(p.shape)) for p in frozen.parameters())
print("冻结模型在新任务上: %.4f" % frozen_loss)
print("全参数微调需要训练: %d 个参数" % total)
```

现在只给两个 `Linear` 加适配器，其余全部冻住。

```{code-cell} ipython3
adapted_net = Net()
adapted_net.load_parameters(base_state)
adapted_net.fc1 = LoRALinear(adapted_net.fc1, rank=4)
adapted_net.fc2 = LoRALinear(adapted_net.fc2, rank=4)

params = trainable(adapted_net)
tuned = sum(int(np.prod(p.shape)) for p in params)
print("训练 %d / %d 个参数 (%.1f%%)"
      % (tuned, total, 100.0 * tuned / total))

optimizer = nn.Adam(params, lr=1e-2)
for step in range(801):
    batch_inputs, batch_targets = make_batch(target_map)
    loss = ((adapted_net(batch_inputs) - batch_targets) ** 2).mean()
    optimizer.step(loss)

adapted_loss = float(((adapted_net(eval_inputs) - eval_targets) ** 2).mean().numpy().ravel()[0])
print("\n冻结模型        : %.4f" % frozen_loss)
print("加上 LoRA 之后  : %.4f" % adapted_loss)
print("改善            : %.1f 倍" % (frozen_loss / adapted_loss))
```

## 4. rank 阶梯：到底需要多大

`r` 是一个超参。这里把 `r` 从 1 扫到 16，每一步都从**同一份预训练权重**出发。

```{code-cell} ipython3
def run_rank(rank, steps=801, lr=1e-2):
    """从同一份预训练权重出发，用指定 rank 适配到新任务。"""
    net = Net()
    net.load_parameters(base_state)
    net.fc1 = LoRALinear(net.fc1, rank=rank)
    net.fc2 = LoRALinear(net.fc2, rank=rank)

    params = trainable(net)
    tuned = sum(int(np.prod(p.shape)) for p in params)
    optimizer = nn.Adam(params, lr=lr)
    for _ in range(steps):
        batch_inputs, batch_targets = make_batch(target_map)
        optimizer.step(((net(batch_inputs) - batch_targets) ** 2).mean())

    loss = float(((net(eval_inputs) - eval_targets) ** 2).mean().numpy().ravel()[0])
    return tuned, loss


print("%-6s %-12s %-8s %-10s" % ("rank", "可训练参数", "占比", "新任务 loss"))
print("%-6s %-12d %-8s %-10.4f" % ("冻结", 0, "0.0%", frozen_loss))
ladder = {}
for rank in (1, 2, 4, 8, 16):
    tuned, loss = run_rank(rank)
    ladder[rank] = (tuned, loss)
    print("%-6d %-12d %-8s %-10.4f"
          % (rank, tuned, "%.1f%%" % (100.0 * tuned / total), loss))
```

**`rank=1` 就够了**——只训练 192 个参数（4.6%）就把损失从 `0.0513` 拉到 `0.0007`。
从 `r=1` 加到 `r=16`（参数量涨 16 倍），损失只从 `0.0007` 变到 `0.0004`。

```{code-cell} ipython3
r1_loss = ladder[1][1]
r16_loss = ladder[16][1]
assert r1_loss < frozen_loss / 20, "rank=1 就应该把损失降下来"
assert r16_loss <= r1_loss + 1e-3, "加大 rank 不该变差"
print("rank 1 -> 16 参数量涨了 %d 倍，损失只从 %.4f 到 %.4f"
      % (ladder[16][0] // ladder[1][0], r1_loss, r16_loss))
```

**但这里有一个必须说明的前提**：目标任务是一个只在**前两维**上发生的旋转，所以它
**本身就只差一个 rank 2 的修正**。曲线这么早变平是任务的性质，不是一条普遍规律。
如果新任务和原任务的差异是全局的、高维的，`r` 就得更大。**判断办法是扫一遍**，
而不是照搬别人的 `r=8`。

## 5. 合并回权重：推理零成本

因为增量就是一个普通的矩阵乘积，训练好的适配器可以直接**折进**冻结权重。
合并后的层形状和开销与原始层完全一致。

```{code-cell} ipython3
def merge(layer):
    """把 LoRA 增量折进权重，返回一个普通的 Linear。"""
    merged = nn.Linear(layer.base.in_features, layer.base.out_features)
    delta = jt.matmul(layer.lora_b, layer.lora_a) * layer.scaling
    merged.weight = layer.base.weight + delta
    merged.bias = layer.base.bias
    return merged


reference = adapted_net(eval_inputs).numpy()
adapted_net.fc1 = merge(adapted_net.fc1)
adapted_net.fc2 = merge(adapted_net.fc2)
merged_out = adapted_net(eval_inputs).numpy()

diff = float(np.abs(reference - merged_out).max())
print("合并后输出一致:", bool(np.allclose(reference, merged_out, atol=1e-4)))
print("最大差异: %.2e" % diff)
assert np.allclose(reference, merged_out, atol=1e-4)
print("\n合并之后，模型里不再有 LoRA 的任何痕迹——推理不会多付任何代价")
```

**什么时候不该合并**：如果同一个冻结主干要配多个适配器、按任务切换，合并就失去了意义——
每合并一次就得存一份完整权重，反而更占地方。**确定只用哪一个之后再合并。**

## 6. 同一件事，用 PyTorch API 再写一遍

LoRA 的两个部件——冻结权重与低秩增量——在两种前端里的写法几乎逐字对应。
**激活必须在创建任何张量之前**。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)


class LoRALinearT(tnn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        for p in self.base.parameters():
            p.requires_grad_(False)                    # ← 对应 stop_grad()
        self.lora_a = tnn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_b = tnn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        update = (x @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(x) + update * self.scaling


base_t = tnn.Linear(64, 64)
adapted_t = LoRALinearT(base_t, rank=4)
x_t = torch.randn(8, 64)

print("初始等价于恒等:",
      bool(torch.allclose(adapted_t(x_t), base_t(x_t), atol=1e-6)))
print("可训练参数: %d / %d"
      % (sum(p.numel() for p in adapted_t.parameters() if p.requires_grad),
         sum(p.numel() for p in adapted_t.parameters())))
print("基础权重不需要梯度:", not base_t.weight.requires_grad)
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 冻结权重 | `p.stop_grad()` | `p.requires_grad_(False)` |
| 新建可训练参数 | `jt.randn(...)`（自动进 `parameters()`） | `nn.Parameter(...)` |
| 前向方法名 | `execute` | `forward` |
| 训练一步 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |

## 7. 检查清单

```{code-block} text
1. B 必须零初始化        —— 保证适配器从「严格恒等」出发，而不是从它附近出发
2. stop_grad 要给权重和偏置都加 —— 漏掉偏置，优化器状态照样翻倍
3. 冻结要看数值，不看标志位 —— 跑一步优化器再逐个比较，标志位对不代表没被改
4. 要有「不微调」的基线   —— 本篇是冻结模型在新任务上的 0.0513
5. rank 要扫，不要照搬   —— 且要说明任务本身是不是低秩的（本篇只差前两维的旋转）
6. 合并前想清楚用几个适配器 —— 多任务切换时合并反而更占空间
7. 合并后要验证输出不变   —— 差异应在 1e-4 量级，来自浮点累加顺序
```

第 3 条和第 5 条是这一篇最容易出问题的地方：**「冻结」和「rank 够用」都是需要验证的
断言，不是可以默认的前提。**

生态里真实使用 LoRA 走的是 PEFT（见 [把真实的生态库跑在 Jittor 上](torch_ecosystem.md)），
它把同样这几件事包装成了 `get_peft_model`；这一篇拆开的就是它内部做的事。

## 接下来

* [从零实现 Transformer](transformer.md)：LoRA 通常加在注意力的投影层上。
* [把真实的生态库跑在 Jittor 上](torch_ecosystem.md)：用 PEFT 做同样的事。
* [检查点与断点续训](checkpointing.md)：只保存适配器与保存整个模型有什么区别。
* 回到 [教程总目录](README.md)。
