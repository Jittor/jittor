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

# LoRA: parameter-efficient fine-tuning

# LoRA：参数高效微调

Fine-tuning every weight of a large model needs optimizer state for every
parameter, which is where the memory goes. LoRA freezes the pretrained weights
and trains a low-rank update beside them, so the number of trained parameters
drops by orders of magnitude while the frozen model stays exactly as it was.

对大模型全参数微调时，每个参数都要维护优化器状态，显存主要就消耗在这里。LoRA
冻结预训练权重，在旁边训练一个低秩增量，训练参数量因此下降几个数量级，而被冻结
的模型保持原样。

This tutorial implements LoRA in Jittor from scratch and verifies the three
properties that make it work: the adapter starts as an exact no-op, only the
adapter receives gradients, and the trained adapter can be merged back into the
frozen weight without changing the output.

本教程用计图从零实现 LoRA，并验证让它成立的三个性质：适配器初始时严格等价于
恒等、只有适配器拿到梯度、训练好的适配器可以合并回冻结权重且输出不变。

## The idea

## 基本思路

A fine-tuned weight is written as `W + BA`, where `A` is `rank x in` and `B` is
`out x rank`. With `rank` much smaller than either dimension, `BA` has far fewer
entries than `W`. `B` is initialized to zero, so the adapter contributes nothing
until training moves it -- fine-tuning therefore starts exactly at the
pretrained model rather than near it.

微调后的权重写成 `W + BA`，其中 `A` 是 `rank x in`，`B` 是 `out x rank`。当 `rank`
远小于两个维度时，`BA` 的元素数远少于 `W`。`B` 初始化为零，所以训练开始前适配器
不产生任何影响——微调因此严格从预训练模型出发，而不是从它附近出发。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.set_global_seed(0)


class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable low-rank update."""

    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        # The frozen weight must not collect gradients or optimizer state.
        self.base.weight.stop_grad()
        if self.base.bias is not None:
            self.base.bias.stop_grad()
        self.lora_a = jt.randn(rank, base.in_features) * 0.01
        self.lora_b = jt.zeros((base.out_features, rank))

    def execute(self, x):
        update = jt.matmul(jt.matmul(x, self.lora_a.transpose()), self.lora_b.transpose())
        return self.base(x) + update * self.scaling


base = nn.Linear(64, 64)
adapted = LoRALinear(base, rank=4)
x = jt.randn(8, 64)
print("adapter starts as a no-op:",
      bool(np.allclose(adapted(x).numpy(), base(x).numpy(), atol=1e-6)))
```

## What is actually trained

## 真正被训练的是什么

`stop_grad` on the base weight is what makes this parameter-efficient. Compare
the trainable parameter count with and without the adapter.

对基础权重调用 `stop_grad` 才让这件事真正省参数。对比一下有无适配器时可训练参数
的数量。

```{code-cell} ipython3
def trainable(module):
    return [p for p in module.parameters() if not p.is_stop_grad()]


full = int(np.prod(base.weight.shape)) + int(np.prod(base.bias.shape))
lora = sum(int(np.prod(p.shape)) for p in trainable(adapted))
print("full fine-tuning: %d parameters" % full)
print("LoRA rank 4:      %d parameters  (%.1f%%)" % (lora, 100.0 * lora / full))
```

The frozen tensor must also stay frozen through a real optimizer step, not just
report the right flag.

冻结的张量不仅要标志位正确，还必须在真实的优化器步骤之后仍然没有被改动。

```{code-cell} ipython3
before = adapted.base.weight.numpy().copy()
optimizer = nn.Adam(trainable(adapted), lr=1e-2)
optimizer.step((adapted(x) ** 2).mean())
print("frozen weight unchanged after a step:",
      bool(np.array_equal(before, adapted.base.weight.numpy())))
print("adapter moved:",
      not bool(np.allclose(adapted.lora_b.numpy(), 0.0)))
```

## Fine-tuning a small network

## 微调一个小网络

Pretrain a two-layer network on one task, then adapt it to a rotated version of
that task using only the adapters. The rotation is a genuinely different target,
so a frozen model with no adapter cannot fit it.

先在一个任务上预训练一个两层网络，然后只用适配器把它迁移到该任务的旋转版本。
旋转是一个确实不同的目标，因此没有适配器的冻结模型无法拟合它。

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


model = Net()
optimizer = nn.Adam(model.parameters(), lr=5e-3)
for step in range(801):
    inputs, targets = make_batch(source_map)
    loss = ((model(inputs) - targets) ** 2).mean()
    optimizer.step(loss)
print("pretraining loss: %.4f" % float(loss.numpy().reshape(-1)[0]))
```

```{code-cell} ipython3
inputs, targets = make_batch(target_map, n=512)
frozen_loss = float(((model(inputs) - targets) ** 2).mean().numpy().reshape(-1)[0])
print("frozen model on the new task: %.4f" % frozen_loss)

model.fc1 = LoRALinear(model.fc1, rank=4)
model.fc2 = LoRALinear(model.fc2, rank=4)

adapter_params = trainable(model)
total = sum(int(np.prod(p.shape)) for p in model.parameters())
tuned = sum(int(np.prod(p.shape)) for p in adapter_params)
print("training %d of %d parameters (%.1f%%)" % (tuned, total, 100.0 * tuned / total))

optimizer = nn.Adam(adapter_params, lr=1e-2)
for step in range(801):
    batch_inputs, batch_targets = make_batch(target_map)
    loss = ((model(batch_inputs) - batch_targets) ** 2).mean()
    optimizer.step(loss)

adapted_loss = float(((model(inputs) - targets) ** 2).mean().numpy().reshape(-1)[0])
print("after LoRA:                   %.4f" % adapted_loss)
print("improvement: %.1fx" % (frozen_loss / adapted_loss))
```

## Merging the adapter

## 合并适配器

Because the update is a plain matrix product, a trained adapter can be folded
into the frozen weight. The merged layer has the original shape and cost, so
inference pays nothing for having been fine-tuned this way.

因为增量就是一个普通的矩阵乘积，训练好的适配器可以直接折进冻结权重。合并后的层
形状和开销与原始层一致，推理不会因为用过 LoRA 而多付任何代价。

```{code-cell} ipython3
def merge(layer):
    merged = nn.Linear(layer.base.in_features, layer.base.out_features)
    delta = jt.matmul(layer.lora_b, layer.lora_a) * layer.scaling
    merged.weight = layer.base.weight + delta
    merged.bias = layer.base.bias
    return merged


reference = model(inputs).numpy()
model.fc1 = merge(model.fc1)
model.fc2 = merge(model.fc2)
print("merged output matches: %s (max difference %.2e)"
      % (bool(np.allclose(reference, model(inputs).numpy(), atol=1e-4)),
         float(np.abs(reference - model(inputs).numpy()).max())))
```

## What to try next

## 接下来可以尝试

* Vary `rank` and plot final loss against trained-parameter count -- the curve
  flattens well before full fine-tuning.
* Adapt only the attention projections of the Transformer from the
  `transformer` tutorial, which is where LoRA is normally applied.
* Keep several adapters for one frozen backbone and swap them per task; merging
  is only worth it once you have settled on one.

* 改变 `rank`，画出最终 loss 与可训练参数量的关系——曲线远在全参数微调之前就已经
  变平。
* 只对 `transformer` 教程里 Transformer 的注意力投影加适配器，这也是 LoRA 通常的
  用法。
* 为同一个冻结主干保留多个适配器并按任务切换；只有在确定用哪一个之后才值得合并。
