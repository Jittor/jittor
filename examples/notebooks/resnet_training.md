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

# Train a residual network

# 训练一个残差网络

Residual connections let a network learn a correction to an identity path.
This tutorial builds a compact ResNet in native Jittor, runs three complete SGD
updates, checks BatchNorm state, and restores the trained state into a fresh
model. The tensors are deliberately small so every cell runs offline on a CPU.

残差连接让网络在恒等路径上学习修正。本教程使用原生计图搭建一个紧凑 ResNet，
执行三次完整 SGD 更新，检查 BatchNorm 状态，并把训练后的 state 恢复到新模型。
张量尺寸刻意保持很小，因此所有单元格都能在离线 CPU 环境执行。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.set_global_seed(7)
```

## A residual block

## 残差块

The main branch applies two convolution-normalization pairs. When the channel
count changes, a 1x1 convolution projects the identity branch to the same
shape. The addition is the residual connection.

主分支包含两组卷积与归一化。通道数变化时，用 1x1 卷积把恒等分支投影到相同形状；
最后的加法就是残差连接。

```{code-cell} ipython3
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def execute(self, x):
        residual = self.skip(x)
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return nn.relu(x + residual)


class TinyResNet(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.block1 = ResidualBlock(8, 8)
        self.block2 = ResidualBlock(8, 16)
        self.head = nn.Linear(16, classes)

    def execute(self, images):
        x = nn.relu(self.stem(images))
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean((2, 3))
        return self.head(x)


model = TinyResNet()
sample = jt.randn((4, 3, 8, 8))
print("logits:", model(sample).shape)
```

## Three complete updates

## 三次完整更新

A training trajectory is more informative than a single backward call: every
step has to use the parameters written by the previous step. The fixed
synthetic batch makes the example deterministic. We check the loss is finite,
the stem weight changes, and BatchNorm records all three batches.

训练轨迹比单次反向更有信息，因为每一步都必须使用上一步写回的参数。固定的合成
batch 让示例可以复现。下面检查 loss 有限、stem 权重确实变化，并确认 BatchNorm
记录了全部三个 batch。

```{code-cell} ipython3
rng = np.random.RandomState(8)
images = jt.array(rng.randn(4, 3, 8, 8).astype("float32"))
targets = jt.array(rng.randn(4, 4).astype("float32"))
model = TinyResNet()
optimizer = nn.SGD(model.parameters(), lr=1e-3)
initial_stem = np.array(model.stem.weight.numpy(), copy=True)
losses = []

for step in range(3):
    logits = model(images)
    loss = ((logits - targets) ** 2).mean()
    losses.append(float(loss.numpy().reshape(-1)[0]))
    optimizer.step(loss)

assert np.isfinite(losses).all()
assert not np.array_equal(model.stem.weight.numpy(), initial_stem)
assert int(model.block1.bn1.num_batches_tracked.item()) == 3
print("loss trajectory:", [round(value, 5) for value in losses])
```

## Restore model state

## 恢复模型状态

`state_dict` includes trainable parameters and persistent normalization state.
After switching to evaluation mode, loading that state into a fresh model must
reproduce the same logits.

`state_dict` 同时包含可训练参数和持久化的归一化状态。切换到 evaluation mode 后，
把 state 加载到一个新模型中，输出 logits 应完全复现。

```{code-cell} ipython3
model.eval()
expected = model(images).numpy()
state = {
    name: np.array(value.numpy(), copy=True)
    for name, value in model.state_dict().items()
}

restored = TinyResNet()
restored.load_parameters(state)
restored.eval()
actual = restored(images).numpy()

assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
print("restored logits match:", np.allclose(actual, expected))
```

## What to try next

## 接下来可以尝试

* Add stride-2 blocks and train on 32x32 images.
* Replace plain SGD with `nn.AdamW` and compare the loss trajectory.
* Set `jt.flags.use_cuda = 1`; the model and optimizer code stay unchanged.

* 加入 stride-2 残差块，并训练 32x32 图像。
* 把普通 SGD 换成 `nn.AdamW`，比较 loss 轨迹。
* 设置 `jt.flags.use_cuda = 1`；模型与优化器代码无需改动。
