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

# Train a vision transformer

# 训练一个视觉 Transformer

A vision transformer cuts an image into patches, treats each patch as a token,
and lets self-attention decide which patches inform which. This tutorial builds
one in native Jittor -- patch embedding, a class token, learned positions,
multi-head attention written out in full, and pre-norm blocks -- then runs three
complete AdamW updates. The tensors stay small enough for every cell to run
offline on a CPU.

视觉 Transformer 把图像切成 patch，将每个 patch 当作一个 token，由自注意力决定
哪些 patch 参考哪些。本教程用原生计图搭建它——patch embedding、class token、
可学习位置编码、完整写出的多头注意力，以及 pre-norm 结构——然后执行三次完整的
AdamW 更新。张量尺寸足够小，所有单元格都能在离线 CPU 环境执行。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.set_global_seed(11)
```

## From image to tokens

## 从图像到 token

A strided convolution whose kernel equals its stride is exactly a patch
embedding: each output position sees one non-overlapping patch and nothing
else. Flattening the spatial grid turns the result into a token sequence.
The class token is one extra learned vector prepended to that sequence; it
carries no patch of its own, so it is free to accumulate whatever the head
needs. Positions are learned rather than fixed, because attention itself is
permutation invariant and would otherwise never know how the patches were laid
out.

卷积核大小等于步长时，卷积就是 patch embedding：每个输出位置恰好看到一个不重叠
的 patch。把空间网格展平，结果就成了 token 序列。class token 是额外的一个可学习
向量，拼在序列最前面；它不对应任何 patch，因此可以自由地积累分类头需要的信息。
位置编码采用可学习参数，因为注意力本身对顺序不敏感，否则无从知道 patch 的排布。

```{code-cell} ipython3
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=8, patch_size=4, channels=3, width=16):
        super().__init__()
        assert image_size % patch_size == 0
        self.patches = (image_size // patch_size) ** 2
        self.project = nn.Conv2d(channels, width, patch_size, stride=patch_size)
        self.class_token = jt.zeros((1, 1, width))
        self.positions = jt.randn((1, self.patches + 1, width)) * 0.02

    def execute(self, images):
        x = self.project(images)
        batch, width = x.shape[0], x.shape[1]
        x = x.reshape((batch, width, -1)).transpose(0, 2, 1)
        tokens = self.class_token.broadcast((batch, 1, width))
        return jt.concat([tokens, x], dim=1) + self.positions


embedding = PatchEmbedding()
print("tokens:", embedding(jt.randn((2, 3, 8, 8))).shape)
```

## Multi-head self-attention

## 多头自注意力

One `Linear` produces queries, keys and values together, which keeps the three
projections in a single matmul. Reshaping to `(batch, heads, tokens, head_dim)`
gives each head its own subspace. Scaling the scores by `1/sqrt(head_dim)`
keeps their variance independent of head width, so the softmax does not
saturate as the model grows.

一个 `Linear` 同时产生 query、key 和 value，三个投影合并为一次矩阵乘。reshape 成
`(batch, heads, tokens, head_dim)` 后，每个头拥有各自的子空间。用
`1/sqrt(head_dim)` 缩放分数，使其方差不随头宽变化，softmax 才不会随模型变大而
饱和。

```{code-cell} ipython3
class SelfAttention(nn.Module):
    def __init__(self, width=16, heads=4):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def execute(self, x, return_weights=False):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape((batch, tokens, 3, self.heads, self.head_dim))
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = nn.bmm(
            query.reshape((-1, tokens, self.head_dim)),
            key.reshape((-1, tokens, self.head_dim)).transpose(0, 2, 1),
        ) * self.scale
        weights = nn.softmax(scores, dim=-1)
        context = nn.bmm(weights, value.reshape((-1, tokens, self.head_dim)))
        context = context.reshape((batch, self.heads, tokens, self.head_dim))
        context = context.transpose(0, 2, 1, 3).reshape((batch, tokens, width))
        if return_weights:
            return self.out(context), weights.reshape(
                (batch, self.heads, tokens, tokens))
        return self.out(context)


attention = SelfAttention()
output, weights = attention(jt.randn((2, 5, 16)), return_weights=True)
row_sums = weights.sum(-1).numpy()
assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
print("attention output:", output.shape, "| rows sum to 1:", True)
```

## Pre-norm blocks and the classifier

## Pre-norm 块与分类头

Each block normalizes *before* its sublayer and adds the result to the residual
stream. That ordering leaves an unnormalized identity path from input to
output, which is why deep pre-norm stacks train without a warmup schedule.
The classifier reads only the class token.

每个块在子层**之前**做归一化，再把结果加回残差流。这种顺序保留了从输入到输出的
未归一化恒等路径，因此较深的 pre-norm 堆叠无需 warmup 也能训练。分类头只读取
class token。

```{code-cell} ipython3
class Block(nn.Module):
    def __init__(self, width=16, heads=4, expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Linear(width * expansion, width),
        )

    def execute(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, classes=4, width=16, depth=2, heads=4):
        super().__init__()
        self.embedding = PatchEmbedding(width=width)
        self.blocks = nn.Sequential(*[Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def execute(self, images):
        x = self.blocks(self.embedding(images))
        return self.head(self.norm(x)[:, 0])


model = TinyViT()
print("logits:", model(jt.randn((4, 3, 8, 8))).shape)
```

## Three complete updates

## 三次完整更新

Every step must consume the parameters the previous step wrote, so a short
trajectory tests more than one backward call does. The batch is fixed, which
makes the run reproducible. We check that the loss stays finite, that the
patch-embedding weight actually moved, and that AdamW's moment estimates exist
after the first step.

每一步都必须使用上一步写回的参数，因此一段短轨迹比单次反向更有说服力。batch
固定，运行可复现。下面检查 loss 保持有限、patch embedding 权重确实变化，并确认
第一步之后 AdamW 的动量估计已经建立。

```{code-cell} ipython3
rng = np.random.RandomState(12)
images = jt.array(rng.randn(4, 3, 8, 8).astype("float32"))
targets = jt.array(rng.randint(0, 4, size=4).astype("int32"))

model = TinyViT()
optimizer = nn.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
initial = np.array(model.embedding.project.weight.numpy(), copy=True)
losses = []

for step in range(3):
    logits = model(images)
    loss = nn.cross_entropy_loss(logits, targets)
    losses.append(float(loss.numpy().reshape(-1)[0]))
    optimizer.step(loss)

assert np.isfinite(losses).all()
assert not np.array_equal(model.embedding.project.weight.numpy(), initial)
assert any(len(group["m"]) > 0 for group in optimizer.param_groups)
print("loss trajectory:", [round(value, 5) for value in losses])
```

## What the class token learned to look at

## class token 关注了什么

Attention weights are a distribution over tokens, so the class-token row of the
last block says how much each patch contributed to the prediction. Reading it
costs one extra forward pass and no training-time bookkeeping.

注意力权重是 token 上的分布，因此最后一个块中 class token 所在的那一行，说明每个
patch 对预测的贡献。读取它只需额外一次前向，训练过程无需任何额外记录。

```{code-cell} ipython3
model.eval()
x = model.embedding(images)
for block in model.blocks[:-1]:
    x = block(x)
last = model.blocks[-1]
_, weights = last.attention(last.norm1(x), return_weights=True)

class_row = weights[:, :, 0].mean(1).numpy()
assert class_row.shape == (4, model.embedding.patches + 1)
assert np.allclose(class_row.sum(-1), 1.0, atol=1e-5)
print("class-token attention over patches:", np.round(class_row[0], 4))
```

## Restore model state

## 恢复模型状态

`state_dict` carries every learned tensor, including the class token and the
position table. Loading it into a fresh model must reproduce the same logits
exactly.

`state_dict` 包含全部学习到的张量，class token 和位置表也在其中。把它加载到新
模型后，输出 logits 应完全复现。

```{code-cell} ipython3
expected = model(images).numpy()
state = {
    name: np.array(value.numpy(), copy=True)
    for name, value in model.state_dict().items()
}

restored = TinyViT()
restored.load_parameters(state)
restored.eval()
actual = restored(images).numpy()

assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
print("restored logits match:", np.allclose(actual, expected))
```

## What to try next

## 接下来可以尝试

* Raise `depth` and `width`, and watch the loss trajectory change shape.
* Swap the learned position table for fixed sinusoids and compare.
* Set `jt.flags.use_cuda = 1`; the model and optimizer code stay unchanged.

* 增大 `depth` 与 `width`，观察 loss 轨迹形状的变化。
* 把可学习位置表换成固定的正弦编码并作比较。
* 设置 `jt.flags.use_cuda = 1`；模型与优化器代码无需改动。
