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

# Transformer from scratch

# 从零实现 Transformer

This tutorial builds a Transformer encoder in pure Jittor and trains it on a
copy task small enough to finish on a CPU. Every piece -- scaled dot-product
attention, multi-head projection, sinusoidal positions, the pre-norm residual
block -- is written out rather than imported, so you can see exactly which
Jittor operators a modern attention stack is made of.

本教程用纯计图从零搭一个 Transformer 编码器，并在一个 CPU 上就能跑完的复制任务
上训练它。缩放点积注意力、多头投影、正弦位置编码、pre-norm 残差块都逐行写出来，
而不是直接调用现成模块，这样可以清楚看到现代注意力网络到底由哪些计图算子组成。

## Scaled dot-product attention

## 缩放点积注意力

Attention is three matrix multiplications and one softmax. The scale
`1/sqrt(d)` keeps the logits in a range where softmax has usable gradients; the
mask is added *before* the softmax so that masked positions receive zero weight.

注意力就是三次矩阵乘加一次 softmax。缩放系数 `1/sqrt(d)` 让 logits 落在 softmax
梯度可用的范围内；掩码要加在 softmax 之前，这样被屏蔽的位置权重才是零。

```{code-cell} ipython3
import math

import numpy as np

import jittor as jt
from jittor import nn

jt.set_global_seed(0)


def attention(q, k, v, mask=None):
    """q, k, v: [batch, heads, length, head_dim]"""
    scores = jt.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    weights = nn.softmax(scores, dim=-1)
    return jt.matmul(weights, v), weights


q = jt.randn(1, 1, 4, 8)
out, weights = attention(q, q, q)
print("output", out.shape)
print("each row of the attention matrix sums to 1:",
      np.allclose(weights.sum(-1).numpy(), 1.0, atol=1e-5))
```

A causal mask is an upper-triangular block of `-inf`: position `i` may only
attend to positions `j <= i`. Building it from `jt.tril` keeps the whole thing
inside the graph.

因果掩码是一块上三角的 `-inf`：位置 `i` 只能看到 `j <= i`。用 `jt.tril` 构造可以
让它留在计算图里。

```{code-cell} ipython3
def causal_mask(length):
    keep = jt.tril(jt.ones((length, length)))
    return (keep - 1.0) * 1e9


mask = causal_mask(4)
_out, weights = attention(q, q, q, mask.reshape(1, 1, 4, 4))
print(np.round(weights.numpy()[0, 0], 3))
```

## Multi-head attention

## 多头注意力

Multiple heads let the layer attend to several relationships at once. A single
`Linear` produces all of q, k and v, and the reshape splits the channel axis
into `heads x head_dim`.

多个头让同一层同时关注多种关系。用一个 `Linear` 一次产生 q、k、v，再通过 reshape
把通道轴拆成 `heads x head_dim`。

```{code-cell} ipython3
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def execute(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        out, _ = attention(qkv[0], qkv[1], qkv[2], mask)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(b, n, c))


layer = MultiHeadAttention(32, 4)
print(layer(jt.randn(2, 6, 32)).shape)
```

## Positional encoding

## 位置编码

Attention is permutation invariant, so position has to be injected. The classic
sinusoidal encoding needs no parameters and extrapolates to lengths never seen
during training.

注意力对位置是置换不变的，所以位置信息必须显式注入。经典的正弦编码不含参数，
并且可以外推到训练时没见过的长度。

```{code-cell} ipython3
def sinusoidal_positions(length, dim):
    position = jt.arange(length).float().reshape(-1, 1)
    scale = jt.exp(-math.log(10000.0) * jt.arange(0, dim, 2).float() / dim)
    angles = position * scale.reshape(1, -1)
    return jt.concat([jt.sin(angles), jt.cos(angles)], dim=-1).reshape(1, length, dim)


positions = sinusoidal_positions(16, 32)
print(positions.shape, "range", float(positions.min()), float(positions.max()))
```

## The encoder block

## 编码器块

A pre-norm block -- normalize, sub-layer, add -- is what modern Transformers
use, because the residual path stays unnormalized and gradients reach the early
layers without a warmup schedule.

现代 Transformer 用的是 pre-norm 结构：先归一化，再过子层，最后相加。残差通路
保持未归一化，梯度能直接到达浅层，不需要额外的 warmup。

```{code-cell} ipython3
class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def execute(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.fc2(nn.gelu(self.fc1(self.norm2(x))))


class TinyTransformer(nn.Module):
    def __init__(self, vocab, dim=32, depth=2, heads=4, max_len=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.positions = sinusoidal_positions(max_len, dim)
        self.positions.requires_grad = False
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def execute(self, tokens):
        n = tokens.shape[1]
        x = self.embed(tokens) + self.positions[:, :n]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


model = TinyTransformer(vocab=12)
print("parameters:", sum(int(np.prod(p.shape)) for p in model.parameters()))
```

## Training on a copy task

## 在复制任务上训练

The task is deliberately trivial: reproduce the input sequence. It is enough to
show the training loop and, more usefully, it has a known perfect score, so you
can tell whether the model actually learned instead of guessing from a loss
curve.

任务故意设得很简单：把输入序列原样输出。它足以展示训练循环，更有用的是它有已知
的满分，因此可以直接判断模型是否真的学会了，而不是只盯着 loss 曲线猜。

```{code-cell} ipython3
VOCAB, LENGTH, BATCH = 12, 8, 32
rng = np.random.RandomState(0)


def batch():
    tokens = rng.randint(0, VOCAB, size=(BATCH, LENGTH)).astype("int32")
    return jt.array(tokens), jt.array(tokens)


model = TinyTransformer(vocab=VOCAB, max_len=LENGTH)
optimizer = nn.Adam(model.parameters(), lr=3e-3)

for step in range(301):
    tokens, targets = batch()
    logits = model(tokens)
    loss = nn.cross_entropy_loss(logits.reshape(-1, VOCAB), targets.reshape(-1))
    optimizer.step(loss)
    if step % 100 == 0:
        print("step %3d  loss %.4f" % (step, float(loss.numpy().reshape(-1)[0])))
```

```{code-cell} ipython3
tokens, targets = batch()
predicted = model(tokens).argmax(dim=-1)[0]
accuracy = float((predicted == targets).float().mean().numpy().reshape(-1)[0])
print("copy accuracy: %.3f" % accuracy)
print("input     ", tokens.numpy()[0].tolist())
print("prediction", predicted.numpy()[0].tolist())
```

## What to try next

## 接下来可以尝试

* Pass `causal_mask(LENGTH)` into the blocks and train next-token prediction
  instead of copying -- that is exactly a small GPT.
* Replace `LayerNorm` with `nn.RMSNorm` and the sinusoidal table with rotary
  positions, which is what current large language models use.
* Move the model to a GPU with `jt.flags.use_cuda = 1`; nothing in the code
  above has to change.

* 把 `causal_mask(LENGTH)` 传进各个块，把任务换成预测下一个 token，那就是一个小
  GPT 了。
* 把 `LayerNorm` 换成 `nn.RMSNorm`，把正弦位置表换成旋转位置编码，这是当前大语言
  模型的常见做法。
* 用 `jt.flags.use_cuda = 1` 把模型搬到 GPU，上面的代码一行都不用改。
