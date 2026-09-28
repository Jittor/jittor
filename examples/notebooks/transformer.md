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

# 从零实现 Transformer

[循环网络](rnn.md) 那一篇结尾留下了一个问题：信息要一步一步往前走，梯度就要穿过同样多的
步数连乘，长度一上去就衰减没了。注意力换了个做法——**一次看到整条序列**，任意两个位置
之间只有一步。代价是计算量随长度平方增长。

这一篇把 Transformer 编码器**从零写出来**：缩放点积注意力、多头投影、正弦位置编码、
pre-norm 残差块，全部逐行写，不调用现成模块。这样你能看清一个现代注意力网络到底由哪些
算子组成。

这一篇会：

* 用**实测数字**说明缩放系数 `1/sqrt(d)` 到底在防什么；
* 验证因果掩码真的把未来挡住了；
* 在一个**有已知满分**的复制任务上训练，留出集准确率 **1.000**——不是猜「loss 应该够低」；
* 讲清 pre-norm 与位置编码各自的必要性；
* 指出两个容易写错的 API 名字。

> 全篇 CPU 可跑，训练部分约 **25 秒**。

```{code-cell} ipython3
import math
import time

import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(0)
```

## 1. 缩放点积注意力

注意力就是三次矩阵乘加一次 softmax：

$$\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

先看那个 `/ sqrt(d)` 在防什么。`Q`、`K` 的每个元素独立、方差为 1 时，点积的方差是
`d`——维度越高，logit 的绝对值越大。而 softmax 的梯度在 logit 很大时**趋于 0**：
它会把注意力压成一个近似 one-hot 的分布，其它位置再也拿不到梯度。

不要信这句描述，量一下。

```{code-cell} ipython3
d = 64
q = jt.randn(1, 1, 4, d)
k = jt.randn(1, 1, 4, d)

raw = jt.matmul(q, k.transpose(0, 1, 3, 2))
scaled = raw / math.sqrt(d)

print("%-14s %-14s %-12s" % ("做法", "logit 标准差", "最大权重"))
for name, scores in (("不缩放", raw), ("除以 sqrt(d)", scaled)):
    w = nn.softmax(scores, dim=-1).numpy()[0, 0]
    print("%-14s %-14.2f %-12.4f" % (name, float(scores.numpy().std()), w.max()))
print("\n4 个位置「理想」的均匀权重是 0.2500")
```

不缩放时最大权重 **0.9997**——几乎全压在同一个位置上，softmax 已经饱和。除以 `sqrt(d)`
之后降到 **0.6267**，分布仍然有梯度可走。**这就是那一行的全部作用。**

```{code-cell} ipython3
def attention(q, k, v, mask=None):
    """q, k, v: [batch, heads, length, head_dim]"""
    scores = jt.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores + mask
    weights = nn.softmax(scores, dim=-1)
    return jt.matmul(weights, v), weights


out, weights = attention(q, q, q)
print("输出", out.shape)
print("注意力矩阵每一行求和为 1:", bool(np.allclose(weights.sum(-1).numpy(), 1.0, atol=1e-5)))
```

## 2. 因果掩码：让位置 `i` 看不到 `j > i`

生成式模型不能看未来。做法是在 softmax **之前**加上一块上三角的 `-inf`；被加到 `-1e9`
的位置，softmax 之后权重是 0。

```{code-cell} ipython3
def causal_mask(length):
    keep = jt.tril(jt.ones((length, length)))       # 下三角为 1
    return (keep - 1.0) * 1e9                       # 上三角变成 -1e9


mask = causal_mask(4)
_, masked_weights = attention(q, q, q, mask.reshape(1, 1, 4, 4))
w = masked_weights.numpy()[0, 0]
print(np.round(w, 4))
print("\n上三角（对角线之后）全为 0:", bool(np.allclose(np.triu(w, 1), 0, atol=1e-6)))
print("每一行仍然求和为 1:", bool(np.allclose(w.sum(-1), 1, atol=1e-5)))
```

**注意掩码是「加」上去的，不是「乘」上去的。** 用 `-1e9` 而不是 `-inf`，是因为 `softmax`
的实现里 `-inf` 会产出 `nan`（`0 * inf`）。这个细节不写出来不会报错，只会在某些位置上
得到 `nan`。

## 3. 多头注意力

多个头让同一层同时关注多种关系：每个头在自己的 `head_dim` 维度上做注意力，最后拼回来。
实现上用一个 `Linear` 一次产生 q、k、v，再用 reshape 把通道轴拆成 `heads × head_dim`。

```{code-cell} ipython3
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "通道数必须能被头数整除"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def execute(self, x, mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)          # -> [3, b, heads, n, head_dim]
        out, _ = attention(qkv[0], qkv[1], qkv[2], mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(out)


layer = MultiHeadAttention(32, 4)
print("多头输出", layer(jt.randn(2, 6, 32)).shape)
```

`assert dim % heads == 0` 这一行是必要的：不整除时 reshape 会静默地算出错误的形状，
而不是报错。

## 4. 位置编码：注意力本身不知道顺序

注意力对输入的**置换是等价的**——把序列打乱，输出只是跟着打乱。所以位置信息必须
**显式注入**。正弦编码不含参数，且能外推到训练时没见过的长度。

```{code-cell} ipython3
def sinusoidal_positions(length, dim):
    position = jt.arange(length).float().reshape(-1, 1)
    scale = jt.exp(-math.log(10000.0) * jt.arange(0, dim, 2).float() / dim)
    angles = position * scale.reshape(1, -1)
    return jt.concat([jt.sin(angles), jt.cos(angles)], dim=-1).reshape(1, length, dim)


positions = sinusoidal_positions(16, 32)
print("位置表", positions.shape)
print("取值范围 %.3f ~ %.3f" % (float(positions.min()), float(positions.max())))
print("同一位置在不同维度上的值不同（不是常量）:",
      bool(float(positions.numpy().std()) > 0.1))
```

给一个「没有位置编码会怎样」的直观对照：把两行位置表分别取第 0 行和第 3 行，
它们必须不同，否则模型无从区分位置——上面的 `std` 检查说的就是这件事。

## 5. 编码器块：pre-norm 残差

现代 Transformer 用的是 **pre-norm**：先归一化，再过子层，最后相加。残差通路因此保持
未归一化，梯度能直接到达浅层。

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
        x = x + self.attn(self.norm1(x), mask)                       # 注意力子层
        return x + self.fc2(nn.gelu(self.fc1(self.norm2(x))))        # 前馈子层


class TinyTransformer(nn.Module):
    def __init__(self, vocab, dim=32, depth=2, heads=4, max_len=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.positions = sinusoidal_positions(max_len, dim)
        self.positions.requires_grad = False        # 位置表是常量，不参与训练
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
print("参数量:", sum(int(np.prod(p.shape)) for p in model.parameters()))
```

`self.positions[:, :n]` 就是「外推」：位置表按 `max_len` 建好，实际序列多长就取多长。

## 6. 在复制任务上训练

任务故意设得很简单：**把输入序列原样输出**。它的价值在于**有已知的满分**——复制任务的
上限就是 100%，所以可以直接说「学会了 / 没学会」，不必盯着 loss 曲线猜。

评估用一批**完全不参与训练**的留出样本。

```{code-cell} ipython3
VOCAB, LENGTH, BATCH = 12, 8, 32
rng = np.random.RandomState(0)


def make_batch(seed=None):
    r = np.random.RandomState(seed) if seed is not None else rng
    return r.randint(0, VOCAB, size=(BATCH, LENGTH)).astype("int32")


held_out = make_batch(seed=999)                     # 留出集


def accuracy(model, tokens=held_out):
    pred = model(jt.array(tokens)).argmax(dim=-1)[0].numpy()
    return float((pred == tokens).mean())


print("没训练时: %.3f   随机猜: %.3f"
      % (accuracy(model), 1.0 / VOCAB))
```

没训练的模型已经比随机猜高（0.219 对 0.083），因为输出层的**偏置**已经偏向出现频率高的
token。这也是为什么「比随机猜高」不能当学会了——要看它能不能到满分。

```{code-cell} ipython3
optimizer = nn.Adam(model.parameters(), lr=3e-3)

t0 = time.time()
for step in range(301):
    tokens = make_batch()
    logits = model(jt.array(tokens))
    loss = nn.cross_entropy_loss(logits.reshape(-1, VOCAB), jt.array(tokens).reshape(-1))
    optimizer.step(loss)
    if step % 100 == 0:
        print("step %3d  loss %.4f  留出集准确率 %.3f"
              % (step, float(loss.numpy()), accuracy(model)))
print("\n训练用时 %.1fs" % (time.time() - t0))
```

```{code-cell} ipython3
pred = model(jt.array(held_out)).argmax(dim=-1)[0].numpy()
token_acc = float((pred == held_out).mean())
seq_acc = float((pred == held_out).all(1).mean())

print("token 级准确率    : %.3f" % token_acc)
print("整条序列全对的样本: %.3f" % seq_acc)
print("输入  ", held_out[0].tolist())
print("预测  ", pred[0].tolist())

assert token_acc > 0.99, "复制任务应该做到接近满分"
assert seq_acc > 0.9, "整条序列也应该基本全对"
print("\n满分是已知的 1.000，所以可以说：它真的学会了 ✅")
```

**这就是「有已知满分」的价值。** 对比 [VAE](vae.md) 里「KL = 0 时采样图看着不错但其实
塌了」——只看损失或只看观感都可能被误导；有一个**已知的上界**时，判断就变成确定的。

## 7. 怎么读这个结果

| 看到的现象 | 含义 |
| --- | --- |
| 准确率到 1.000 且整条序列全对 | 真的学会了（任务上限已知） |
| 准确率卡在 0.2 左右 | 只学会了输出高频 token，注意力没起作用 |
| 准确率上不去、loss 也不降 | 检查位置编码有没有加上、掩码是否加错（比如加了两次） |
| loss 下降但准确率不动 | 多半是评估用的 `argmax` 取错维度（原生 `argmax` 返回二元组） |

第三行值得留意：**掩码加错不会报错**。如果把掩码加在 softmax 之后，权重就不是分布了；
如果对非方阵加了形状不对的掩码，广播会「成功」但算的是另一回事。

## 8. 同一件事，用 PyTorch API 再写一遍

`nn.MultiheadAttention` 在兼容层里直接可用，不必手写多头拆分。**激活必须在创建任何
张量之前**。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)


class BlockT(tnn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = tnn.LayerNorm(dim)
        self.attn = tnn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = tnn.LayerNorm(dim)
        self.mlp = tnn.Sequential(tnn.Linear(dim, 4 * dim), tnn.GELU(), tnn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


block_t = BlockT(32, 4)
x_t = torch.randn(2, 6, 32)
y_t = block_t(x_t)
loss_t = (y_t ** 2).mean()
loss_t.backward()

print("torch 侧输出", tuple(y_t.shape))
print("损失 %.4f" % float(loss_t))
print("梯度回传到 qkv 投影了吗:", block_t.attn.in_proj_weight.grad is not None)
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 多头拆分 | 手写 reshape / transpose | `nn.MultiheadAttention` 内部完成 |
| batch 维度位置 | `(batch, seq, dim)` | 默认 `(seq, batch, dim)`，要加 `batch_first=True` |
| 求导 | `optim.step(loss)` 内部完成 | 显式 `loss.backward()` |

**`batch_first` 是最容易错的一处**：默认布局是 `(seq, batch, dim)`，忘了它代码照样跑，
只是把序列维和 batch 维对调了。

## 9. 检查清单

```{code-block} text
1. 除以 sqrt(head_dim)     —— 不缩放时 softmax 会饱和，梯度消失（实测 max 权重 0.9997）
2. 掩码加在 softmax 之前    —— 加在之后得到的就不是分布了，且不报错
3. 用 -1e9 而不是 -inf     —— softmax 里 -inf 会产出 nan
4. 位置信息必须显式注入     —— 注意力对置换等价，不给位置就没有顺序概念
5. dim 要能被 heads 整除    —— 不整除时 reshape 静默算错，加 assert
6. 位置表设 requires_grad=False —— 它是常量，不该进优化器
7. 留出集 + 已知满分       —— 复制任务的上限是 1.000，判断是确定的而不是猜的
```

## 接下来

* [视觉 Transformer 训练](vit_training.md)：把同一套结构用在图像上。
* [训练 GPT 风格语言模型](gpt2_training.md)：把掩码换成因果的，任务换成预测下一个 token。
* 想换成当前大模型更常用的两个部件，这棵树里已经有实现：把 `nn.LayerNorm` 换成
  `nn.rms_norm(x, weight)`（注意是**函数**，没有 `nn.RMSNorm` 这个类），把正弦位置表换成
  `nn.rotary_embedding(positions, query, key, cos_sin_cache, ...)`。
* 回到 [教程总目录](README.md)。
