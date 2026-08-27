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

# Train a GPT-style language model

# 训练一个 GPT 风格语言模型

A GPT is a decoder-only transformer trained to predict the next token. What
makes it a decoder is one detail: attention may look left but never right.
This tutorial builds such a model in native Jittor, trains it on a task whose
answer is knowable in closed form, then generates text with a key/value cache
and proves the cache changes nothing but the cost. Everything runs offline on a
CPU.

GPT 是只有 decoder 的 Transformer，训练目标是预测下一个 token。使其成为 decoder
的只有一个细节：注意力可以向左看，但绝不能向右。本教程用原生计图搭建这样的模型，
在一个答案可解析求解的任务上训练，然后用 key/value 缓存生成文本，并证明缓存只
改变开销、不改变结果。全部内容可在离线 CPU 上运行。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.set_global_seed(23)
```

## Why the mask is lower triangular

## 为什么掩码是下三角

Training computes the loss at every position at once. Without a mask, position
`i` would attend to position `i + 1` -- and position `i + 1` is precisely the
answer position `i` is asked to predict. The model would score perfectly and
learn nothing. Setting the forbidden scores to `-inf` *before* the softmax
makes their weight exactly zero, so no gradient flows back along them either.

训练时会一次性计算所有位置的 loss。若没有掩码，位置 `i` 会注意到位置 `i + 1`
——而位置 `i + 1` 恰恰是位置 `i` 要预测的答案。模型会得到完美分数却什么也学不到。
在 softmax **之前**把被禁止的分数置为 `-inf`，它们的权重恰好为零，梯度也不会沿
这些路径回传。

```{code-cell} ipython3
def causal_mask(tokens):
    allowed = jt.tril(jt.ones((tokens, tokens)))
    return (1.0 - allowed) * -1e9


mask = causal_mask(4)
weights = nn.softmax(jt.zeros((4, 4)) + mask, dim=-1).numpy()

assert np.allclose(np.triu(weights, 1), 0.0)
assert np.allclose(weights.sum(-1), 1.0, atol=1e-6)
print("row 0 sees:", np.round(weights[0], 3))
print("row 3 sees:", np.round(weights[3], 3))
```

## Causal self-attention with an optional cache

## 带可选缓存的因果自注意力

During generation the prompt never changes, so recomputing its keys and values
for every new token is pure waste. The cache stores them and the new token
attends to the concatenation. Note the mask disappears in cached mode: a single
query at the end of the sequence is already allowed to see everything before
it, so there is nothing left to forbid.

生成时提示词不再变化，为每个新 token 重算它的 key/value 纯属浪费。缓存把它们存
下来，新 token 对拼接后的结果做注意力。注意缓存模式下掩码消失了：序列末尾的单个
query 本来就允许看到它之前的全部内容，已无可禁止。

```{code-cell} ipython3
class CausalSelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def _split(self, x, batch, tokens):
        x = x.reshape((batch, tokens, self.heads, self.head_dim))
        return x.transpose(0, 2, 1, 3).reshape((-1, tokens, self.head_dim))

    def execute(self, x, cache=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(x)
        query = self._split(qkv[:, :, :width], batch, tokens)
        key = self._split(qkv[:, :, width:width * 2], batch, tokens)
        value = self._split(qkv[:, :, width * 2:], batch, tokens)
        if cache is not None and cache.get("key") is not None:
            key = jt.concat([cache["key"], key], dim=1)
            value = jt.concat([cache["value"], value], dim=1)
        if cache is not None:
            cache["key"], cache["value"] = key, value
        scores = nn.bmm(query, key.transpose(0, 2, 1)) * self.scale
        if tokens > 1:
            scores = scores + causal_mask(tokens)
        context = nn.bmm(nn.softmax(scores, dim=-1), value)
        context = context.reshape((batch, self.heads, tokens, self.head_dim))
        context = context.transpose(0, 2, 1, 3).reshape((batch, tokens, width))
        return self.out(context)
```

## The model

## 模型

Each block is pre-norm, matching GPT-2. The output projection reuses the token
embedding matrix -- weight tying -- which removes `vocab * width` parameters and
ties "what a token means as input" to "what evidence predicts it as output".

Weight tying is also why the embedding initialization matters more than usual.
A logit is now an inner product of two rows of the same matrix, so unit-variance
rows across `width = 32` produce logits of order `sqrt(32)`, and the first loss
comes out near 18 instead of `ln(16) = 2.77`. GPT-2's `0.02` keeps the untrained
model at the uniform-guess loss, which is where a language model should start.

每个块都是 pre-norm，与 GPT-2 一致。输出投影复用 token embedding 矩阵，即权重
共享，既省下 `vocab * width` 个参数，也把"token 作为输入时的含义"与"何种证据预测
出它"绑定在一起。

权重共享也让 embedding 的初始化比平时更关键。此时 logit 是同一矩阵两行的内积，
若各行方差为 1、`width = 32`，logit 的量级就是 `sqrt(32)`，初始 loss 会接近 18 而
不是 `ln(16) = 2.77`。GPT-2 采用的 `0.02` 让未训练模型停在均匀猜测的 loss 上，
这正是语言模型应有的起点。

```{code-cell} ipython3
class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def execute(self, x, cache=None):
        x = x + self.attention(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab=16, width=32, depth=2, heads=4, context=12):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(context, width)
        self.tokens.weight = jt.init.gauss((vocab, width), "float32", 0.0, 0.02)
        self.positions.weight = jt.init.gauss((context, width), "float32", 0.0, 0.02)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def execute(self, ids, caches=None, offset=0):
        length = ids.shape[1]
        steps = jt.array(np.arange(offset, offset + length).astype("int32"))
        x = self.tokens(ids) + self.positions(steps).unsqueeze(0)
        for index, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[index])
        return nn.matmul(self.norm(x), self.tokens.weight.transpose(0, 1))


model = TinyGPT()
print("logits:", model(jt.array(np.zeros((2, 5), "int32"))).shape)
```

## A task with a known answer

## 一个答案已知的任务

The sequences are `a, b, a, b, ...`, so the correct next token is always the one
from two positions back. Every position is therefore determined -- except the
very first, which is asked to predict a token drawn at random and can only
guess. A model that has learned the rule is exactly right everywhere else, so
the trajectory is readable without a validation set, and the ceiling is a known
number rather than an empirical one.

序列形如 `a, b, a, b, ...`，因此正确的下一个 token 永远是往前数第二个。除了第一个
位置——它要预测的是随机抽取的 token，只能靠猜——其余位置都被完全确定。学会规则的
模型在其余位置上应当全对，因此无需验证集也能读懂轨迹，而且上限是已知的解析值而非
经验值。

```{code-cell} ipython3
rng = np.random.RandomState(24)
pairs = rng.randint(0, 16, size=(32, 2))
batch = np.concatenate([pairs] * 6, axis=1)[:, :12].astype("int32")
inputs = jt.array(batch[:, :-1])
labels = jt.array(batch[:, 1:])

model = TinyGPT()
optimizer = nn.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
losses = []

for step in range(120):
    logits = model(inputs)
    loss = nn.cross_entropy_loss(logits.reshape((-1, 16)), labels.reshape((-1,)))
    losses.append(float(loss.numpy().reshape(-1)[0]))
    optimizer.step(loss)

model.eval()
predicted = jt.argmax(model(inputs), dim=-1)[0].numpy()

assert abs(losses[0] - np.log(16)) < 0.2
assert losses[-1] < 0.2
assert (predicted[:, 1:] == batch[:, 2:]).all()
print("loss: %.4f -> %.4f  (uniform guess: %.4f)" % (
    losses[0], losses[-1], np.log(16)))
print("accuracy after the unpredictable first position: 100%")
```

## Greedy generation with a cache

## 带缓存的贪心生成

`jt.argmax` returns indices *and* values, unlike the single-return convention
some other frameworks use; unpack both. Each step feeds exactly one token and
passes `offset` so the position embedding stays correct even though the model
only sees the newest position.

`jt.argmax` 返回索引**和**取值，与某些框架只返回一个的约定不同，需要同时解包。
每一步只输入一个 token，并传入 `offset`，这样即使模型只看到最新位置，位置编码
依然正确。

```{code-cell} ipython3
prompt = jt.array(batch[:1, :4])
caches = [{"key": None, "value": None} for _ in model.blocks]

logits = model(prompt, caches, offset=0)
generated = []
current = jt.argmax(logits[:, -1], dim=-1)[0].reshape((1, 1))
for step in range(4):
    generated.append(int(current.numpy().reshape(-1)[0]))
    logits = model(current, caches, offset=4 + step)
    current = jt.argmax(logits[:, -1], dim=-1)[0].reshape((1, 1))

expected = [int(v) for v in batch[0, 4:8]]
assert generated == expected
print("generated:", generated)
print("expected :", expected)
```

## The cache must not change the answer

## 缓存不得改变结果

A cache is an optimization, so it earns its place only if the logits are the
ones the uncached model would have produced. Recomputing the whole sequence and
comparing the final position is the direct check -- and the one that catches an
off-by-one in the position offset.

缓存属于优化手段，唯有在 logits 与无缓存模型一致时才站得住脚。重算整段序列并
比较最后一个位置是最直接的检验，也能抓出位置 offset 的 off-by-one。

```{code-cell} ipython3
full = jt.array(np.concatenate([batch[:1, :4], np.array([generated], "int32")], axis=1))
uncached = model(full)[:, -1].numpy()

caches = [{"key": None, "value": None} for _ in model.blocks]
model(full[:, :-1], caches, offset=0)
cached = model(full[:, -1:], caches, offset=full.shape[1] - 1)[:, -1].numpy()

assert np.allclose(cached, uncached, atol=1e-4, rtol=1e-4)
print("max logit difference:", float(np.abs(cached - uncached).max()))
```

## What to try next

## 接下来可以尝试

* Replace greedy decoding with temperature sampling and top-k filtering.
* Trim the cache to the last `context` tokens to generate past the window.
* Set `jt.flags.use_cuda = 1`; the model and generation code stay unchanged.

* 把贪心解码换成温度采样加 top-k 过滤。
* 将缓存裁剪到最后 `context` 个 token，以便生成超出窗口的内容。
* 设置 `jt.flags.use_cuda = 1`；模型与生成代码无需改动。
