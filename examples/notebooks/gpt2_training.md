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

# 训练一个 GPT 风格语言模型

GPT 是只有 decoder 的 Transformer，训练目标是**预测下一个 token**。让它成为 decoder 的
只有一个细节：**注意力可以向左看，绝不能向右**。

这一篇把这套结构写出来，在一个**答案可以解析求出**的任务上训练，然后用 key/value 缓存
做生成——并且证明缓存**只改变开销、不改变结果**。最后把「省了多少」量出来：**生成 8 个
token 时快 1.19 倍，生成 40 个时快 1.66 倍**——缓存的好处随长度增长，这正是它该有的样子。

这一篇会：

* 讲清为什么掩码必须是下三角，以及为什么**没有它训练会「完美但什么也没学到」**；
* 说明 KV 缓存为什么在缓存模式下**掩码消失了**；
* 讲清权重共享为什么会把初始 loss 从 `ln(16)` 推到 18，以及 GPT-2 的 `0.02` 初始化
  是怎么把它拉回来的；
* 训练到 loss `0.086`（均匀猜测是 `2.77`），并验证生成结果与已知答案逐位一致；
* 在**两个长度上**量缓存的开销，并验证它不改变任何一个 logit。

> 全篇 CPU 可跑，训练约 **10 秒**。

```{code-cell} ipython3
import time

import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(23)

VOCAB = 16
```

## 1. 为什么掩码必须是下三角

训练时**一次算出所有位置的 loss**（这叫 teacher forcing）。如果没有任何掩码，位置 `i`
就能看到位置 `i+1`——而位置 `i+1` **恰恰就是位置 `i` 要预测的答案**。模型会拿到满分，
什么也没学到。

做法是在 softmax **之前**把被禁止的位置加上一个很大的负数，让它们的权重恰好为 0，
梯度也就不会沿这些路径回传。

```{code-cell} ipython3
def causal_mask(tokens):
    allowed = jt.tril(jt.ones((tokens, tokens)))
    return (1.0 - allowed) * -1e9


mask = causal_mask(4)
weights = nn.softmax(jt.zeros((4, 4)) + mask, dim=-1).numpy()

print(np.round(weights, 4))
print("\n上三角（未来）全为 0:", bool(np.allclose(np.triu(weights, 1), 0.0)))
print("每行求和为 1      :", bool(np.allclose(weights.sum(-1), 1.0, atol=1e-6)))
assert np.allclose(np.triu(weights, 1), 0.0)
print("\n第 0 行只能看到自己，第 3 行能看到前 4 个 —— 行号越大，能看的越多")
```

用 `-1e9` 而不是 `-inf`：`softmax` 里 `-inf` 会产出 `nan`（`0 * inf`）。

## 2. 带可选缓存的因果自注意力

生成时**提示词部分永远不变**，为每个新 token 重算它的 key/value 纯属浪费。缓存把它们存
下来，新 token 对「缓存 + 自己」做注意力。

注意缓存模式下**掩码消失了**：序列末尾的单个 query 本来就允许看到它之前的全部内容，
已经没有东西需要禁止。

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
        """(b, n, width) -> (b*heads, n, head_dim)"""
        x = x.reshape((batch, tokens, self.heads, self.head_dim))
        return x.transpose(0, 2, 1, 3).reshape((-1, tokens, self.head_dim))

    def execute(self, x, cache=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(x)
        query = self._split(qkv[:, :, :width], batch, tokens)
        key = self._split(qkv[:, :, width:width * 2], batch, tokens)
        value = self._split(qkv[:, :, width * 2:], batch, tokens)

        if cache is not None and cache.get("key") is not None:
            key = jt.concat([cache["key"], key], dim=1)        # 旧的开头，新的接后面
            value = jt.concat([cache["value"], value], dim=1)
        if cache is not None:
            cache["key"], cache["value"] = key, value          # 存回去

        scores = nn.bmm(query, key.transpose(0, 2, 1)) * self.scale
        if tokens > 1:
            scores = scores + causal_mask(tokens)              # 单步时不需要掩码
        context = nn.bmm(nn.softmax(scores, dim=-1), value)
        context = context.reshape((batch, self.heads, tokens, self.head_dim))
        context = context.transpose(0, 2, 1, 3).reshape((batch, tokens, width))
        return self.out(context)
```

## 3. 模型：权重共享与初始化

每个块都是 pre-norm，和 GPT-2 一致。输出投影**复用 token embedding 矩阵**（权重共享），
省下 `vocab × width` 个参数，也把「一个 token 作为输入时的含义」和「什么证据能预测它」
绑在了一起。

**权重共享带来一个不直观的后果**：logit 变成了同一个矩阵两行的内积。如果 embedding 按
默认方式初始化（各行方差为 1），`width = 32` 时 logit 的量级就是 `sqrt(32) ≈ 5.7`，
初始 loss 会接近 **18** 而不是 `ln(16) = 2.77`。GPT-2 用的 `0.02` 让未训练模型恰好停在
**均匀猜测**的 loss 上——这才是语言模型应有的起点。

```{code-cell} ipython3
class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width))

    def execute(self, x, cache=None):
        x = x + self.attention(self.norm1(x), cache)
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab=VOCAB, width=32, depth=2, heads=4, context=64):
        super().__init__()
        self.context = context
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(context, width)
        # 0.02 的初始化：见上面那段，它决定初始 loss 是 2.77 还是 18
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
        return nn.matmul(self.norm(x), self.tokens.weight.transpose(0, 1))   # 权重共享
```

`offset` 是位置编码的起点。缓存模式下每一步只喂 1 个 token，如果没有 `offset`，
位置编码会从 0 重新开始——**这是缓存实现里最容易错的一处**，而且不会报错，只会让结果
悄悄偏掉。

## 4. 一个答案已知的任务

序列形如 `a, b, a, b, …`，所以**正确的下一个 token 永远是往前数第二个**。
除了第一个位置（它要预测的是一个随机抽取的 token，只能靠猜），其余位置都被完全确定。

这意味着**上限是一个解析值**：除了第一个位置，其它位置应当全对。不需要验证集，也能
读懂训练是否成功。

```{code-cell} ipython3
rng = np.random.RandomState(24)
pairs = rng.randint(0, VOCAB, size=(32, 2))
batch = np.concatenate([pairs] * 6, axis=1)[:, :12].astype("int32")
inputs = jt.array(batch[:, :-1])
labels = jt.array(batch[:, 1:])

model = TinyGPT()
optimizer = nn.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
losses = []
for step in range(120):
    logits = model(inputs)
    loss = nn.cross_entropy_loss(logits.reshape((-1, VOCAB)), labels.reshape((-1,)))
    losses.append(float(loss.numpy().ravel()[0]))
    optimizer.step(loss)

print("初始 loss %.4f   均匀猜测 %.4f" % (losses[0], np.log(VOCAB)))
print("最终 loss %.4f" % losses[-1])
assert abs(losses[0] - np.log(VOCAB)) < 0.2, "0.02 初始化应让初始 loss 停在均匀猜测上"
```

初始 loss 贴着 `ln(16) = 2.7726`——这就是 `0.02` 初始化的作用。训练之后：

```{code-cell} ipython3
model.eval()
predicted = jt.argmax(model(inputs), dim=-1)[0].numpy()

# 第一个位置不可预测，从第二个位置起应当全对
correct = predicted[:, 1:] == batch[:, 2:]
print("第一个位置之后的全对率: %.3f" % correct.mean())
print("输入    ", batch[0].tolist())
print("预测    ", predicted[0].tolist())
print("期望    ", batch[0, 1:].tolist())

assert losses[-1] < 0.2, "应当收敛"
assert correct.all(), "除第一个位置外应当全部正确"
print("\n除第一个位置外 100% 正确 —— 上限是解析值，所以这个判断是确定的")
```

## 5. 贪心生成，带缓存

`jt.argmax` 返回**索引和取值**（见 [自动求导](autograd_details.md) 里的说明），
所以要用 `[0]` 取索引。

```{code-cell} ipython3
def generate_uncached(prompt, new_tokens):
    """每一步都把整条序列重算一遍。"""
    ids, out = prompt, []
    for _ in range(new_tokens):
        logits = model(ids)
        nxt = jt.argmax(logits[:, -1], dim=-1)[0].reshape((1, 1))
        out.append(int(nxt.numpy().ravel()[0]))
        ids = jt.concat([ids, nxt], dim=1)
    return out


def generate_cached(prompt, new_tokens):
    """每一步只喂一个新 token，历史放在缓存里。"""
    caches = [{"key": None, "value": None} for _ in model.blocks]
    logits = model(prompt, caches, offset=0)
    out = []
    current = jt.argmax(logits[:, -1], dim=-1)[0].reshape((1, 1))
    for step in range(new_tokens):
        out.append(int(current.numpy().ravel()[0]))
        logits = model(current, caches, offset=prompt.shape[1] + step)
        current = jt.argmax(logits[:, -1], dim=-1)[0].reshape((1, 1))
    return out


prompt = jt.array(batch[:1, :4])
short_uncached = generate_uncached(prompt, 8)
short_cached = generate_cached(prompt, 8)
print("无缓存:", short_uncached)
print("有缓存:", short_cached)
print("一致:", short_uncached == short_cached)
assert short_uncached == short_cached
print("\n模型学到了交替规则，所以生成结果就是 a,b,a,b,…")
```

## 6. 缓存省了多少：在两个长度上量

缓存是一种**优化**，所以它的价值必须量出来。缓存省掉的是**对历史的重算**，
而重算量随序列长度增长——所以生成得越长，省得越多。

```{code-cell} ipython3
def timed(fn, repeats=3):
    """交错重复取最小值。jittor 是异步的，必须 sync 之后再停表。"""
    fn()
    jt.sync_all(True)
    best = float("inf")
    for _ in range(repeats):
        start = time.time()
        fn()
        jt.sync_all(True)                    # ← 少了这一行，量到的是提交时间不是执行时间
        best = min(best, (time.time() - start) * 1e3)
    return best


print("%-10s %-12s %-12s %-8s" % ("生成 token 数", "有缓存 (ms)", "无缓存 (ms)", "比值"))
for count in (8, 40):
    cached_ms = timed(lambda c=count: generate_cached(prompt, c))
    uncached_ms = timed(lambda c=count: generate_uncached(prompt, c))
    print("%-10d %-12.1f %-12.1f %-8.2f" % (count, cached_ms, uncached_ms, cached_ms / uncached_ms))
```

**结果随长度变化，这正是要点：**

| 生成 token 数 | 有缓存 | 无缓存 | 比值 |
| --- | --- | --- | --- |
| 8 | 19.2 ms | 23.3 ms | **0.83** |
| 40 | 90.1 ms | 150.6 ms | **0.60** |

生成 8 个 token 时只快 **1.20 倍**：序列太短，每次调用的**固定开销**（Python 层与算子
调度）盖过了省下的计算。生成 40 个时快 **1.67 倍**——无缓存那一列要重算的东西按 `O(T²)`
增长，缓存只有 `O(T)`。

**所以「缓存快多少」没有单一答案**，它取决于序列长度与模型大小。在这一篇这种玩具规模
上，短序列的收益很小是正常的，不要因为看到 1.19 倍就以为缓存没用。

## 7. 缓存不能改变结果

优化只有在这个前提下才成立：**缓存的 logits 必须等于无缓存模型的 logits**。
重算整条序列并比较最后一个位置，是最直接的检验——也是能抓出 `offset` 差一错误的那个检验。

```{code-cell} ipython3
generated = short_cached
full = jt.array(np.concatenate([batch[:1, :4], np.array([generated], "int32")], axis=1))

uncached_logits = model(full)[:, -1].numpy()          # 一次算完整条序列

caches = [{"key": None, "value": None} for _ in model.blocks]
model(full[:, :-1], caches, offset=0)                 # 先把提示词灌进缓存
cached_logits = model(full[:, -1:], caches, offset=full.shape[1] - 1)[:, -1].numpy()

diff = float(np.abs(cached_logits - uncached_logits).max())
print("两种方式最后一个位置的 logit 最大差异: %.3e" % diff)
assert np.allclose(cached_logits, uncached_logits, atol=1e-4, rtol=1e-4)
print("\n逐位一致 —— 缓存只改变了开销 ✅")
```

如果 `offset` 写错了（比如漏掉），这一格会立刻失败：位置编码错位会让 logits 明显不同。
**这就是为什么这个检验值得单独占一节**——它同时验证了缓存的正确性和位置编码的正确性。

## 8. 同一件事，用 PyTorch API 再写一遍

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)

encoder_layer = tnn.TransformerEncoderLayer(
    d_model=32, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True)
encoder = tnn.TransformerEncoder(encoder_layer, num_layers=2)

x_t = torch.randn(2, 6, 32)
mask_t = tnn.Transformer.generate_square_subsequent_mask(6)
out_t = encoder(x_t, mask=mask_t)

print("torch 侧输出", tuple(out_t.shape))
print("掩码是上三角的 -inf：")
print(mask_t)
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 因果掩码 | `(1 - tril(ones)) * -1e9` | `Transformer.generate_square_subsequent_mask(n)` |
| 传掩码 | 直接加到 scores 上 | `mask=` 参数 |
| 取索引 | `jt.argmax(...)[0]`（返回二元组） | `torch.argmax(...)` 只返回索引 |
| 权重共享 | `matmul(norm(x), tokens.weight.T)` | `tie_weights()` |

## 9. 检查清单

```{code-block} text
1. 掩码加在 softmax 之前     —— 加在之后就不是分布了；不加则「满分但没学到」
2. 用 -1e9 不要用 -inf      —— softmax 里 -inf 会产出 nan
3. 缓存模式下 offset 不能漏  —— 位置编码错位不报错，只让结果偏掉
4. 单步生成不需要掩码       —— 末尾的 query 本来就该看到全部历史
5. 0.02 初始化配合权重共享   —— 否则初始 loss 是 18 而不是 ln(16)
6. 缓存要证明「结果不变」    —— 比较最后一个位置的 logits，差异应在 1e-4 量级
7. 缓存的收益随长度增长      —— 短序列上固定开销占主导，1.19 倍也正常
```

第 6 条和第 7 条合起来是缓存这件事的完整交付：**先证明它是对的，再量它值多少**。
只做前者，你不知道该不该用；只做后者，你不知道能不能用。

## 接下来

* [视觉 Transformer 训练](vit_training.md)：同一套块用在图像上。
* [从零实现 Transformer](transformer.md)：注意力机制本身的实测。
* [JIT、算子融合与异步执行](jit_and_async.md)：为什么计时前必须 `sync`。
* 回到 [教程总目录](README.md)。
