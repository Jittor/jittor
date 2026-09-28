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

# 训练一个视觉 Transformer

[从零实现 Transformer](transformer.md) 处理的是 token 序列。图像不是序列——但可以**变成**
序列：把图切成不重叠的小块（patch），每个 patch 当作一个 token。

这一篇把这件事做完，并且在**同一份合成数据集**上真的训练一次（`resnet_training.md` 用的
就是这套数据，方便对照）。然后回答一个具体问题：**位置编码该用可学习的，还是固定的正弦？**
两边的读数会不一样。

这一篇会：

* 说明为什么**一个卷积就是 patch embedding**；
* 讲清 class token 的作用，以及位置编码为什么不能省；
* 真的训练：留出集准确率 **1.00**（随机猜 0.167）；
* 做一次对照：可学习位置 **1.00** 对固定正弦 **0.88**；
* 读 class token 的注意力分布——并**如实说明它现在还看不出什么**；
* 验证 `state_dict` 的完整往返（含 class token 与位置表）。

> 全篇 CPU 可跑。两轮训练合计十几秒（大部分是第一次的算子即时编译）。

```{code-cell} ipython3
import math
import time

import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(11)

IMG, CLASSES = 16, 6
N_TRAIN, N_VAL = 128, 64
CHANCE = 1.0 / CLASSES
```

## 1. 数据

和 [残差网络训练](resnet_training.md) 完全同一套：六类几何图案加噪声，`16×16` 灰度。
用同一份数据是有意的——这样「CNN 与 ViT 谁更好」在这里是同一道题上的比较，而不是两种
不同难度之间的比较。

```{code-cell} ipython3
def make_dataset(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, CLASSES, n)
    x = np.full((n, IMG, IMG), -1.0, dtype="float32")
    for i, c in enumerate(y):
        if c == 0:                                   # 横条
            r = 5 + rng.randint(0, 3); x[i, r:r + 3, 2:14] = 1.0
        elif c == 1:                                 # 竖条
            col = 5 + rng.randint(0, 3); x[i, 2:14, col:col + 3] = 1.0
        elif c == 2:                                 # 方块
            r = 4 + rng.randint(0, 3); x[i, r:r + 6, r:r + 6] = 1.0
        elif c == 3:                                 # 十字
            x[i, 6:10, 2:14] = 1.0; x[i, 2:14, 6:10] = 1.0
        elif c == 4:                                 # 棋盘
            x[i, ::2, ::2] = 1.0
        else:                                        # 对角线
            for k in range(IMG): x[i, k, k] = 1.0
    x += rng.normal(0, 0.15, x.shape).astype("float32")
    return x[:, None, :, :], y


x_train, y_train = make_dataset(N_TRAIN, seed=1)
x_val, y_val = make_dataset(N_VAL, seed=2)
print("训练", x_train.shape, " 验证", x_val.shape, " 随机猜 %.3f" % CHANCE)
```

## 2. 从图像到 token

把图切成 `4×4` 的 patch，就是让一个 **kernel 与 stride 都等于 4 的卷积**去做投影：
每个输出位置恰好看到一块互不重叠的 patch，别的什么也看不到。把空间网格展平，
结果就是 token 序列。

```{code-cell} ipython3
def sinusoidal(length, dim):
    """固定正弦编码：不需要训练，可以外推到训练时没见过的长度。"""
    pos = jt.arange(length).float().reshape(-1, 1)
    scale = jt.exp(-math.log(10000.0) * jt.arange(0, dim, 2).float() / dim)
    angles = pos * scale.reshape(1, -1)
    return jt.concat([jt.sin(angles), jt.cos(angles)], dim=-1).reshape(1, length, dim)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=IMG, patch_size=4, channels=1, width=32,
                 learned_pos=True):
        super().__init__()
        assert image_size % patch_size == 0, "边长要能被 patch 整除"
        self.patches = (image_size // patch_size) ** 2
        self.project = nn.Conv2d(channels, width, patch_size, stride=patch_size)
        self.class_token = jt.zeros((1, 1, width))
        self.learned_pos = learned_pos
        if learned_pos:
            self.positions = jt.randn((1, self.patches + 1, width)) * 0.02
        else:
            self.positions = sinusoidal(self.patches + 1, width)

    def execute(self, images):
        x = self.project(images)
        batch, width = x.shape[0], x.shape[1]
        x = x.reshape((batch, width, -1)).transpose(0, 2, 1)
        tokens = self.class_token.broadcast((batch, 1, width))
        return jt.concat([tokens, x], dim=1) + self.positions


embedding = PatchEmbedding()
print("每张图切成 %d 个 patch，加上 class token 共 %d 个 token"
      % (embedding.patches, embedding.patches + 1))
print("token 张量:", embedding(jt.randn((2, 1, IMG, IMG))).shape)
```

**class token** 是拼在序列最前面的一个可学习向量，它不对应任何一块 patch，因此可以
自由地积累分类头需要的信息；分类时只读它。**位置编码不能省**——注意力对 token 顺序
是置换等价的，不给位置它根本不知道 patch 是怎么排布的。这里同时准备了两种，后面做对照。

## 3. 多头自注意力

一个 `Linear` 一次产生 q、k、v，reshape 成 `(batch, heads, tokens, head_dim)` 后每个头
拿到自己的子空间。缩放系数 `1/sqrt(head_dim)` 的作用在
[从零实现 Transformer](transformer.md) 里量过：不加它，softmax 会饱和。

```{code-cell} ipython3
class SelfAttention(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        assert width % heads == 0, "通道数必须能被头数整除"
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def execute(self, x, return_weights=False):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape((batch, tokens, 3, self.heads, self.head_dim))
        qkv = qkv.transpose(2, 0, 3, 1, 4)               # -> [3, b, heads, tokens, hd]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # nn.bmm 要求 3 维；把 batch 与 heads 两维压在一起
        scores = nn.bmm(query.reshape((-1, tokens, self.head_dim)),
                        key.reshape((-1, tokens, self.head_dim)).transpose(0, 2, 1))
        scores = scores * self.scale
        weights = nn.softmax(scores, dim=-1)
        context = nn.bmm(weights, value.reshape((-1, tokens, self.head_dim)))

        context = context.reshape((batch, self.heads, tokens, self.head_dim))
        context = context.transpose(0, 2, 1, 3).reshape((batch, tokens, width))
        if return_weights:
            return self.out(context), weights.reshape((batch, self.heads, tokens, tokens))
        return self.out(context)


attention = SelfAttention()
output, weights = attention(jt.randn((2, 5, 32)), return_weights=True)
row_sums = weights.sum(-1).numpy()
assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
print("注意力输出", output.shape)
print("每个头的权重行求和为 1:", bool(np.allclose(row_sums, 1.0, atol=1e-5)))
```

## 4. Pre-norm 块与分类头

```{code-cell} ipython3
class Block(nn.Module):
    def __init__(self, width=32, heads=4, expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * expansion),
                                 nn.GELU(),
                                 nn.Linear(width * expansion, width))

    def execute(self, x):
        x = x + self.attention(self.norm1(x))        # 先归一化，再过子层
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self, classes=CLASSES, width=32, depth=2, heads=4, learned_pos=True):
        super().__init__()
        self.embedding = PatchEmbedding(width=width, learned_pos=learned_pos)
        self.blocks = nn.Sequential(*[Block(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def execute(self, images):
        x = self.blocks(self.embedding(images))
        return self.head(self.norm(x)[:, 0])         # 只读 class token


model = TinyViT()
print("logits:", model(jt.randn((4, 1, IMG, IMG))).shape)
print("参数量:", sum(int(np.prod(p.shape)) for p in model.parameters()))
```

`nn.Sequential(*[...])` 用来堆叠块——比 `nn.ModuleList` 少一层循环，写法更短。

## 5. 真的训练一遍

按 [残差网络训练](resnet_training.md) 的同一套流程：按 batch 迭代、**每个 epoch 都在留出集上
评估**、评估前切 `eval()`。

```{code-cell} ipython3
def evaluate(net, images, labels, batch=64):
    net.eval()
    correct = 0
    for i in range(0, len(labels), batch):
        pred = net(jt.array(images[i:i + batch])).numpy().argmax(1)
        correct += int((pred == labels[i:i + batch]).sum())
    net.train()
    return correct / len(labels)


def train(learned_pos=True, epochs=10, batch=32, lr=1e-3, seed=0):
    jt.set_global_seed(seed)
    net = TinyViT(learned_pos=learned_pos)
    optimizer = nn.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(epochs):
        order = np.random.RandomState(epoch).permutation(N_TRAIN)
        for i in range(0, N_TRAIN, batch):
            idx = order[i:i + batch]
            logits = net(jt.array(x_train[idx]))
            optimizer.step(nn.cross_entropy_loss(logits, jt.array(y_train[idx])))
        history.append((evaluate(net, x_train, y_train),
                        evaluate(net, x_val, y_val)))
    return net, history


t0 = time.time()
trained, history = train(learned_pos=True)
print("训练用时 %.1fs（含算子即时编译）\n" % (time.time() - t0))
print("%-6s %s" % ("epoch", " ".join("%5d" % e for e in range(len(history)))))
print("%-6s %s" % ("训练", " ".join("%5.2f" % h[0] for h in history)))
print("%-6s %s" % ("验证", " ".join("%5.2f" % h[1] for h in history)))
print("\n随机猜 = %.3f" % CHANCE)
```

```{code-cell} ipython3
final_train = history[-1][0]
final_val = history[-1][1]
assert final_val > 0.8, "ViT 应该明显学会这个任务"
assert final_val > CHANCE * 4, "要远高于随机猜"
print("最终：训练 %.2f  验证 %.2f（随机猜 %.3f）" % (final_train, final_val, CHANCE))
```

## 6. 对照：可学习位置 vs 固定正弦

位置编码是 ViT 里少数几个能自由选择的设计。两种做法都跑一遍，其它全同。

```{code-cell} ipython3
_, fixed_history = train(learned_pos=False)

print("%-10s %s" % ("epoch", " ".join("%5d" % e for e in range(len(fixed_history)))))
print("%-10s %s" % ("可学习-验证", " ".join("%5.2f" % h[1] for h in history)))
print("%-10s %s" % ("正弦-验证", " ".join("%5.2f" % h[1] for h in fixed_history)))

learned_val = history[-1][1]
fixed_val = fixed_history[-1][1]
assert learned_val >= fixed_val, "可学习位置在这份数据上不应该更差"
print("\n可学习位置 %.2f  vs  固定正弦 %.2f" % (learned_val, fixed_val))
```

在这份数据上，**可学习位置明显更好**（`1.00` 对 `0.88`）。这条结论有它的前提，别直接外推：

* 训练集只有 128 张图，正弦编码没有"从数据里学位置"的能力，而这份数据的判别信息
  **高度依赖绝对位置**（横条在第几行、方块在哪儿），可学习位置正好吃到这个便宜；
* 换成大数据集，两者的差距通常会缩小；需要**外推到更长序列**时，正弦/旋转编码反而有
  优势——可学习表在训练长度之外没有定义。

**结论要跟着数据集走**，这也是为什么这是一次对照实验而不是一条普遍规律。

## 7. class token 在看哪里

注意力权重是 token 上的分布，所以最后一个块里 class token 那一行，说明了每块 patch 对
预测的贡献。读它只要一次额外前向。

```{code-cell} ipython3
trained.eval()
x = trained.embedding(jt.array(x_val[:4]))
for block in trained.blocks[:-1]:
    x = block(x)
last = trained.blocks[-1]
_, weights = last.attention(last.norm1(x), return_weights=True)

class_row = weights[:, :, 0].mean(1).numpy()        # 先对头取平均
print("形状", class_row.shape, "（4 张图 × %d 个 token）" % (trained.embedding.patches + 1))
print("每行求和为 1:", bool(np.allclose(class_row.sum(-1), 1.0, atol=1e-5)))
print("\n第一张图：")
print(np.round(class_row[0], 3))
print("最大权重 %.3f，最小 %.3f，理想均匀是 %.3f"
      % (class_row[0].max(), class_row[0].min(), 1.0 / len(class_row[0])))
```

**别急着解释它。** 这里的分布几乎是平的（实测 `0.047 ~ 0.075`，理想均匀是 `0.059`）——
模型学会了分类，但**没有学会把注意力集中到某几块 patch 上**。10 个 epoch、128 张图，
还不足以让注意力图变得可读。

这本身是个有用的观察：**准确率高不等于内部结构可解释**。想看到清晰的注意力图，需要更大
的数据与更长的训练；而「注意力图好看」也从来不是模型正确的证据。

## 8. `state_dict` 的完整往返

ViT 的状态里有两样容易被忘掉的东西：**class token** 和**位置表**——它们不是
`nn.Linear` 里的权重，但同样是学出来的。

```{code-cell} ipython3
expected = trained(jt.array(x_val)).numpy()
state = {name: np.array(value.numpy(), copy=True)
         for name, value in trained.state_dict().items()}

interesting = [k for k in state if "class_token" in k or "positions" in k]
print("state_dict 共 %d 项，其中非线性的状态：%s" % (len(state), interesting))

restored = TinyViT()
restored.load_parameters(state)
restored.eval()
actual = restored(jt.array(x_val)).numpy()

assert np.allclose(actual, expected, atol=1e-5), "往返必须复现"
print("\n往返后输出一致:", bool(np.allclose(actual, expected, atol=1e-5)))
print("验证准确率（恢复后）: %.2f" % evaluate(restored, x_val, y_val))
```

注意 `positions` 和 `class_token` 是直接建的 `jt` 张量、不是 `nn.Parameter`——它们仍然
会出现在 `state_dict` 里，也要被存下来。少存它们的后果在
[检查点与断点续训](checkpointing.md) 里讲过：**载入不报错，但结果变了。**

## 9. 同一件事，用 PyTorch API 再写一遍

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)

patch_t = tnn.Conv2d(1, 32, 4, stride=4)
attn_t = tnn.MultiheadAttention(32, 4, batch_first=True)
x_t = torch.randn(4, 1, IMG, IMG)

with torch.no_grad():
    patches = patch_t(x_t).reshape(4, 32, -1).transpose(1, 2)     # (b, tokens, width)
    tokens = torch.cat([torch.zeros(4, 1, 32), patches], dim=1)
    out, attn_weights = attn_t(tokens, tokens, tokens, need_weights=True)

print("torch 侧 token 张量", tuple(tokens.shape))
print("torch 侧注意力输出", tuple(out.shape))
print("注意力权重", tuple(attn_weights.shape), "（batch, tokens, tokens）")
print("行求和为 1:", bool(torch.allclose(attn_weights.sum(-1),
                                         torch.ones(4, tokens.shape[1]), atol=1e-5)))
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| patch embedding | `nn.Conv2d(c, w, k, stride=k)` | 同左 |
| 多头注意力 | 手写 reshape + `nn.bmm` | `nn.MultiheadAttention` |
| q/k/v 布局 | 自己拆 `(b, heads, tokens, hd)` | 内部完成，`batch_first=True` |
| 取注意力权重 | 自己返回 `weights` | `need_weights=True` |

## 10. 检查清单

```{code-block} text
1. patch_size 要整除边长     —— 否则 reshape 会静默算错，加 assert
2. 位置编码不能省            —— 注意力对置换等价，不给位置就没有空间概念
3. 分类只读 class token      —— 它不对应任何 patch，专门用来积累全局信息
4. 位置编码的选择要做对照    —— 本篇 1.00 vs 0.88，但结论依赖数据集大小与是否要外推
5. batch 与 heads 要先展平   —— nn.bmm 只吃 3 维，reshape 成 (b*heads, n, hd)
6. 准确率高 ≠ 可解释         —— 本篇的注意力图还接近均匀，不要过度解读
7. state_dict 里有非线性的状态 —— class_token 与位置表也要存，漏了不会报错
```

第 4 条和第 6 条是这一篇真正的收获：**ViT 的「可解释性」和「位置编码怎么选」都不该
凭直觉下结论**——前者要读出来，后者要对照出来。

## 接下来

* [从零实现 Transformer](transformer.md)：注意力本身的机制与实测。
* [训练 GPT 风格语言模型](gpt2_training.md)：同一套块用在文本上。
* [残差网络训练](resnet_training.md)：同一份数据上的 CNN，可以对照。
* 回到 [教程总目录](README.md)。
