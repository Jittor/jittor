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

# Mixed precision as a graph policy

# 把混合精度当作图策略

Half precision buys speed and memory, but not every operator survives it: a
matrix product is happy in float16, while a sum of many terms or an exponential
is not. Most frameworks let you mark regions of code. Jittor instead takes one
integer, `jt.flags.auto_mixed_precision_level`, and applies a policy to the
whole graph -- the decision is per *operator kind*, not per line. This tutorial
shows what each level actually does, why the useful default holds some
operators back, and that a model trains the same way under it. Everything runs
offline on a CPU.

半精度带来速度与显存收益，但并非所有算子都经得起：矩阵乘在 float16 下很自在，
而多项求和或指数运算则不然。多数框架让你标注代码区域，计图则用一个整数
`jt.flags.auto_mixed_precision_level` 对整张图施加策略——决定是按**算子种类**做出
的，而不是按代码行。本教程展示每个级别的实际行为、为何常用的那一级要留住某些
算子，以及模型在它之下的训练表现。全部内容可在离线 CPU 上运行。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn
```

## What each level does

## 每个级别做了什么

Rather than trusting the documentation, ask the graph. Each level is set,
five representative operators are built, and their output dtypes are read back.
Restoring level `0` afterwards matters: the flag is global, so leaving it set
would change every later cell.

与其相信文档，不如直接问图。下面逐个设置级别，构造五个有代表性的算子，读回它们的
输出 dtype。事后恢复到 `0` 很重要：这个 flag 是全局的，不复位会影响后续所有单元格。

```{code-cell} ipython3
def dtypes_at(level):
    jt.flags.auto_mixed_precision_level = level
    try:
        a = jt.array(np.random.randn(8, 8).astype("float32"))
        return {
            "array": str(a.dtype),
            "multiply": str((a * 2).dtype),
            "sum": str(a.sum().dtype),
            "exp": str(jt.exp(a).dtype),
            "matmul": str(nn.matmul(a, a.transpose(0, 1)).dtype),
        }
    finally:
        jt.flags.auto_mixed_precision_level = 0


table = {level: dtypes_at(level) for level in (0, 4, 5, 6)}

assert table[0]["matmul"] == "float32"
assert table[4]["matmul"] == "float16"
assert table[4]["sum"] == "float32"
assert table[5]["array"] == "float16" and table[5]["exp"] == "float32"
assert table[6]["exp"] == "float16"

for level, row in table.items():
    print(level, row)
```

Level `4` narrows only the matrix product. Level `5` additionally creates arrays
in half precision, so elementwise work follows its inputs down. Level `6`
narrows even the exponential. Levels `1` to `3` reserve the mixed-precision
bookkeeping without narrowing anything yet, so they read like level `0` here.

`4` 只收窄矩阵乘。`5` 额外让 array 以半精度创建，逐元素运算随输入一起下降。`6`
连指数运算也收窄。`1` 到 `3` 只保留混合精度的记账而尚未收窄任何算子，因此在这里
读起来与 `0` 相同。

## Why the exponential is held back

## 为什么指数运算被留住

float16 tops out at `65504`, so `exp` overflows once its input passes
`ln(65504) = 11.0899` -- a value an attention logit reaches easily. Level `4`
keeps `exp` in float32 for exactly this reason; level `6` does not, and the
result is `inf` rather than a large number.

float16 的上限是 `65504`，因此输入超过 `ln(65504) = 11.0899` 后 `exp` 即溢出——注意力
的 logit 很容易达到这个量级。`4` 正是为此把 `exp` 留在 float32；`6` 不留，于是结果
是 `inf` 而不是一个大数。

```{code-cell} ipython3
probe = jt.array(np.array([[10.0, 11.5, 12.0]], dtype="float32"))
results = {}
for level in (4, 6):
    jt.flags.auto_mixed_precision_level = level
    try:
        results[level] = jt.exp(probe).numpy()
    finally:
        jt.flags.auto_mixed_precision_level = 0

assert np.isfinite(results[4]).all()
assert not np.isfinite(results[6]).all()
assert float(np.log(65504.0)) < 11.5

print("level 4:", results[4])
print("level 6:", results[6])
```

## Training under level 4

## 在 level 4 下训练

The parameters stay float32. Only the compute narrows -- the optimizer keeps
updating full-precision weights, which is what stops many small updates from
being rounded away. The loss trajectory is the check that matters: it should
track the float32 run rather than merely stay finite.

参数保持 float32，收窄的只是计算——优化器始终更新全精度权重，这正是许多微小更新
不至于被舍入抹掉的原因。真正该看的是 loss 轨迹：它应当贴合 float32 那一轮，而不是
仅仅保持有限。

```{code-cell} ipython3
def train(level, steps=5):
    jt.set_global_seed(17)
    x = jt.array(np.random.RandomState(0).randn(32, 16).astype("float32"))
    y = jt.array(np.random.RandomState(1).randn(32, 8).astype("float32"))
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 8))
    optimizer = nn.SGD(model.parameters(), lr=5e-2)
    jt.flags.auto_mixed_precision_level = level
    try:
        losses = []
        for _ in range(steps):
            loss = ((model(x) - y) ** 2).mean()
            losses.append(float(loss.numpy().reshape(-1)[0]))
            optimizer.step(loss)
        return losses, str(model[0].weight.dtype), str(model(x).dtype)
    finally:
        jt.flags.auto_mixed_precision_level = 0


full, full_param, full_out = train(0)
mixed, mixed_param, mixed_out = train(4)

assert full_param == "float32" and mixed_param == "float32"
assert full_out == "float32" and mixed_out == "float16"
assert np.allclose(full, mixed, atol=1e-3)
assert mixed[-1] < mixed[0]

print("float32 :", [round(v, 5) for v in full])
print("level 4 :", [round(v, 5) for v in mixed])
print("largest difference: %.2e" % np.abs(np.array(full) - np.array(mixed)).max())
```

## What to try next

## 接下来可以尝试

* Raise the level to `5` and watch which activations change dtype.
* Set `jt.flags.use_cuda = 1`; on a tensor-core GPU level `4` is where the
  speedup appears, since that is the level the matrix products narrow at.
* Feed a model deliberately large logits and compare level `4` with level `6`.

* 把级别提高到 `5`，观察哪些激活值的 dtype 发生变化。
* 设置 `jt.flags.use_cuda = 1`；在带 tensor core 的 GPU 上，加速正出现在 `4`，
  因为矩阵乘正是在这一级被收窄的。
* 给模型刻意输入很大的 logit，比较 `4` 与 `6` 的差别。
