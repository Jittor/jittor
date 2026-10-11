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

# 混合精度：一个整数控制整张图

半精度能换来速度与显存，但**不是所有算子都经得起**：矩阵乘在 `float16` 下很自在，
而多项求和、指数、对数值域很宽的运算则不然。

多数框架让你**标注代码区域**（`with autocast():`）。Jittor 走另一条路：一个整数
`jt.flags.auto_mixed_precision_level` 对**整张图**施加策略，决定是按**算子种类**做出的，
而不是按你写的代码行。

这一篇会：

* 实测**十个算子**在不同级别下的输出 dtype，把「每一级到底做了什么」列成一张能读的表；
* **更正一个常见的误解**——`level 4` 收窄的不只是矩阵乘；
* 解释为什么 `exp` 被**单独留住**，并用溢出边界（`ln(65504) = 11.0899`）把它量化；
* 说明为什么参数始终保持 `float32`，以及这为什么是训练不崩的关键；
* 给出「什么时候用哪一级」的读法。

> 全篇在 CPU 上跑，几秒钟。**级别是全局 flag**，每个单元格结束都要复位，
> 否则会污染后面的所有单元格——下面的代码用 `try/finally` 保证这一点。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
print("当前级别:", jt.flags.auto_mixed_precision_level)
print("底层寄存器 amp_reg:", jt.flags.amp_reg)
```

## 1. 级别是图策略

级别是 0 到 6 的整数。不要背文档，**直接问图**：把级别设上，构造一批有代表性的算子，
读回输出的 dtype。

```{code-cell} ipython3
def dtypes_at(level):
    """在指定级别下，构造十个算子并读回输出 dtype。结束后一定复位。"""
    jt.flags.auto_mixed_precision_level = level
    try:
        a = jt.array(np.random.randn(8, 8).astype("float32"))
        w = jt.array(np.random.randn(8, 8).astype("float32"))
        conv_in = jt.array(np.random.randn(2, 3, 8, 8).astype("float32"))
        conv_w = jt.array(np.random.randn(4, 3, 3, 3).astype("float32"))
        bn = nn.BatchNorm(3)
        return {
            "创建数组": str(a.dtype),
            "逐元素乘": str((a * 2).dtype),
            "求和": str(a.sum().dtype),
            "exp": str(jt.exp(a).dtype),
            "log": str(jt.log(a.abs() + 1.0).dtype),
            "matmul": str(nn.matmul(a, w).dtype),
            "conv2d": str(nn.conv2d(conv_in, conv_w).dtype),
            "batchnorm": str(bn(conv_in).dtype),
            "linear": str(nn.Linear(8, 4)(a).dtype),
            "softmax": str(nn.softmax(a, dim=1).dtype),
        }
    finally:
        jt.flags.auto_mixed_precision_level = 0


LEVELS = (0, 4, 5, 6)
table = {level: dtypes_at(level) for level in LEVELS}

print("%-10s %-9s %-9s %-9s %-9s" % ("算子", *["level %d" % lv for lv in LEVELS]))
for name in table[0]:
    print("%-10s %-9s %-9s %-9s %-9s"
          % (name, *[table[lv][name].replace("float", "f") for lv in LEVELS]))
```

这张表是可以直接读的结论，箭头标出每一级**新收窄**的算子：

| | level 4 | level 5 | level 6 |
| --- | --- | --- | --- |
| 新增收窄 | `matmul`、`conv2d`、`batchnorm`、`linear`、`log` | 上面全部 **＋** 创建数组、逐元素、求和、`softmax` | 上面全部 **＋** `exp` |
| 仍保持 float32 | 创建数组、逐元素、求和、`softmax`、`exp` | 只剩 `exp` | 无 |

`level 1~3` 只保留混合精度的记账（供梯度缩放等使用）而尚未收窄任何算子，所以在这张表里
读起来和 `0` 完全一样。

> **一处常见的误读**：`level 4` 常被说成「只收窄矩阵乘」。表里看得很清楚，
> 卷积、批归一化、全连接、`log` 也在这一级收窄了。**别按文档猜，按 dtype 读。**

## 2. 为什么 `exp` 被单独留住

`float16` 能表示的最大值是 **65504**。取对数：

```{code-cell} ipython3
print("float16 上限:", 65504.0)
print("ln(65504)   = %.4f" % np.log(65504.0))
print("也就是说，输入一超过这个数，exp 的结果就溢出成 inf")
```

注意力里的 logit 达到 11~12 是**很常见**的，所以这不是理论问题。看实测：

```{code-cell} ipython3
probe = jt.array(np.array([[10.0, 11.0, 11.5, 12.0]], dtype="float32"))

print("%-9s %s" % ("级别", "exp 的结果"))
exp_results = {}
for level in (0, 4, 5, 6):
    jt.flags.auto_mixed_precision_level = level
    try:
        exp_results[level] = jt.exp(probe).numpy()
    finally:
        jt.flags.auto_mixed_precision_level = 0
    print("%-9s %s" % ("level %d" % level,
                       np.array2string(exp_results[level], precision=1)))

assert np.isfinite(exp_results[4]).all(), "level 4 应该把 exp 留在 float32"
assert not np.isfinite(exp_results[6]).all(), "level 6 应该让 exp 溢出"
```

`level 4` 和 `5` 的结果和 `0` **逐位相同**——`exp` 根本没被收窄。到 `level 6`，`11.5` 和
`12.0` 直接变成 `inf`：不是「一个大数」，而是**失去意义的无穷**。这就是「留住 `exp`」的
全部理由。

## 3. 那 `softmax` 为什么敢收窄

`softmax` 在 `level 5` 之后是 `float16`，上面那段大 logit 却不会溢出。原因是实现里先减去
最大值再取指数——这是数值稳定的标准做法，和精度无关。

```{code-cell} ipython3
big = jt.array(np.array([[10.0, 11.5, 12.0, 0.0]], dtype="float32"))
for level in (0, 4, 6):
    jt.flags.auto_mixed_precision_level = level
    try:
        s = nn.softmax(big, dim=1).numpy()
    finally:
        jt.flags.auto_mixed_precision_level = 0
    print("level %d softmax -> %s  有限=%s"
          % (level, np.array2string(s, precision=5), bool(np.isfinite(s).all())))
```

三行都有限。**但这不代表 `level 6` 安全**：`softmax` 之前通常是一次 `matmul`，而
`matmul` 在 `level 4` 就已经是 `float16` 了——logit 本身可能已经溢出。**逐算子安全
不等于整条链安全。**

## 4. 参数为什么必须留在 float32

收窄的只是**计算**。优化器始终更新全精度权重——这正是许多微小更新不至于被舍入抹掉的
原因。如果参数也降到 `float16`，一次 `1e-4` 量级的更新会比半个 `float16` 的间隔还小，
直接被舍入成 0，训练会静默地停住。

```{code-cell} ipython3
def train(level, steps=8):
    """在指定级别下训练同一个模型，返回损失轨迹与几个 dtype。"""
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

print("参数 dtype:   float32 下 %s   level 4 下 %s" % (full_param, mixed_param))
print("输出 dtype:   float32 下 %s   level 4 下 %s" % (full_out, mixed_out))
print("\n%4s %-12s %-12s" % ("步", "float32", "level 4"))
for i, (a, b) in enumerate(zip(full, mixed)):
    print("%4d %-12.6f %-12.6f" % (i, a, b))

assert full_param == "float32" and mixed_param == "float32", "参数必须留在全精度"
assert full_out == "float32" and mixed_out == "float16", "只有计算收窄"
assert np.allclose(full, mixed, atol=1e-3), "两条轨迹应当贴合"
assert mixed[-1] < mixed[0], "混合精度下也必须真的在下降"

print("\n最大差异: %.2e —— 两条轨迹贴合，混合精度没有改变训练结果"
      % np.abs(np.array(full) - np.array(mixed)).max())
```

判断混合精度有没有改变训练，标准是**损失轨迹贴不贴合 float32 那一轮**，
而不是「有没有变成 `nan`」。只检查有限性的测试会放过一条走偏的轨迹。

## 5. 什么时候用哪一级

| 级别 | 收窄范围 | 什么时候用 |
| --- | --- | --- |
| `0` | 全部 float32 | 对拍、复现、排查数值问题时的基线 |
| `4` | 矩阵乘 / 卷积 / BN / 全连接 / `log` | **默认选择**：`exp` 和求和留在全精度，训练最稳 |
| `5` | 再加数组创建、逐元素、`softmax` | 显存紧张，且已经确认没有大 logit 进 `exp` |
| `6` | 连 `exp` 也收窄 | 只在**明确知道**数值范围安全时用；否则会静默出 `inf` |

在带 Tensor Core 的 GPU 上，加速出现在 **`4`**——因为矩阵乘与卷积正是在这一级被收窄的，
而它们才是真正吃算力的算子。`5` 和 `6` 收窄的那些逐元素算子并不能再带来多少加速，
却显著增加了数值风险。

`jt.flags.amp_reg` 是这层策略底下的**位掩码寄存器**，`level` 是它的高层封装。日常调优
只动 `level` 就够了；需要精确控制哪些算子参与时才去碰 `amp_reg`。

## 6. 和 PyTorch 的 `autocast` 是什么关系

如果你从 PyTorch 迁过来，会习惯 `with autocast():` 这种**按代码区域**的控制方式。
兼容层提供了 `torch.autocast`，但它的行为要看清——它会打印一条明确的警告：

```{code-block} text
RuntimeWarning: torch.autocast(dtype=torch.bfloat16) is approximated by Jittor's torch
compatibility layer: jittor's amp register keeps bfloat16 only when an operand already
is bfloat16; an all-float32 region computes in float16 instead.
```

也就是说：请求 `bfloat16` 时，一个全 float32 的区域**实际会按 float16 计算**。
如果你的代码依赖 `bfloat16` 的数值范围（动态范围比 `float16` 大得多），这一点必须验证，
不能假设。

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 控制方式 | 全局 `jt.flags.auto_mixed_precision_level` | `with torch.autocast(...)` 或同一个 flag |
| 决策粒度 | 算子种类（整图） | 代码区域 |
| 请求 bf16 | 用 `amp_reg` 精确控制 | 近似为 float16，**有警告** |

## 7. 检查清单

```{code-block} text
1. 级别是全局的           —— 每段实验结束都要复位成 0，否则污染后面所有单元格
2. 按 dtype 读，别按文档猜 —— level 4 收窄的是 matmul/conv/BN/linear/log，不止 matmul
3. exp 在 level 4/5 保持 f32 —— ln(65504)=11.0899，注意力 logit 很容易超过
4. 参数永远 float32        —— 否则微小更新会被舍入抹掉，训练静默停住
5. 判断标准是损失轨迹贴合   —— 不是「没有 nan」
6. 逐算子安全 ≠ 整条链安全  —— softmax 安全，但它前面的 matmul 已经可能是 f16
7. 加速来自 level 4        —— 矩阵乘与卷积被收窄；再往上收益小、风险大
```

## 接下来

* [性能分析器](profiler.md)：先用它找出热点，再决定值不值得上混合精度。
* [设备与驻留](device_placement.md)：精度策略与设备放置是两件独立的事。
* [把真实的生态库跑在 Jittor 上](torch_ecosystem.md)：`torch.autocast` 在兼容层里的行为。
* 回到 [教程总目录](README.md)。
