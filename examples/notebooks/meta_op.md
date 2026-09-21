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

# 元算子：用三个算子拼出卷积

Jittor 的算子体系建立在一组很小的原语上，称为**元算子**（meta-operator）。所有内置算子——
包括卷积——都可以用它们拼出来。这一篇会真的拼一个卷积出来，并证明它和内置卷积算的是
同一件事。

理解这一层有两个好处：**想加一个内置算子表达不出来的运算**时知道从哪儿下手；
以及理解 Jittor 为什么快——元算子把一串操作**融合成一个内核**，而不是逐个执行。

这一篇会：

* 说明元算子的三类，以及它们各自是什么映射；
* 用三个元算子拼出 `matmul`，并验证与 `jt.matmul` **逐位相同**；
* 用元算子拼出一个**真正的卷积**，并与 `nn.Conv2d` 对拍到 `3.8e-06`；
* 把 `reindex` 展开成循环，看清它到底在做什么；
* 看**编译器生成的融合内核**——三个算子变成一个，以及它为什么支持自动求导。

> 全篇 CPU 可跑，几秒钟，不需要下载任何数据。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(0)
```

## 1. 元算子的三类

> ![](figs/mop.svg)
> 元算子的层级结构。元算子包含三类算子：重索引算子、重索引化简算子、元素级算子。
> 它们的反向传播算子仍然是元算子；元算子组成常用的深度学习算子，这些深度学习算子
> 再进一步组成模型。

| 类别 | 映射 | 形状 | 例子 |
| --- | --- | --- | --- |
| **重索引** reindex | 一对多 | 输出形状可任意改变 | `broadcast`、`pad`、切片 |
| **重索引化简** reindex-reduce | 多对一 | 沿某些维求和 | `sum`、`prod`、`max` |
| **元素级** elementwise | 一对一 | 输入输出形状必须一致 | 加法、乘法、`exp` |

**所有元算子的反向传播还是元算子**——这是这套体系能自洽的关键。重索引的反向是重索引
化简，反之亦然；元素级算子的反向是它自己（乘以传回来的梯度）。所以「用元算子写一个新
算子」不需要手写 CUDA 反向，[自动求导](autograd_details.md) 会替你完成。

## 2. 三个元算子拼出 matmul

[模型定义与训练](example.md) 里出现过这个例子，这里先用它确认元算子的用法，再往卷积走。

```{code-cell} ipython3
def matmul_meta(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n, m, k], dims=[2])     # 重索引：(n,m) -> (n,m,k)
    b = b.broadcast([n, m, k], dims=[0])     # 重索引：(m,k) -> (n,m,k)
    return (a * b).sum(dim=1)                # 元素级乘 + 重索引化简


A = jt.randn(3, 4)
B = jt.randn(4, 5)
print("与 jt.matmul 的最大差异:", float((matmul_meta(A, B) - jt.matmul(A, B)).abs().max()))
```

差异是 **0.0**——不是「接近」，是完全一致。两次广播把两个矩阵铺成同一个形状，
元素级乘法之后沿第 1 维求和，三步合起来就是矩阵乘。

## 3. 用元算子拼出卷积

卷积和矩阵乘的差别只在**索引方式**：输出位置 `(i1, i2)` 要读输入的一块 `K×K` 窗口。
这正是重索引算子能表达的——索引表达式里允许出现**多个循环变量**。

```{code-cell} ipython3
def conv_meta(x, w):
    """x: (N,H,W,C)，w: (Kh,Kw,C,Kc)。注意是 NHWC 布局。"""
    N, H, W, C = x.shape
    Kh, Kw, _C, Kc = w.shape
    assert C == _C, (x.shape, w.shape)

    # 重索引：输出是 7 维的 (N, H-Kh+1, W-Kw+1, Kh, Kw, C, Kc)
    xx = x.reindex([N, H - Kh + 1, W - Kw + 1, Kh, Kw, C, Kc], [
        "i0",        # 第 0 维直接照搬：batch
        "i1+i3",     # 输出行 i1 加上核内行偏移 i3
        "i2+i4",     # 输出列 i2 加上核内列偏移 i4
        "i5",        # 输入通道
    ])
    ww = w.broadcast_var(xx)                 # 重索引：核权重铺到同样的形状
    return (xx * ww).sum([3, 4, 5])          # 元素级乘 + 对 Kh, Kw, C 求和
```

**索引表达式里的 `i1+i3` 就是卷积的全部秘密。** 输出第 `i1` 行取的是输入第 `i1+i3` 行，
`i3` 遍历核的高度——窗口滑动这件事被表达成了一个加法。

```{code-cell} ipython3
N, H, W, C, Kc, K = 2, 8, 8, 3, 4, 3
x = jt.randn(N, H, W, C)
w = jt.randn(K, K, C, Kc)
y = conv_meta(x, w)
print("输出形状", y.shape, "（H-Kh+1 = %d）" % (H - K + 1))
```

## 4. 它真的是卷积吗：和 `nn.Conv2d` 对拍

形状对不代表算得对。把同一个权重装进 `nn.Conv2d`，比数值。

```{code-cell} ipython3
conv = nn.Conv2d(C, Kc, K)                   # 权重布局 (Kc, C, Kh, Kw)

state = conv.state_dict()
state["weight"].update(jt.array(np.transpose(w.numpy(), (3, 2, 0, 1))).stop_grad())
state["bias"].update(jt.zeros((Kc,)))        # 先把偏置清零，好跟无偏置的实现比
conv.load_state_dict(state)

x_nchw = jt.array(np.transpose(x.numpy(), (0, 3, 1, 2)))   # NHWC -> NCHW
y_ref = np.transpose(conv(x_nchw).numpy(), (0, 2, 3, 1))

diff = float(np.abs(y.numpy() - y_ref).max())
print("与 nn.Conv2d 的最大差异: %.3e" % diff)
print("形状一致:", y.shape == y_ref.shape)
assert diff < 1e-4, "元算子卷积与内置卷积必须一致"
print("✅ 元算子拼出来的真的是卷积")
```

差异 `3.8e-06` 来自浮点累加顺序（求和次序不同），不是算法差异。**上面那三行就是卷积的
完整定义**，没有任何隐藏成分。

顺手确认它能求导——这正是「元算子的反向还是元算子」的意思。

```{code-cell} ipython3
xx = jt.randn(2, 8, 8, 3)
xx.start_grad()
ww = jt.randn(3, 3, 3, 4)
out = conv_meta(xx, ww)

gx = jt.grad(out.sum(), xx)          # 对输入求导
gw = jt.grad(out.sum(), ww)          # 对卷积核求导
print("对输入的梯度", gx.shape, " 非零:", bool((gx.numpy() != 0).any()))
print("对卷积核的梯度", gw.shape, " 非零:", bool((gw.numpy() != 0).any()))
print("\n没有手写任何反向代码——元算子的反向由元算子自己组合出来")
```

## 5. 看编译器做了什么：循环融合

索引表达式到底允许写什么，`help` 里写得比任何二手描述都准。这一段是**交互式输出**
（不是可复现的计算），所以不纳入冒烟运行：

```{code-cell} ipython3
:tags: [interactive, skip-execution]

help(jt.reindex)
```

`x.reindex(...)` 看起来是一次「取数据」，实际上它是**一整组循环**。展开之后是这样的：

```{code-block} python
shape = [N, H-Kh+1, W-Kw+1, Kh, Kw, C, Kc]

# x.reindex 展开成：把输入按索引表达式搬到新数组
xx = np.zeros(shape)
for i0 in range(shape[0]):
  for i1 in range(shape[1]):
    for i2 in range(shape[2]):
      for i3 in range(shape[3]):
        for i4 in range(shape[4]):
          for i5 in range(shape[5]):
            for i6 in range(shape[6]):
              if is_overflow(i0,i1,i2,i3,i4,i5,i6):
                  xx[i0,i1,i2,i3,i4,i5,i6] = 0
              else:
                  xx[i0,i1,i2,i3,i4,i5,i6] = x[i0, i1+i3, i2+i4, i5]

# w.broadcast_var(xx) 展开成：把核权重铺开
ww[...] = w[i3, i4, i5, i6]

# (xx * ww).sum([3,4,5]) 展开成：逐元素相乘再累加
y[i0,i1,i2,i6] += xx[...] * ww[...]
```

三组七层循环分别做三件事。如果**逐个执行**，中间要写两块 `(N,H',W',Kh,Kw,C,Kc)` 大小的
临时数组——尺寸是输出大小的 `Kh*Kw*C` 倍，而且全是多余的读写。

**融合**就是把这三组循环合成一组：

```{code-block} python
y = np.zeros([N, H-Kh+1, W-Kw+1, Kc])
for i0 in range(...):
  for i1 in range(...):
    for i2 in range(...):
      for i3 in range(...):
        for i4 in range(...):
          for i5 in range(...):
            for i6 in range(...):
              if not is_overflow(i0,i1,i2,i3,i4,i5,i6):
                  y[i0,i1,i2,i6] += x[i0,i1+i3,i2+i4,i5] * w[i3,i4,i5,i6]
```

这个七重循环和你手写的朴素卷积**逐字相同**。中间数组消失了，循环可以重新排序、可以
向量化。细节见 [JIT、算子融合与异步执行](jit_and_async.md)。

### 验证它真的被融合了

不要相信描述，去看编译产物。`compile_shapes` 让编译器把形状当常量编进内核，
`profile_scope` 会把运行记录写成一份报告。

```{code-cell} ipython3
jt.flags.compile_options = {"compile_shapes": 1}
with jt.profile_scope() as report:
    _ = conv_meta(x, w).fetch_sync()
jt.flags.compile_options = {}

entry = report[1]
fused_key = entry[0]
print("报告里的算子键（前 200 字符）：")
print(" ", fused_key[:200], "...")

# 真正要验证的事：三个元算子的标记都在同一个算子键里
for token in ("opkey0:reindex", "opkey1:binary", "opkey2:reduce"):
    assert token in fused_key, token
print("\n三个元算子的标记出现在同一个算子键里 —— 它们被融合成了一次执行 ✅")
```

报告里只有一个条目——**三个元算子合成了一次执行**。算子键里能直接读出来源的三个算子：

```{code-block} text
opkey0:reindex   XDIM=4  YDIM=7  INDEX0:i0  INDEX1:i1+i3  INDEX2:i2+i4  INDEX3:i5
opkey1:binary    OP:multiply   YSTRIDED=1  YSMASK=<广播掩码>
opkey2:reduce    OP:add        REDUCE=<对 dim 3,4,5 求和>
```

`INDEX1:i1+i3` 就是第 3 节写的那个索引表达式，`OP:multiply` 与 `OP:add` 是相乘与求和。
**源代码里的三行，在这里变成了一个内核的三个标记。**

生成的 C++ 源码也在报告里，可以直接打开看：

```{code-cell} ipython3
with open(entry[1], "r") as handle:
    generated = handle.read()
print("生成的源码 %d 字符。摘一段宏定义：" % len(generated))
for line in generated.splitlines():
    if line.startswith("#define op0_INDEX") or line.startswith("#define op1_OP") \
            or line.startswith("#define op2_OP") or line.startswith("#define op2_REDUCE"):
        print(" ", line)
```

## 6. 那它比内置卷积快吗？

**不要这样期待。** 测一下就会发现答案取决于配置：

```{code-cell} ipython3
import time


def timed(fn, repeats=5):
    """重复多次取最小值；单次采样在这个栈上噪声很大。"""
    fn().fetch_sync()                                  # 先焐热
    best = float("inf")
    for _ in range(repeats):
        start = time.time()
        fn().fetch_sync()
        best = min(best, (time.time() - start) * 1e3)
    return best


big_x = jt.randn(8, 32, 32, 16) * 0.1
big_w = jt.randn(3, 3, 16, 64) * 0.05

meta_ms = timed(lambda: conv_meta(big_x, big_w))
print("张量 (8,32,32,16) → 3×3 → 64 通道，CPU、关闭 MKL：")
print("  元算子融合卷积  %.1f ms" % meta_ms)

conv_big = nn.Conv2d(16, 64, 3)
conv_big.load_state_dict({
    "weight": jt.array(np.transpose(big_w.numpy(), (3, 2, 0, 1))).stop_grad(),
    "bias": jt.zeros((64,)),
})
big_nchw = jt.array(np.transpose(big_x.numpy(), (0, 3, 1, 2)))
print("  nn.Conv2d       %.1f ms" % timed(lambda: conv_big(big_nchw)))
```

在这台机器上、关闭 MKL 的 CPU 配置里，融合版本反而更快——因为此时 `nn.Conv2d` 也走不了
加速库。**但这不是一条性能结论**：一旦有 MKL 或 GPU，内置卷积会调用专门优化过的实现
（`jt.flags.enable_tuner = 1` 就是打开这类硬件库的开关），元算子写法没有优势。

元算子的价值在**表达力**：

* 内置卷积**表达不出来**的运算——各种卷积变体、自定义窗口、非标准索引——可以用元算子写；
* 写出来就自动**拥有反向**，因为它由元算子组成；
* 写出来就自动**被融合**，不需要手写 CUDA。

## 7. 检查清单

```{code-block} text
1. 三类元算子要分清       —— 重索引(一对多) / 重索引化简(多对一) / 元素级(一对一)
2. 元素级算子形状必须一致   —— 不一致先 broadcast，别指望自动对齐
3. reindex 的索引可含多个循环变量 —— i1+i3 就是卷积窗口滑动
4. 形状对不代表算得对     —— 一定要和 nn.Conv2d 对拍数值
5. 元算子的反向还是元算子   —— 不用手写 grad，但要验证梯度非零
6. 融合要看编译产物       —— compile_shapes + profile_scope，算子键里能读出三个 opkey
7. 不要拿它当性能路径     —— 有加速库时内置实现更快；它的价值是表达力与自动融合
```

第 4 条和第 6 条是这一篇的实操核心：**对拍数值**确认正确，**看编译产物**确认融合。
两者都是「不能靠读代码判断」的事。

## 接下来

* [JIT、算子融合与异步执行](jit_and_async.md)：融合与调度的完整机制。
* [自定义 C++ 与 CUDA 算子](custom_op.md)：元算子表达不出来时，直接写内核。
* [自动求导](autograd_details.md)：元算子的反向是怎么组合出来的。
* 回到 [教程总目录](README.md)。
