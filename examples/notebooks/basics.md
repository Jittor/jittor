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

# 算子与 Var

用 Jittor 训练模型只需要先理解两个概念：

* **Var**：Jittor 的基本数据类型，相当于其它框架里的张量。
* **算子**：作用在 Var 上的运算，写法与 NumPy 基本一致。

这一篇把这两样从头讲一遍。数据在 CPU 还是显卡上、什么时候会被搬动，是下一篇
[设备与驻留](device_placement.md) 的内容。

```{code-cell} ipython3
import time

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0        # 本篇全程在 CPU 上；有显卡时把它改成 1
```

## 1. 创建 Var

最常用的几个构造函数是 `jt.array` / `jt.zeros` / `jt.ones` / `jt.randn` / `jt.arange`：

```{code-cell} ipython3
a = jt.array([1, 2, 3], dtype="float32")
print(a)

z = jt.zeros((2, 3))
o = jt.ones((2, 3))
n = jt.randn((2, 3))
r = jt.arange(5)
f = jt.full((2, 2), 3.5)

print("zeros ", z.shape, z.dtype)
print("ones  ", o.shape, o.dtype)
print("randn ", n.shape, n.dtype)
print("arange", r.shape, r.dtype, "  <- 整数默认是 int32")
print("full  ", f.shape, f.dtype)
```

打印一个 Var 会显示 `jt.Var([...], dtype=...)`，**连数值一起显示**。这件事有个副作用值得
先记住：**显示它需要把数据算出来**，所以 `print` 不是"看一眼"的廉价操作，它和计算一样慢。
只看形状和类型就用 `.shape` / `.dtype`，不要 `print` 整个 Var。

### 默认类型：从 Python / NumPy 过来会被收窄

Jittor 的默认浮点是 **float32**，默认整数是 **int32**。从 NumPy 数组构造时，
64 位的类型会被**收窄**，不会保留：

```{code-cell} ipython3
print("Python 整数列表   ->", jt.array([1, 2, 3]).dtype)
print("Python 浮点列表   ->", jt.array([1.0, 2.0]).dtype)
print("numpy int64       ->", jt.array(np.array([1, 2], dtype="int64")).dtype)
print("numpy float64     ->", jt.array(np.zeros((2, 2), dtype="float64")).dtype)
print("显式要求 float64  ->", jt.array([1.0]).float64().dtype)
```

如果你确实需要 64 位（比如和 NumPy 的高精度计算对拍），要**显式转换**：
`.float64()`、`.int64()`。不写就默认收窄，而且不会给你任何提示。

可用的转换方法有 `float16 / float32 / float64 / int32 / int64 / bool` 等，
写成 `v.float32()` 这种形式。

## 2. 形状、类型与取值

```{code-cell} ipython3
v = jt.randn((2, 3)).float32()

print("形状 shape :", v.shape)
print("类型 dtype :", v.dtype)
print("维数 ndim  :", v.ndim)
print("元素数     :", v.numel())
print("第 0 维长度 :", len(v))
print("v[0] 的形状 :", v[0].shape)
print("v[:, 1] 的形状:", v[:, 1].shape)
```

`v.shape` 打印出来是 `[2,3,]`——它是 Jittor 自己的形状类型，不是 Python 的
`tuple`。想当元组用就 `tuple(v.shape)`。

### 异步：什么时候才真的算

写下一个表达式**不会**立刻算出结果，它只是把算子加进计算图。要拿到数值，用 `.numpy()`
或 `.item()`，它们会等计算完成：

```{code-cell} ipython3
big = jt.randn((4096, 4096))
big.sync()                         # 先让 big 本身准备好

start = time.perf_counter()
s = big.sum()                      # 只建图，不算
build_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
value = s.numpy()                  # 这一步才真的算
compute_ms = (time.perf_counter() - start) * 1000

print("写下 s = big.sum() 用了 %.4f ms" % build_ms)
print("真正算出 s 用了    %.1f ms" % compute_ms)
print("s =", value)
```

建图几乎是免费的，计算要花时间。异步是 Jittor 做算子融合和自动调优的前提，代价是
"什么时候真正算"不完全由你写代码的顺序决定。

几个会**强制等到算完**的操作：`.numpy()`、`.item()`、`张量.sync()`、以及 `print` / `repr`
（因为它要显示数值）。要等**全部**任务用 `jt.sync_all()`——注意 `jt.sync()` 不带参数时
不等任何东西，直接返回，这个坑在 [性能与显存](profiler.md) 里有详细展开。

## 3. 算子

算子的写法与 NumPy 类似：

```{code-cell} ipython3
a = jt.array([1, 2, 3], dtype="float32")
b = jt.array([4, 5, 6], dtype="float32")
c = a + b
print(a, b, c)
print(c.numpy())
```

每个 `jt.xxx(Var, ...)` 形式的算子都有一个等价的方法写法 `Var.xxx(...)`，两者是
**同一个实现**，选顺手的即可：

```{code-cell} ipython3
print(c.max())                 # 等价于 jt.max(c)
print(c.add(a).numpy())        # 等价于 jt.add(c, a)
print(c.min(keepdims=True))    # 等价于 jt.min(c, keepdims=True)
```

具体有哪些算子可以在 `jt.ops` 上查看。在 `jt.ops.xxx` 里找到的算子同样可以用
`jt.xxx` 这个别名调用，而且是**同一个对象**：

```{code-cell} ipython3
for name in ("add", "max", "min", "mul", "reduce"):
    print("jt.%-8s 就是 jt.ops.%-8s: %s"
          % (name, name, getattr(jt, name) is getattr(jt.ops, name)))
```

不过两边不是完全重合：像 `matmul`、`concat`、`stack` 这些只有 `jt.matmul` 这种写法，
`jt.ops` 里没有。**不确定就试 `jt.xxx`**，它覆盖面更广：

```{code-cell} ipython3
print("jt.ops 里有 matmul 吗:", hasattr(jt.ops, "matmul"))
print("jt 上有 matmul 吗    :", hasattr(jt, "matmul"))
```

矩阵乘用 `@`，和 `jt.matmul` 等价：

```{code-cell} ipython3
p = jt.ones((2, 3))
q = jt.ones((3, 4))
print("(2,3) @ (3,4) ->", (p @ q).shape)
```

## 4. 变形与广播

```{code-cell} ipython3
w = jt.arange(6).float32()
print("w              ", w.shape)
print("w.reshape(2,3) ", w.reshape(2, 3).shape)
print("w.reshape(-1,2)", w.reshape(-1, 2).shape, " <- -1 表示这一维自动推出来")
print("w[None]        ", w[None].shape, " <- 前面加一维")
print("w[:, None]     ", w[:, None].shape, " <- 后面加一维")

m = w.reshape(2, 3)
print("m.transpose()  ", m.transpose().shape)
print("m.permute(1,0) ", m.permute(1, 0).shape)
```

广播规则与 NumPy 相同：从最后一维往前对齐，长度相等或其中一个是 1 就能对上。

```{code-cell} ipython3
a = jt.ones((2, 3))
print("(2,3) + (3,)  ->", (a + jt.ones((3,))).shape)
print("(2,3) * (2,1) ->", (a * jt.ones((2, 1))).shape)
```

## 5. 索引与筛选

切片、整数索引、布尔掩码都能用：

```{code-cell} ipython3
s = jt.arange(5).float32()
print("s          :", s.numpy())
print("s[1:4]     :", s[1:4].numpy())
mask = s > 2
print("s > 2      :", mask.numpy(), mask.dtype)
print("s[s > 2]   :", s[mask].numpy())
```

布尔掩码的形状会变——`s[s > 2]` 的长度取决于有多少个 `True`。想保持形状就得用
`jt.where`。

## 6. 归约

```{code-cell} ipython3
t = jt.arange(6).float32().reshape(2, 3)
print("t            :", t.numpy())
print("t.sum()      :", t.sum().numpy(), "形状", t.sum().shape)
print("t.sum(0)     :", t.sum(0).numpy(), " <- 沿第 0 维求和")
print("t.sum(1, keepdims=True):", t.sum(1, keepdims=True).numpy().tolist())
print("t.mean()     :", t.mean().numpy())
print("t.max()      :", t.max().numpy())
```

归约默认**去掉**被归约的那一维（`t.sum()` 得到 0 维）。要保持形状就加
`keepdims=True`——写广播时经常需要。

### `argmax` 需要 `dim`，而且返回两个值

这一点和 NumPy 不一样，容易踩：

```{code-cell} ipython3
t = jt.arange(6).float32().reshape(2, 3)
indices, values = t.argmax(1)
print("argmax(1) 的下标:", indices.numpy())
print("argmax(1) 的取值:", values.numpy())
```

* `dim` 是**必填**的：`t.argmax()` 会直接报 `TypeError`，不会像 NumPy 那样先展平。
* 返回值是 `(下标, 取值)` 二元组，**下标在前**。只想要下标就
  `indices, _ = t.argmax(1)`。

## 7. 改一个已经存在的 Var

要让一个 Var 换成新内容，而不是新建一个名字，用 `update` 或 `assign`：

```{code-cell} ipython3
u = jt.zeros((3,))
u.update(jt.ones((3,)))
print("update 之后的内容:", u.numpy())

v = jt.zeros((3,))
v.assign(jt.ones((3,)))
print("assign 之后的内容:", v.numpy())
```

两种写法都是**就地**改掉原来那个 Var 的内容——名字还指向同一个 Var，不会多出一个新对象。
优化器里更新参数用的就是 `update`，所以读优化器源码时会经常见到它。

> 这两个方法的**意图**不同：`jt.Var.update` 的注释说明它是给参数/全局变量用的，
> 会切断"旧值 → 新值"的梯度联系并在后台更新，而 `assign` 就是单纯地赋值。
> 在本篇这种一步到位的简单例子里，两者对梯度的表现是一样的（都能把梯度传过去）；
> 差异出现在跨越多个训练步、需要防止计算图无限增长的时候。要用哪个，
> 先看 `help(jt.Var.update)`，再用 `jt.grad` 实测一次你关心的那条路径。

## 接下来

* [设备与驻留](device_placement.md)：数据在哪、什么时候会被搬动。
* [模型定义与训练](example.md)：把这些拼成一个能训练的模型。
