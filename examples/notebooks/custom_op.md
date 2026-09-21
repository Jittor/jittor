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

# 自定义算子：从几行源码到完整的 Op

Jittor 的算子不是写死的——你可以在 Python 里附上一段 C/CUDA 源码，**运行时就地编译**，
然后像内置算子一样调用它。这一篇讲清三条路各自适合什么，以及一个几乎所有人第一次都会
踩到的坑：**反向传播必须自己写**。

这一篇会：

* 用 `jt.code` **在 CPU 上真的跑起来**一个自定义算子（不是伪代码）；
* 列出源码里**能用的变量名**，并指出一个实测会编译失败的名字；
* 用实验证明 `jt.code` **不会自动提供反向**——梯度是 0，而且不报错；
* 用 `jt.Function` 写出手写的反向，验证它算得对，并接进一个可训练模块；
* 给出同时支持 CPU 与 CUDA 的双后端写法；
* 说明 `jt.compile_custom_op`（完整 Op 类）用在什么场合。

> 第 1~5 节**在 CPU 上可跑**，几秒钟。带 CUDA 的小节标了 `skip-execution`，
> 因为没有显卡的环境跑不了。

```{code-cell} ipython3
import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
```

## 1. 三条路，怎么选

| 方式 | 适合 | 需要写反向吗 | 成本 |
| --- | --- | --- | --- |
| **组合已有算子** | 能用现有算子搭出来的 | 不用，自动微分覆盖 | 最低 —— **先考虑这个** |
| `jt.code` | 几行 C/CUDA 就能写清的内核 | **要** | 低（一段源码字符串） |
| `jt.Function` | 上面的内核 + 自定义反向 | **要**（写在 `grad` 里） | 中 |
| `jt.compile_custom_op` | 需要在 C++ 侧管理状态、声明多个输出与 `jit_prepare` 的复杂算子 | 由 Op 自身的反向规则决定 | 最高 |

**第一行最重要。** 很多「需要自定义算子」的场景，其实是几个内置算子拼一下——那种情况
用组合写法还能白拿自动微分。只有在**性能**或**内置算子表达不出来**时才往下走。

## 2. `jt.code`：写一段源码就是一个算子

`jt.code` 接收输出的形状与 dtype、输入列表，以及**至少一个后端**的源码字符串。
CPU 后端用 `cpu_src`，CUDA 后端用 `cuda_src`。

```{code-cell} ipython3
a = jt.array(np.arange(8, dtype="float32"))
print("输入:", a.numpy())


def add_five(x):
    """返回 x + 5；内核是下面这几行 C 代码，运行时编译。"""
    return jt.code(
        x.shape,
        x.dtype,
        [x],
        cpu_src="""
            for (int i = 0; i < in0_shape0; ++i)
                out0_p[i] = in0_p[i] + 5.0f;
        """,
    )


b = add_five(a)
print("输出:", b.numpy())
assert np.allclose(b.numpy(), a.numpy() + 5)
print("自定义算子算对了 ✅")
```

第一次调用会触发即时编译，比之后慢——这是 [JIT、算子融合与异步执行](jit_and_async.md)
讲的那个过程。

### 源码里能用的名字

| | CPU 侧（`cpu_src`） | CUDA 侧（`cuda_src`） |
| --- | --- | --- |
| 第 i 个输入指针 | `in0_p`、`in1_p`… | 同左（设备指针） |
| 第 i 个输出指针 | `out0_p`、`out1_p`… | 同左 |
| 第 i 个输入的第 k 维长度 | `in0_shape0`、`in0_shape1`… | `in0->shape[k]` 或 `x_shape0` |
| 输入元素总数 | —— | `in0->num` |
| 输出元素类型 | —— | `out0_type` |
| 指针重命名 | —— | `@alias(x, in0)` |

**一个实测会编译失败的名字：`in0_num`。** 在 CPU 侧它没有定义，编译器会报
`'in0_num' was not declared in this scope; did you mean 'in0_dim'`。CPU 侧要按维度写
`in0_shape0`，或者在进入循环前自己算总数。这个错误信息相当直白，但第一次遇到时容易
以为是用法不对。

### 多个输出

输出形状与 dtype 都传列表，返回值就是一个**元组**。

```{code-cell} ipython3
m = jt.array(np.arange(6, dtype="float32").reshape(2, 3))
doubled, negated = jt.code(
    [m.shape, m.shape], [m.dtype, m.dtype], [m],
    cpu_src="""
        for (int i = 0; i < in0_shape0; ++i)
            for (int j = 0; j < in0_shape1; ++j) {
                out0_p[i * in0_shape1 + j] =  in0_p[i * in0_shape1 + j] * 2.0f;
                out1_p[i * in0_shape1 + j] = -in0_p[i * in0_shape1 + j];
            }
    """,
)

print("输入  :", m.numpy().tolist())
print("×2    :", doubled.numpy().tolist())
print("取负  :", negated.numpy().tolist())
assert np.allclose(doubled.numpy(), m.numpy() * 2)
assert np.allclose(negated.numpy(), -m.numpy())
print("\n两个输出都对：返回的是元组，按顺序对应 outputs 列表")
```

## 3. 关键：`jt.code` 不提供反向

这是整篇最要紧的一节。用 `jt.code` 写出来的算子，**自动微分不知道它是干什么的**——
梯度会静默地变成 0。

```{code-cell} ipython3
v = jt.array(np.array([1.0, 2.0], dtype="float32"))
v.start_grad()

tripled = jt.code(v.shape, v.dtype, [v], cpu_src="""
    for (int i = 0; i < in0_shape0; ++i)
        out0_p[i] = in0_p[i] * 3.0f;
""")

g = jt.grad(tripled.sum(), v)
print("前向结果:", tripled.numpy())
print("梯度     :", g.numpy(), " ← 数学上应该是 [3, 3]")
```

**梯度是 0，而且没有任何警告。** 如果把这个算子放进一个模型，那个位置的参数**永远
不会更新**，训练看起来一切正常，只是学不动。这和
[自动求导](autograd_details.md) 里「`jt.grad` 什么时候静默返回全 0」是同一类陷阱。

所以在自定义算子上你必须自己回答一个问题：**反向传播是什么？**

## 4. `jt.Function`：把反向写出来

`jt.Function` 是一个只做两件事的类：`execute` 定义前向，`grad` 定义反向。
`grad` 接收「输出端的梯度」，返回「每个输入对应的梯度」。

看一个完整的例子：`y = x²`。

```{code-cell} ipython3
class Square(jt.Function):
    """y = x^2。前向用自定义核，反向手写。"""

    def execute(self, x):
        self.saved = x                       # 前向里存下反向需要的东西
        return jt.code(x.shape, x.dtype, [x], cpu_src="""
            for (int i = 0; i < in0_shape0; ++i)
                out0_p[i] = in0_p[i] * in0_p[i];
        """)

    def grad(self, grad_output):
        # d(x^2)/dx = 2x，再乘上从输出端传回来的梯度
        return grad_output * 2.0 * self.saved


w = jt.array(np.array([1.0, 2.0, 3.0], dtype="float32"))
w.start_grad()
out = Square()(w)
g = jt.grad(out.sum(), w)

print("x^2     ->", out.numpy(), " (期望 [1, 4, 9])")
print("梯度    ->", g.numpy(), " (期望 [2, 4, 6])")
assert np.allclose(out.numpy(), [1.0, 4.0, 9.0])
assert np.allclose(g.numpy(), [2.0, 4.0, 6.0])
print("\n手写的反向是对的 ✅")
```

注意 `grad_output` 是**链式法则里已经传回来的那一部分**。你只需要乘上自己这层的局部
导数——不需要从头算整条链。

### 接进一个可训练模块

手写的反向必须能和优化器串起来，否则只是好看。

```{code-cell} ipython3
class SquareLayer(nn.Module):
    """只有一层的玩具模块：y = (x * w)^2。"""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(jt.ones((1,)) * 0.5)

    def execute(self, x):
        return Square()(x * self.w)


layer = SquareLayer()
sample = jt.array(np.array([[2.0]], dtype="float32"))
optimizer = nn.SGD(layer.parameters(), lr=0.1)


def current_loss():
    return Square()(sample * layer.w).sum()


before_w = float(layer.w.numpy().reshape(-1)[0])
before_loss = float(current_loss().numpy().reshape(-1)[0])
for _ in range(20):
    optimizer.step(current_loss())
after_w = float(layer.w.numpy().reshape(-1)[0])
after_loss = float(current_loss().numpy().reshape(-1)[0])

print("w    : %.4f -> %.4f" % (before_w, after_w))
print("loss : %.4f -> %.4f" % (before_loss, after_loss))
assert after_w != before_w, "参数必须真的被更新"
assert after_loss < before_loss, "损失必须下降"
print("\n自定义算子的梯度确实驱动了优化器 ✅")
```

如果第 3 节的实验里梯度是 0，这里 `w` 会一动不动——这就是自定义算子最典型的失败方式：
**不是报错，是不学。**

## 5. 同时支持 CPU 与 CUDA

真实项目里两个后端都要给。CPU 侧是直白的循环，CUDA 侧是**内核定义 + 启动代码**。
`cuda_src` 给的是启动函数的主体，里面用的指针名字和 CPU 侧一致。

```{code-cell} ipython3
:tags: [cuda, skip-execution]

def double_any(x):
    """两个后端都实现同一个算子：y = 2x。"""
    return jt.code(
        x.shape, x.dtype, [x],
        cpu_src="""
            for (int i = 0; i < in0_shape0; ++i)
                out0_p[i] = in0_p[i] * 2.0f;
        """,
        cuda_src="""
            __global__ void double_kernel(const in0_type* x, out0_type* y, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) y[i] = x[i] * 2.0f;
            }
            const int n = in0->num;
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            double_kernel<<<blocks, threads>>>(in0_p, out0_p, n);
            CHECK(0 == cudaGetLastError());
        """,
    )
```

CUDA 侧有几处和 CPU 侧不同，值得记下来：

* 线程总数要从 `in0->num` 拿（CPU 侧没有这个名字）；
* 指针名仍然是 `in0_p` / `out0_p`，类型名是 `in0_type` / `out0_type`；
* 线程数要自己算，并且**必须**检查 `cudaGetLastError()`——否则内核静默不执行，你会
  拿到一个没被写过的输出；
* 元素多的时候要写 grid-stride 循环，别让 `blocks` 无限大。

## 6. `jt.compile_custom_op`：完整的 Op 类

需要在 C++ 侧管理输出、声明 `jit_prepare`、或者复用 `Op` 基类的机制时，用
`jt.compile_custom_op(header, source, op_name, warp=...)`。它接收一个头文件和一个实现，
JIT 编译成一个新的算子类。

下面这个算子不读输入，直接产出 `0, 1, 2, …`。CPU 与 CUDA 两份实现都写在同一个
`#ifdef` 结构里，这是 Jittor 内核代码的通用写法。

```{code-cell} ipython3
:tags: [cuda, skip-execution]

header = """
#pragma once
#include "core/op.h"

namespace jittor {

struct CustomOp : Op {
    Var* output;
    CustomOp(NanoVector shape, NanoString dtype=ns_float32);

    const char* name() const override { return "custom"; }
    DECLARE_jit_run;
};

} // jittor
"""

src = """
#include "core/var.h"
#include "custom_op.h"

namespace jittor {
#ifndef JIT
CustomOp::CustomOp(NanoVector shape, NanoString dtype) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(shape, dtype);
}

void CustomOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "T", output->dtype());
}

#else // JIT
#ifdef JIT_cpu
void CustomOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    for (index_t i = 0; i < num; i++)
        x[i] = (T)i;
}
#else // JIT_cuda
__global__ void kernel(index_t n, T* x) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = (T)-i;
}

void CustomOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    int blockSize = 256;
    int numBlocks = (num + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(num, x);
}
#endif // JIT_cpu
#endif // JIT
} // jittor
"""

my_op = jt.compile_custom_op(header, src, "custom", warp=False)
```

调用方式和普通算子一样。CPU 版产出 `0, 1, 2, …`，CUDA 版产出 `0, -1, -2, …`
——两份实现故意不同，用来确认当前真的跑在后端上。

```{code-cell} ipython3
:tags: [cuda, skip-execution]

jt.flags.use_cuda = 0
a = my_op([3, 4, 5], "float").fetch_sync()
assert (a.flatten() == range(3 * 4 * 5)).all()
print("CPU 版:", a.flatten()[:6].numpy(), "...")

if jt.compiler.has_cuda:
    jt.flags.use_cuda = 1
    b = my_op([3, 4, 5], "float").fetch_sync()
    assert (-b.flatten() == range(3 * 4 * 5)).all()
    print("CUDA 版:", b.flatten()[:6].numpy(), "...")
```

`warp=False` 表示不按 warp 对齐做额外处理；这个开关影响的是调度方式，不影响算子语义。

## 7. 编译与缓存

自定义算子的源码是**运行时编译**的，所以有几件事值得知道：

| 问题 | 答案 |
| --- | --- |
| 什么时候编译 | 第一次真正执行到它的时候，不是 import 的时候 |
| 编译多久 | 几秒到几十秒，取决于有没有 CUDA 头文件与模板 |
| 缓存在哪 | `JITTOR_HOME` 下的编译缓存目录，键是源码与编译选项的哈希 |
| 改了源码会怎样 | 哈希变了 → 重新编译。**不要指望改一行就立刻生效而不重编** |
| 并行编译 | 首次编译要串行，多个进程同时冷启动会互相等待 |

一个实用的习惯：**把算子源码放在字符串常量里**，而不是拼在调用点上。这样哈希稳定，
换机器、换进程都能命中同一份缓存。

## 8. 检查清单

```{code-block} text
1. 先问能不能用内置算子组合   —— 能的话白拿自动微分，别写内核
2. 反向必须自己写             —— jt.code 的梯度是 0，而且不报错，参数会永远不更新
3. 前向里存下反向要用的量      —— 在 execute 里存，在 grad 里用
4. grad 返回的是「输入梯度」顺序 —— 与 inputs 列表一一对应
5. CPU 侧用 in0_shapeN        —— in0_num 不存在，会编译失败
6. CUDA 侧必须查 cudaGetLastError —— 否则内核静默不跑，输出是没写过的内存
7. 参数和损失都要验证会动      —— 只检查「没有报错」等于没检查
8. 源码放字符串常量           —— 编译缓存按哈希命中，别在调用点拼字符串
```

第 2 条和第 7 条是一件事的两面：**自定义算子最危险的失败方式是「不学」而不是「崩」**。
所以写完之后一定要做第 4 节那个实验——参数动了、损失降了，才算真的接上了。

## 接下来

* [元算子](meta_op.md)：用三个元算子拼出卷积——Jittor 表达算子的另一种方式。
* [自动求导](autograd_details.md)：`jt.grad` 的完整语义与静默返回 0 的各种情况。
* [JIT、算子融合与异步执行](jit_and_async.md)：即时编译与融合到底做了什么。
* 回到 [教程总目录](README.md)。
