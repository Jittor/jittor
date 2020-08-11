# Custom Op: write your operator with C++ and CUDA and JIT compile it

# 自定义算子：使用C ++和CUDA编写您的算子，并其进行即时编译

> NOTE: This tutorial is still working in progress

In this tutorial, we will show:

1. how to write your operator with C++ and CUDA and JIT compile it
2. execute your custom operation

If you want to implement a very simple op with few lines of code, please use code op, please see `help(jt.code)`.
custom_op is used for implement a complicated op. The capabilities of custom_op and built-in operations are exactly the same.

> 注意：本教程仍在持续更新中

在本教程中，我们将展示：

1. 如何用C ++和CUDA编写您的算子并对其进行即时编译
2. 运行您的自定义算子

如果您想用几行代码来实现一个非常简单的算子，请使用code运算，请参阅`help(jt.code)`.
custom_op用于实现复杂的算子。 custom_op和内置运算的功能完全相同。

```python
import jittor as jt

header ="""
#pragma once
#include "op.h"

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
#include "var.h"
#include "custom_op.h"

namespace jittor {
#ifndef JIT
CustomOp::CustomOp(NanoVector shape, NanoString dtype) {
    flags.set(NodeFlags::_cuda, 1);
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(shape, dtype);
}

void CustomOp::jit_prepare() {
    add_jit_define("T", output->dtype());
}

#else // JIT
#ifdef JIT_cpu
void CustomOp::jit_run() {
    index_t num = output->num;
    auto* __restrict__ x = output->ptr<T>();
    for (index_t i=0; i<num; i++)
        x[i] = (T)i;
}
#else
// JIT_cuda
__global__ void kernel(index_t n, T *x) {
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

Let's check the result of this op.

让我们查看一下这个运算的结果。

```python
# run cpu version
jt.flags.use_cuda = 0
a = my_op([3,4,5], 'float').fetch_sync()
assert (a.flatten() == range(3*4*5)).all()

if jt.compiler.has_cuda:
    # run cuda version
    jt.flags.use_cuda = 1
    a = my_op([3,4,5], 'float').fetch_sync()
    assert (-a.flatten() == range(3*4*5)).all()
```