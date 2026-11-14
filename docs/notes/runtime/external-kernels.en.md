# Native Triton and TileLang kernels

[Core concepts](../index.en.md) · [中文](external-kernels.md)

`jittor.kernels` schedules externally compiled kernels in the ordinary Jittor computation graph. The executor owns input dependencies, output allocation, device streams, and storage lifetime. It requires neither the Torch shim nor conversion to PyTorch tensors. Triton and TileLang are kernel implementation mechanisms; CUDA, ROCm, and NPU are device backends.

## Installation and scope

With CUDA already configured for Jittor, install `pip install 'jittor[triton]'` or `pip install 'jittor[tilelang]'`. These extras pin Triton 3.9.0 and TileLang 0.1.15 respectively. TileLang itself depends on PyTorch, but the execution path below uses Jittor allocations and bypasses TileLang's Torch call wrapper.

This is an experimental interface. The compiler frontends currently accept CUDA only. The CPU callback interface is available for other implementations; ROCm and NPU never silently fall back to CPU. Hardware tests live in `tests/backends/cuda/test_native_kernel_compilers.py`; executor semantics are tested in `tests/runtime/test_external_kernel.py`. Current validation results belong in the PR.

## Triton

Keep the upstream `@triton.jit` function. Place a `TensorSpec` in each output-pointer argument slot; other arguments follow the kernel's signature:

```python
import jittor as jt
import triton
import triton.language as tl
from jittor.kernels import TensorSpec, triton_call

@triton.jit
def add(X, Y, O, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, 0) + tl.load(Y + i, i < N, 0)
    tl.store(O + i, value, i < N)

with jt.flag_scope(use_cuda=1):
    x, y = jt.ones((257,)), jt.ones((257,))
    z = triton_call(add, x, y, TensorSpec(x.shape, x.dtype),
                    N=257, BLOCK=128, grid=(3,))
    print((z * 2).numpy())
```

Upstream JIT specialization and persistent compiler caching, constexpr arguments, compiler options, callable grids, and multiple outputs are supported. Noncontiguous inputs retain their actual strides: the kernel must explicitly accept and use them. A linear-addressing kernel does not automatically support arbitrary layouts. Autotuners, interpreters, and kernels requiring global scratch are not yet accepted.

## TileLang

Keep the upstream `T.prim_func`, select the TVM-FFI execution backend, and pass all outputs explicitly:

```python
import jittor as jt
import tilelang
import tilelang.language as T
from jittor.kernels import TensorSpec, tilelang_call

@T.prim_func
def add_tile(A: T.Tensor((256,), "float32"),
             B: T.Tensor((256,), "float32"),
             C: T.Tensor((256,), "float32")):
    with T.Kernel(2, threads=128) as block:
        for i in T.Parallel(128):
            index = block * 128 + i
            C[index] = A[index] + B[index]

kernel = tilelang.compile(add_tile, out_idx=[], target="cuda",
                          execution_backend="tvm_ffi")
with jt.flag_scope(use_cuda=1):
    x, y = jt.ones((256,)), jt.ones((256,))
    z = tilelang_call(kernel, x, y, TensorSpec((256,)))
    print(z.numpy())
```

`out_idx=[]` selects a function accepting caller-provided output buffers. Jittor supplies owning DLPack views; TVM-FFI executes on Jittor's current stream. Output shapes and dtypes must match the compiled ABI. Callee-allocated return values and other TileLang execution backends are not integrated yet.

## Backward and execution contract

External kernels do not acquire mathematical derivatives automatically. Define backward with `jt.Function`, using ordinary Jittor operations or another external kernel call. For example, addition:

```python
class Add(jt.Function):
    def execute(self, x, y):
        return triton_call(add, x, y, TensorSpec(x.shape, x.dtype),
                           N=x.numel(), BLOCK=128,
                           grid=((x.numel() + 127) // 128,))

    def grad(self, dout):
        return dout, dout
```

The caller must keep inputs read-only, write every declared output, and avoid out-of-bounds accesses. These are native device-kernel memory safety obligations. All tensors must share a device. Input mutation, output aliases, device graph capture/replay, and implicit higher-order derivatives are not supported. Supporting a compiler does not imply supporting all these features.

Custom providers can use `call(launch, *inputs, outputs=..., backend=...)`. The callback receives `context.inputs`, `context.outputs`, `context.device`, and `context.stream`. Buffer views expose pointers, shape, dtype, strides, and DLPack. Enqueue work on the supplied stream only; do not reenter Jittor execution, switch consumer streams, or retain callback views. The executor preserves dependencies so inputs are computed before consumption and storage is not reused prematurely.
