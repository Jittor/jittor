# Native Triton and TileLang kernels

[Core concepts](../index.en.md) · [中文](external-kernels.md)

`jittor.kernels` schedules externally compiled kernels in the ordinary Jittor computation graph. The executor owns input dependencies, output allocation, device streams, and storage lifetime. It requires neither the Torch shim nor conversion to PyTorch tensors. Triton and TileLang are kernel implementation mechanisms; CUDA, ROCm, and NPU are device backends.

## Installation and scope

With CUDA already configured for Jittor, install `pip install 'jittor[triton]'` or `pip install 'jittor[tilelang]'`. These extras pin Triton 3.9.0 and TileLang 0.1.15 respectively. TileLang itself depends on PyTorch, but the execution path below uses Jittor allocations and bypasses TileLang's Torch call wrapper.

The optional compiler extras require Linux and Python 3.10 or newer. This is an experimental interface. The compiler frontends currently accept CUDA only. The CPU callback interface is available for other implementations; ROCm and NPU never silently fall back to CPU. Hardware tests live in `tests/backends/cuda/test_native_kernel_compilers.py`; executor semantics are tested in `tests/runtime/test_external_kernel.py`. Current validation results belong in the PR.

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

Upstream JIT specialization and persistent compiler caching, constexpr arguments, compiler options, scalar/tuple/callable grids, zero-input kernels, multiple outputs, `@triton.autotune`, and global/profile scratch workspaces (allocated from executor temporary storage) are supported. Launches execute natively in C++ via the CUDA driver `cuLaunchKernel` on Jittor's compute stream without requiring PyTorch, enabling `async_execution` and CUDA Graph capture/replay (`jt.graph_replay` and `jt.capture_step`, for both inference and `jt.Function` training). Noncontiguous inputs retain their actual strides: the kernel must explicitly accept and use them. A linear-addressing kernel does not automatically support arbitrary layouts. Triton interpreter mode, `launch_cooperative_grid`, `launch_pdl`, TMA tensor descriptors, and autotuner `pre_hook`/`post_hook` are not yet supported.

Use `jittor.kernels.BoundOutput(buffer)` in a Triton output slot to update an existing CUDA allocation during inference. Bound outputs keep their addresses, and the executor orders the writes after existing readers. The call accepts fresh `TensorSpec` outputs alongside bound outputs. This path prepares the launch before CUDA recording and requires `jt.no_grad()`, contiguous CUDA inputs on the current device, ready nonempty bound buffers, and positive grid dimensions. Output buffers cannot overlap each other. If a kernel also reads a buffer it writes, it must avoid races within the launch.

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

`out_idx=[]` selects a function accepting caller-provided output buffers. Native TVM-FFI executables run directly in C++ via the TVM-FFI C ABI (stack-allocated `DLTensor` views and `TVMFFIEnvSetStream`) on Jittor's compute stream without Python callbacks, enabling `async_execution`, CUDA Graph capture/replay, zero-input kernels, and `BoundOutput` in inference mode (`jt.no_grad()`, contiguous CUDA inputs/outputs, reader-ordered writes); non-C-ABI executables fall back to the Python callback path with lazy DLPack export. Output shapes and dtypes must match the compiled ABI. Callee-allocated return values and other TileLang execution backends are not integrated yet.

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

The caller must keep inputs read-only, write every declared output, and avoid out-of-bounds accesses. These are native device-kernel memory safety obligations. All tensors must share a device. The ordinary `TensorSpec` path does not support in-place input mutation or output aliasing (use `BoundOutput` explicitly for inference-time in-place updates), and implicit higher-order derivatives are not supported. Supporting a compiler does not imply supporting all these features.

Custom providers can use `call(launch, *inputs, outputs=..., backend=...)` (including zero-input kernels). The callback receives `context.inputs`, `context.outputs`, `context.device`, and `context.stream`. Buffer views expose pointers, shape, dtype, strides, and lazy DLPack export, and callback failures preserve the Python exception type and message. Enqueue work on the supplied stream only; do not reenter Jittor execution, switch consumer streams, or retain callback views. The executor preserves dependencies so inputs are computed before consumption and storage is not reused prematurely; generic Python callbacks via `call(...)` do not support device graph capture/replay.
