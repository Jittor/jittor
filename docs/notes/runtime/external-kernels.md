# Triton 与 TileLang 原生 kernel

[核心概念](../index.md) · [English](external-kernels.en.md)

`jittor.kernels` 将外部编译器生成的 kernel 放进 Jittor 的普通计算图。输入依赖、输出分配、设备流和内存寿命由执行器管理。无需开启 Torch shim，也无需把 Jittor 张量转换成 PyTorch 张量。Triton 和 TileLang 是算子实现方式，CUDA、ROCm、NPU 则是设备后端，两者不是同一层。

## 安装与边界

在已经配置好 CUDA 的 Jittor 环境中，按需安装 `pip install 'jittor[triton]'` 或 `pip install 'jittor[tilelang]'`。这两个 extra 分别固定 Triton 3.9.0 和 TileLang 0.1.15；TileLang 编译器本身依赖 PyTorch，但以下执行路径由 Jittor 分配输入、输出，不经过 TileLang 的 Torch 调用包装。

当前接口是实验性的，编译器入口只接受 CUDA。CPU 回调接口可用于其他外部实现；ROCm 和 NPU 不会偷偷回退到 CPU。硬件测试位于 `tests/backends/cuda/test_native_kernel_compilers.py`，通用执行语义测试位于 `tests/runtime/test_external_kernel.py`；最新验收结果记录在 PR 中。

## Triton

保留上游 `@triton.jit` 函数。用 `TensorSpec` 占据输出指针的位置，其余参数沿用 kernel 的签名：

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

支持上游 JIT 特化与磁盘编译缓存、constexpr、编译选项、可调用 grid，以及多个输出。非连续输入保留真实 stride；kernel 必须显式接收并正确使用 stride，线性寻址的 kernel 不会自动变成支持任意布局的 kernel。暂不接受 autotuner、解释器和需要全局 scratch 的 kernel。

## TileLang

保留上游 `T.prim_func`，使用 TVM-FFI 执行后端，并显式传递所有输出：

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

`out_idx=[]` 表示编译后的函数接收调用者提供的输出缓冲区。Jittor 用带所有权的 DLPack 视图传递这些缓冲区，TVM-FFI 使用 Jittor 当前流执行。输出形状和 dtype 必须与编译后的 ABI 相符；TileLang 自行分配返回值的 ABI、其他执行后端尚未接入。

## 反向与执行契约

外部 kernel 不会自动获得数学导数。用 `jt.Function` 定义反向，反向可以使用普通 Jittor 算子，也可以再次调用上述接口。例如加法：

```python
class Add(jt.Function):
    def execute(self, x, y):
        return triton_call(add, x, y, TensorSpec(x.shape, x.dtype),
                           N=x.numel(), BLOCK=128,
                           grid=((x.numel() + 127) // 128,))

    def grad(self, dout):
        return dout, dout
```

调用端负责保证输入只读、所有声明的输出都已写入，不能越界；这是原生设备 kernel 的内存安全契约。所有张量位于同一设备。暂不支持输入原地修改、输出别名、设备图捕获/回放和隐式高阶梯度；这些限制不能用“支持 Triton/TileLang”掩盖。

自定义提供方可使用 `call(launch, *inputs, outputs=..., backend=...)`。回调接收 `context.inputs`、`context.outputs`、`context.device` 与 `context.stream`；缓冲区提供指针、shape、dtype、stride 和 DLPack。只允许在指定流上提交工作，不得调用 Jittor 执行器、切换消费流或保留回调视图。执行器保留依赖关系，避免提前读取尚未计算的输入或复用仍在使用的存储。
