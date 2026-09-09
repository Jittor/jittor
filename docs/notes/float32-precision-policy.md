# float32 累加精度

矩阵乘和卷积可以用比 float32 更低的精度做累加换取速度。这一页说明 Jittor 提供哪些
档位、默认是什么、以及在惰性图下什么时候生效。

## 档位

一个统一的档位同时决定 cuBLAS 的 compute type 和 cuDNN 的 math type：

| 档位 | cuBLAS compute type | cuDNN math type | 含义 |
| --- | --- | --- | --- |
| `highest`（默认） | `CUBLAS_COMPUTE_32F` | `CUDNN_FMA_MATH` | 真正的 float32 累加 |
| `high` | `CUBLAS_COMPUTE_32F_FAST_TF32` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | tf32 |
| `medium` | `CUBLAS_COMPUTE_32F_FAST_16BF` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | bfloat16 |

**档位只作用于 float32 操作数。** float16 与 bfloat16 一律用 float32 累加
（`CUBLAS_COMPUTE_32F` / `CUDNN_DATA_FLOAT`），float64 用 float64。这与 torch
的规则一致——低精度输入本来就需要高精度累加，"更快"的设置不应该在这里额外花掉
精度。

cuBLAS 的算法提示跟随 compute type：请求了低精度 compute type 时才用 tensor-op，
两者不再分别选择。

## 怎么设置

原生接口：Runtime 的 `float32_matmul_precision` 同时写 matmul 与 cuDNN 两层。

Torch 前端是**分开的两个开关**，与 torch 语义一致：

```python
torch.set_float32_matmul_precision("high")   # 只影响 matmul
torch.backends.cudnn.allow_tf32 = True       # 只影响 cuDNN（卷积与 RNN）
torch.backends.cuda.matmul.allow_tf32 = True # matmul 的另一种拼写，同一个开关
```

默认值不同：**torch 前端** matmul 为 `highest`、cuDNN 为 `high`；**原生前端**两者
都是 `highest`。两个前端的 setter 互不影响对方的状态。

同一个语义的多种拼写（`allow_tf32`、`fp32_precision`、
`get/set_float32_matmul_precision`）都是同一个标志的**视图**，不各自存一份状态，
因此不会出现三种拼写给出三个答案的情况。

## 惰性图下什么时候生效

Jittor 的执行是惰性的，所以"什么时候读取这个设置"必须明确：

- **每个算子在构造时**捕获当时的档位对（matmul 档 + cuDNN 档）；
- 融合要求两个算子的档位对**相等**；
- 图重写、并行编译、执行和反向构造都会恢复捕获时的档位对。

也就是说，**构造完惰性图之后再改开关，不会偷偷改变那张图已经请求的库精度**。
这些作用域不修改全局 flag，也不会同步尚未完成的张量。

## 实现位置

| 内容 | 位置 |
| --- | --- |
| 策略与作用域 | `src/runtime/float32_precision.{h,cc}` |
| cuBLAS 选择（三个 gemm 算子共用） | `backends/cuda/libraries/cublas/include/cublas_compute_type.h` |
| cuDNN 选择（六个卷积算子共用） | `backends/cuda/libraries/cudnn/include/cudnn_wrapper.h` |
