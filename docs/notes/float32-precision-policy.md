# float32 累加精度

矩阵乘和卷积可以用比 float32 更低的精度做累加换取速度。这一页说明 Jittor 提供哪些
档位、默认是什么、以及在惰性图下什么时候生效。

## 档位

同一套档位分别作用于 cuBLAS 的 compute type 和 cuDNN 的 math type，两边各有一个
开关（见下文）：

| 档位 | cuBLAS compute type | cuDNN math type | 含义 |
| --- | --- | --- | --- |
| `highest` | `CUBLAS_COMPUTE_32F` | `CUDNN_FMA_MATH` | 真正的 float32 累加 |
| `high` | `CUBLAS_COMPUTE_32F_FAST_TF32` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | tf32 |
| `medium` | `CUBLAS_COMPUTE_32F_FAST_16BF` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` | bfloat16 |

**档位只作用于 float32 操作数。** float16 与 bfloat16 一律用 float32 累加
（`CUBLAS_COMPUTE_32F` / `CUDNN_DATA_FLOAT`），float64 用 float64。这与 torch
的规则一致——低精度输入本来就需要高精度累加，"更快"的设置不应该在这里额外花掉
精度。

cuBLAS 的算法提示跟随 compute type：请求了低精度 compute type 时才用 tensor-op，
两者不再分别选择。

## 怎么设置

原生接口也是**分开的两个开关**：`float32_matmul_precision` 只管 matmul，
`float32_cudnn_precision` 只管 cuDNN（卷积与 RNN）。

```python
jt.flags.float32_matmul_precision = "high"     # 只影响 matmul
jt.flags.float32_cudnn_precision = "highest"   # 只影响 cuDNN，回到严格 float32
```

Torch 前端同样是分开的两个开关，与 torch 语义一致：

```python
torch.set_float32_matmul_precision("high")   # 只影响 matmul
torch.backends.cudnn.allow_tf32 = True       # 只影响 cuDNN（卷积与 RNN）
torch.backends.cuda.matmul.allow_tf32 = True # matmul 的另一种拼写，同一个开关
```

两个前端的默认值相同，都是 torch 的默认：matmul 为 `highest`，cuDNN 为 `high`
（即 `torch.backends.cudnn.allow_tf32 = True`）。Ampere 及以后的卡上 float32 卷积
因此默认走 tf32：每次乘法的输入只保留 10 位尾数，累加仍是 float32，卷积输出的相对
误差约在 1e-4–1e-3。需要严格 float32 卷积的代码（对拍、梯度检查）应显式设
`float32_cudnn_precision = "highest"`。两个前端的 setter 互不影响对方的状态。

原生前端的 cuDNN 默认值曾经是 `highest`，并且由 `float32_matmul_precision` 一并
设置。那比 torch 和 Jittor 1.3.10 都严格——1.3.10 不设置 math type，用的是 cuDNN 的
默认值，在 Ampere 及以后允许 tf32——代价是 ResNet-50 训练（batch 64）每步 128 ms，
1.3.10 为 95 ms，差距全部在卷积反向。

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

## 这一页之外的数值契约

次正规数在 CUDA 上被刷成零、卷积算法的自动选择会改变梯度、IEEE 特殊值算术、
`maximum`/`minimum` 的 NaN 语义、CPU 归约的精度——这些原本散落在这一页尾部的条目，
现在收在[数值契约](numerics-contract.md)一页里，每条带实测证据和钉住它的测试文件。
这一页只讲 float32 的**累加精度档位**。
