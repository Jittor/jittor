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

# 次正规数：CUDA 默认刷成零

这是一个**决定**，不是缺陷，但此前没有写在任何地方，于是每次设备对拍撞上它
都读起来像缺陷。

float32 的最小正规数是 `1.18e-38`。比它更小的非零值（次正规数）在 CPU 上按
IEEE-754 保留，在 CUDA 上**默认被刷成零**：

| 值 | CPU | CUDA 默认 | CUDA `strict` | NumPy |
| --- | --- | --- | --- | --- |
| `1e-45`（次正规） | 保留 | **0.0** | 保留 | 保留 |
| `1e-40`（次正规） | 保留 | **0.0** | 保留 | 保留 |
| `1e-30`（正规） | 保留 | 保留 | 保留 | 保留 |

根因是 nvcc 的 `--use_fast_math`，它蕴含 `-ftz=true`。后果不止于值本身：输入
在进入函数**之前**就已经是零了，所以 `log(1e-45)` 在 CPU 上给正确的
`-103.28`，在 CUDA 默认档下给 `-inf`；`count_nonzero([1e-45])` 在 CPU 上是 1，
在 CUDA 默认档下是 0。一个下溢到次正规区间的梯度在 CUDA 上恰好是零、在 CPU
上是一个极小的非零，两个设备于是对同一个模型走上不同的更新路径——在有人把
它们并排比较之前，这件事是不可见的。

## 怎么关掉

```python
jt.flags.cuda_kernel_math = "strict"     # 去掉 --use_fast_math
```

这个开关已经存在（`src/runtime/jit_policy.cc`），按进程生效，切换时会先把待
执行的图排空。实测它**完全恢复**次正规数：上表 `strict` 一列是量出来的，不是
推断的。

## 代价

2026-09-10 在 RTX 4090 上量过，两轮交替、取最小值：

| 算子（16M float32） | 默认 | strict |
| --- | --- | --- |
| `divide` | 218.6 µs | 218.8 µs |
| `sqrt` | 148.1 | 148.8 |
| `exp` | 150.0 | 150.2 |
| `log` | 148.1 | 148.2 |
| `mul-add`（对照，不受 fast-math 影响） | 218.6 | 218.6 |

**在这个规模下代价测不出来**：差异是 0.1–0.8%，和不受 fast-math 影响的对照
组同量级。原因是这些核受访存带宽限制，fast-math 省下的 ALU 周期本来就被访存
掩盖了。

计算密集的情形没有得出结论：把 `exp/log/sqrt` 串成 8 层和 32 层的链之后，同
一个档位在两轮之间的差异（1.01e-3 与 4.16e-4，2.4 倍）**大于**两个档位之间的
差异，那是融合方式的波动而不是策略的效果。所以这里只能说**没测出来**，不能说
"没有代价"。要给出一个可靠的数字，需要一个融合形态固定的基准。

## 契约

`tests/backends/parity/test_subnormal_contract.py` 把上面这张表断言下来。它
不是在说哪一种行为更对，而是在说**当前的行为是什么**：默认档刷成零、`strict`
保留、CPU 始终保留。三条里任何一条变了，那个测试就红——包括"有人把默认改成
了 strict"这种好的改变，那时应该改的是这份文档和那个测试，而不是让它们继续
描述一个已经不成立的世界。

# 卷积算法的自动选择会改变梯度

这一条和上面两条不同:它改变的不是某个算子的语义,而是**同一份代码在不同调度
下会训练出不同的权重**,而且**前向的 loss 一位不差**——任何盯着 loss 的检查都
不会响。

卷积算法不是按启发式选的,是**实测候选者再挑最快的那个**,结果按形状缓存
(`backends/cuda/kernels/cudnn/cudnn_conv_op.cc`;代码注释写明理由:启发式在寻常
fp32 3×3 上会挑到慢 2–3 倍的算法)。于是选中哪个算法取决于两件与数学无关的事:

* **实测时有什么驻留**——`auto_flush_ops` 改变执行调度,也就改变驻留;
* **在此之前见过多少个形状**——缓存满了会静默退回启发式
  (`if (cache.size() >= max_cache_size) benchmark = false;`),所以
  `set_algorithm_cache_size` 也是一个会改变数值的旋钮。

实测(RTX 4090,五个 bottleneck 块,598 万个参数元素):

| | `auto_flush_ops=0` | `=64` | `=256` |
| --- | --- | --- | --- |
| loss | 23080.078125 | 23080.078125 | 23080.078125 |
| 梯度范数(缓存 100) | 45839.379 | **45822.207** | 45839.379 |
| 梯度范数(缓存 100000) | 45839.379 | 45822.207 | **45822.207** |

相对差 3.8e-4,**确定性**(同一配置三次同一个数)——不是竞态,是真的在算不同的
东西。

## 怎么让它稳定

```python
import jittor.compile_extern as ce
ce.cudnn.set_benchmark(0)      # 强制走确定性的启发式
```

实测这样之后梯度不再随 `auto_flush_ops` 变化(残差 2e-6,是不同批边界带来的
重结合)。代价是卷积可能选到较慢的算法——上面那条注释说的正是这个,所以这是
**取舍而不是纯改进**,默认值没有动。

可对照的是 PyTorch 的 `torch.backends.cudnn.benchmark`:它有同样的性质,而且
在文档里写明了。这一节存在的理由就是把这里的也写下来。

## 契约

`tests/backends/cuda/test_autotuning_isolation.py` 钉住的是**边界**而不是行为:
关掉自动调优后,梯度**不得**随调度 flag 变化(这一条有牙齿——如果将来是**别的**
东西开始改变数值,它会红);以及无论开关,loss 都必须逐位相同。它**不**断言
"开着调优时梯度会不同"——那取决于机器、形状和 cuDNN 版本,在实测恰好稳定的硬件
上断言它会造成假红。
