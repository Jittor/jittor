---
name: cuda-backend-choice-proof
description: 证明一个 CUDA 后端算子（cuBLAS/cuDNN/cuFFT/cuTT/cuSPARSE）真的选到了预期的算法、计算精度或缓存键，以及证明随机性（dropout/curand）每次调用都在推进。当一个修复改的是"选了哪条路"而不是"算出什么值"、因而数值断言看不出差别时用它。
---

# 证明 CUDA 后端算子选对了路

后端库的很多 bug 改的是**选择**（algo、computeType、mathType、缓存键、dropout 状态），
不是**数值**。这类改动用 `assert_allclose` 验证是无效的：

- CUDA ≥ 11 起 `CUBLAS_GEMM_DEFAULT_TENSOR_OP` 只是提示，选错了结果照样对；
- 算法缓存键少一个 dtype，命中错条目时 cuDNN 往往还是能算出正确结果，只是慢或偶发；
- dropout 掩码每步相同，loss 曲线看起来完全正常。

**判据：跑绿了不算通过。必须把"选了什么"读出来断言。**

## 手法：让算子把选择打进日志，用 log_capture_scope 读回来

JIT 算子里的 `LOGvvv` 可以被 Python 捕获，前缀匹配的是**生成出来的 jit 源文件名**，
它以算子名开头（`cublas_matmul__T_float32__Trans_a_N__...__op.cc`），所以
`log_vprefix="cublas_matmul=100"` 能命中，且不会误命中 `cublas_batched_matmul`
（前缀哈希从第一个字符逐字符走，`cublas_b...` 与 `cublas_m...` 第 8 个字符就分叉）。

算子侧（放在真正调库之前，让日志反映最终值）：

```cpp
LOGvvv << "cublas_matmul algo select:"
    << "use_tensorcore=" >> use_tensorcore
    << "computeType=" >> cublas_compute_type_name(computeType)
    << "algo=" >> cublas_gemm_algo_name(algo);
```

**打名字不要打枚举数字。** 枚举值（`CUBLAS_GEMM_DEFAULT` = -1、`_TENSOR_OP` = 99、
`CUBLAS_COMPUTE_32F` = 68…）写进测试没人看得懂，换个 CUDA 版本还会变。在对应的
`*_wrapper.h` 里加一个 `static inline const char* xxx_name(enum)` 给两边共用。

测试侧：

```python
from _helpers.logs import find_log_with_re

with jt.log_capture_scope(log_silent=1, log_v=0,
                          log_vprefix="cublas_matmul=100") as raw_log:
    jt.matmul(a, b).sync()          # 必须 .sync()，否则惰性图还没跑
found = find_log_with_re(raw_log, r"algo select: use_tensorcore=(\S+) computeType=(\S+) algo=(\S+)")
assert found, "no selection log captured"
tc, compute, algo = found[-1]       # 取最后一条：一次 sync 可能跑了不止一个 gemm
```

要点：

- `log_v=0` + `log_vprefix="<前缀>=100"`：只放行这一个文件的 LOGvvv，别的算子不吵。
- `log_silent=1`：日志不打到终端，只进 buffer。
- 取 `found[-1]`，不要取 `[0]`：同一次 sync 里 cast/random 也可能触发别的 gemm。
- 改 flag（`jt.flags.use_tensorcore` 等）不会改 JIT key，不会重编，运行期读全局变量即可生效。
  所以一个测试里循环 4 种取值是安全的。

同样的手法适用于**缓存键**：把 `jk.to_string()` 打出来，断言 fp32 与 fp16 的键不同
（`cudnn_conv3d` 的 fwd/bwdx/bwdw 三个 legacy 缓存就是靠这个证明不再互相串）。

## 手法：证明随机状态每次调用都在推进

「dropout 掩码每步不同」「随机数不重复」这类，不要去读掩码——读不到。用**同输入两次前向**：

```python
jt.set_seed(0)
y1 = layer(x)                 # 训练模式、dropout > 0
y2 = layer(x)                 # 同一个 x，同一个 layer
assert not np.allclose(y1.numpy(), y2.numpy())
```

陷阱：

- **cuDNN RNN 的 dropout 只作用在层与层之间**，`num_layers=1` 时 dropout 完全没有效果，
  测试会假绿（两次一样但不是因为 bug）。**必须 `num_layers >= 2`**。
- 先确认"关掉随机性时两次一致"，再确认"打开时两次不一致"。只测后者的话，
  一个恒返回垃圾的实现也能通过。

## 确定性的失败触发器（证明「失败会抛」而不是「失败被吞掉」）

改 `XXX_CALL` 宏从 fprintf 改成抛，必须有一个**确定性**的失败输入，否则无法证明修前修后
的差别。已知可用的：

| 库 | 触发方式 | 得到的错误 |
| --- | --- | --- |
| cuFFT | `jt.nn._fft2(jt.zeros((1, 0, 4, 2), "float32"))`——任一变换维长度为 0 | `cufftPlanMany` 返回 `CUFFT_INVALID_SIZE` |

判据不止「抛了」。**修前那一版会把无效句柄写进缓存**，所以还要断言：

1. 同一形状连续失败两次（缓存里没有留下"能用"的坏句柄）；
2. 失败之后正常形状的变换仍然正确（缓存没被污染）。

修前跑这个测试，退出时 `peekCudaErrors(cufftDestroy(...))` 会打出
`code=1( CUFFT_INVALID_PLAN )`——那就是坏句柄进了缓存的直接证据。

`jt.nn._fft2` 是 cuFFT 算子唯一的入口（`jt.fft.*` 走 DFT 矩阵乘，不碰 cuFFT），
且要求 `jt.flags.use_cuda == 1`、输入形状 `(batch, n1, n2, 2)`。

## 后端库的 `_cudaGetErrorEnum` 重载约定

`checkCudaErrors(x)` 需要一个 `_cudaGetErrorEnum(该状态类型)` 重载。
`extern/cuda/src/helper_cuda.cc` 里那些重载被 `#ifdef _CUFFT_H_` / `#ifdef CUSPARSEAPI`
之类包着，而该文件并不 include 这些库的头，所以**它们一个都没被编进 libcuda_extern**。
每个后端靠自己目录下的 `*/src/helper_<lib>.cc` 提供重载（cublas/cudnn/curand/cusparse 都有）。

症状：`ImportError: .../gen_ops_xxx.so: undefined symbol: _Z17_cudaGetErrorEnum13cufftResult_t`，
接着被 `compile_extern.py` 翻译成误导性的 `CUDA found but cufft is not loaded`。
修法：照抄 `curand/src/helper_curand.cc` 的写法，在该后端的 `src/` 下补一个。

## 环境（少设一个测的就是别的东西）

```bash
JITTOR_HOME=<自己的> TMPDIR=<自己的> CUDA_VISIBLE_DEVICES=<自己的卡> \
nvcc_path=/usr/local/cuda/bin/nvcc taskset -c <自己的核段> \
python -m pytest tests/backends/cuda/<...> -q
```

- **`nvcc_path` 少了就跑成 CPU 版**，cuBLAS/cuDNN 算子根本不会被实例化，测试静默跳过或走别的路。
- **`JITTOR_HOME` 与别人共用会损坏缓存**，症状是毫不相干的算子大面积报错。

### 已知坑：cuda key 被日志污染导致 `FileNotFoundError`

首次在一个新的 `JITTOR_HOME` 里跑时可能崩在：

```
FileNotFoundError: .../cu12.2.140_..._sm_0902_211521.482820_88_89_Create_[i_file..._jittor.lock_lock_lock.py85]
```

原因：`compiler.py` 用 `sp.getstatusoutput(... -m jittor_utils.query_cuda_cc)` 取 SM 版本，
`getstatusoutput` 把 stderr 也合进来了，而该子进程首次运行会往 stderr 打一行
`Create lock file:...`，于是这行被拼进了 cache 目录名。

修法（一次性）：

```bash
find $JITTOR_HOME/.cache/jittor -maxdepth 9 -name "*sm_[0-9][0-9][0-9][0-9]_*" -exec rm -rf {} +
JITTOR_HOME=<自己的> python -m jittor_utils.query_cuda_cc   # 预热，让 lock 文件先建出来
```

之后 `query_cuda_cc` 只输出 `89` 这样的纯数字，key 就正常了。

## 重编代价

`extern/cuda/**` 下的算子按**所有源码的哈希**编成一个 `custom_ops` .so：改一个算子会
把 cublas/cudnn/cufft/cutt/cusparse 全部重编。单次约 30~60 秒（核心 .so 不动的话），
但改了 `python/jittor/src/**` 就是十分钟级全量重编。**攒着一次验证，不要改一行跑一次。**

## 「修前失败」怎么低成本演示

选择类改动往往只有一两行。先把日志与测试加好，再把那一两行**临时改回旧写法**跑一次
（只重编 custom_ops，几十秒），确认测试红；然后改回来跑绿。比 `git stash` 整个改动便宜，
因为不会碰到核心 .so。
