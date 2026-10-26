# TensorCircuit 报告的 Jittor 核心性能问题

- Status: 部分修复，CPU 与 RTX 4090 上实测；第二轮 CUDA 原生 QR/solve 已验证；余项见 KI-OPS-016
- Date: 2026-10-26
- Baseline: `refactor-2.0@1a2366b4a`（报告方所用基线）；本次改动未提交
- Owner: Jittor 核心维护者
- Review when: `Var.real`/`Var.imag`/`norm`、`backends/cpu/libraries/blas`、
  `linalg` 的 `numpy_code` 回调、`nn.tensordot` 或 CBLAS 行的选择规则发生变化

## 结论

TensorCircuit-NG 的 Jittor 后端在 MPS 模拟（χ = 64，complex64）中报告了 6 项根因在
核心的问题。处理如下：

| # | 问题 | 根因 | 处理 |
| --- | --- | --- | --- |
| 1 | `.real`/`.imag` 建图约 20 µs，复数 `norm` 约 47 µs | 每次都是 `view_as_real` 的 `jt.Function` 加 Python `getitem` | `real` 改为 complex→float 的 cast（C++ 已有反向 `dout + 0i`）；新增内部一元算子 `imag` 与 `as_imag`（x→x·i），互为伴随，反向在 C++；不生成公开的 `jt.imag`。复数 `norm` 改为 `Re(x·conj x)`，三个可融合的原生算子 |
| 1′ | （顺带）复数 `abs` 在 z = 0 处梯度为 NaN | 反向 `z/|z|` 是 0/0 | 在 0 处按 torch 取 0 |
| 2 | CUDA `solve` 每次两次主机同步 | CuPy 在 `errstate(linalg="raise")` 下对 `getrf`、`getrs` 各做一次 elementwise + 归约 + 阻塞读 | 二维 `solve`/`inv` 直接调 `getrf`/`getrs`，两个状态写进同一缓冲、读一次；QR 与 Cholesky 反向对三角因子的求解改为一次 `trsm`（CUDA 为 cuBLAS，CPU 为 numpy 自带 OpenBLAS 的 `cblas_?trsm`），只保留一次对角线检查，奇异输入仍然报错 |
| 3 | QR/SVD/solve 比 torch 慢 1.7–4 倍 | CPU：numpy 的 OpenBLAS 线程池与 Jittor 仍在自旋的 OpenMP 线程组争核，小矩阵多线程反而慢 3 倍；CUDA：CuPy `svd` 用 QR 迭代 `gesvd` | CPU 回调按工作量设 OpenBLAS 线程数（≤2^25 单线程，≤2^28 四线程，之上默认），调用后恢复；CUDA 二维约化 SVD 先用 Jacobi `gesvdj`，不收敛再回落 `gesvd`（与 torch 相同） |
| 4 | `nn.tensordot` 建图 35 µs，求导链比 torch 慢 1.6 倍 | 规划用 numpy bitmap 并逐维重读 `.shape`；CBLAS 行的选择谓词每次读两次环境变量 | 规划改为纯列表运算、恒等置换不再下发；CBLAS 行增加原生选择规则 `same_blas`；求导链的提升主要来自第 1 项 |
| 5 | 无 cmake 时 CPU float32 matmul 落到通用 kernel | oneDNN 构建不出来，CBLAS 行只收复数 | CBLAS 行加 `sgemm`/`dgemm`：float64 总是走它，float32 在 oneDNN 之后 |
| 6 | GPU 峰值显存偏高 | 报告方未缩小到只依赖 Jittor 的复现 | 未处理 |

## 验证

环境：cscg-qh04，作业分配 16 核 AMD EPYC 7763 + 1 张 RTX 4090（驱动 580.178.04，
CUDA 12.2），空闲节点（开始时 load 0.34）。Jittor 与 PyTorch 2.5.1 在同一 Slurm
作业里依次运行，各自独立进程；Jittor 使用独立的 `JITTOR_HOME`。复现脚本是报告方
`jittor_perf_repro.py` 的原样副本，与原始日志一起放在
`$JITTOR_LAB_ROOT/core-perf/`、`$JITTOR_LAB_ROOT/_state/slurm/logs/`（未版本化）。

µs/次（逐次同步；`build` 为只建图），“报告”为报告方在同型节点上测得的修复前数字：

| 用例（complex64） | CPU 现在 | CPU 报告 | torch CPU | GPU 现在 | GPU 报告 | torch GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `x.real` build | 1.5 | 19.5 | — | 1.5 | — | — |
| `x.real` sync | 9.3 | 39.7 | — | 16.7 | 42.7 | 7.6 |
| `jt.norm(x.reshape(-1))` build | 10.3 | 47.1 | — | 9.3 | — | — |
| `jt.norm(x.reshape(-1))` sync | 37.0 | 107.0 | — | 61.9 | 141.7 | 14.4 |
| `nn.tensordot` build | 26.3 | 35.1 | — | 32.0 | — | — |
| `nn.tensordot` sync | 97.3 | — | 30.1 | 68.1 | 82.4 | 34.1 |
| qr 128×64 | 534 | 1,427 | 358 | 570 | 567 | 337 |
| svd 128×128 | 4,276 | 7,094 | 3,321 | 3,280 | 9,431 | 3,090 |
| solve 64×64 | 254 | 415 | 97 | 313 | 463 | 150 |
| 200 步求导链（ms） | 41.4 | 79 | 49.6 | 37.1 | 68 | 39.7 |

CPU 实数矩阵乘法（ms）：

| | Jittor（有 oneDNN） | Jittor（`JT_BUILD_USE_MKL=0`） | torch |
| --- | --- | --- | --- |
| float32 2048³ | 14.2 | 14.9（报告：5,541） | 16.7 |
| float64 2048³ | 29.1 | 29.0 | 40.7 |

每个进程第一次走 OpenBLAS 的调用会多出约 15 ms（线程池首次创建后自旋，见
`blas_runtime.cc`），表现在 256³ 的第一行。

测试（`OMP_NUM_THREADS=16`，CUDA 用例在 RTX 4090 上）：

| 检查 | 结果 |
| --- | --- |
| `tools/run_test_suite.py --tier core` | native 98 passed，torch 224 passed |
| `tests/structure`（`JITTOR_TORCH_SHIM=1`） | 通过，除 `test_import_layering` 的环路面 165 > 164（改动前已存在） |
| `tests/linalg`、`tests/ops/test_linalg.py`（带 torch 参照） | 通过，除 `test_eigh`（与 autograd 的折叠约定不一致，改动前已存在） |
| 新增 `tests/linalg/test_lapack_paths.py`；扩充 `tests/type/test_complex64_native.py`、`tests/ops/test_complex_matmul_relays.py` | CPU 与 CUDA 通过 |

## 第二轮：CUDA 上 QR 与 solve 改为原生算子

torch 的 QR 与 solve 同样调用 cuSOLVER/cuBLAS（`torch.profiler` 看到的是
`geqr2_smem`/`larfb`/`orgqr` 与 `getrf_pivot` 加两次 `trsm_l_mul32`），差距在调用
外围。二维的 CUDA 输入因此不再经 `numpy_code` 回调和 CuPy：

| 新增 | 做法 |
| --- | --- |
| `cusolver` 库（`backends/cuda/libraries/cusolver`，按需加载） | 每设备一个 cuSOLVER 句柄，绑定 Jittor 的计算流；工作区取自 `temp_allocator` |
| `cusolver_solve`、`cusolver_inv` | 行优先的 a 对 cuSOLVER 即 aᵀ：拷贝一次后原样 `getrf`，`getrs` 取 `op = T`；`inv` 以单位阵为右端、`op = N`，两侧都不转置。`getrf` 与 `getrs` 的状态写进同一缓冲，读回一次 |
| `cusolver_qr` | 转成列优先后 `geqrf`、`orgqr`/`ungqr`，再拷回 q 与上三角 r；状态只会报非法参数，不读回，因此不同步（与 torch 相同） |
| `cublas_trsm`（cuBLAS 库） | QR 与 Cholesky 反向的三角求解：a x = b 即 xᵀ aᵀ = bᵀ，按右侧求解直接作用在 a、b 的原缓冲上，无拷贝；读回对角线一次，奇异时仍报错 |
| C++ 错误类别 `UserLinAlgError` | Python 侧的类同时继承 `JittorError` 与 `numpy.linalg.LinAlgError`，奇异输入的异常类型与回调路径一致 |

这几个算子在执行时读回主机，录制中的 CUDA graph 遇到它们会被标记为含主机工作，
`graph_capture_end` 拒绝该录制（QR 不读回，可以录制）。批量输入、以及 NVIDIA wheel 形式的
CUDA（不含 cuSOLVER）仍走 CuPy。

同一节点、同一作业（µs/次，逐次同步）：

| 用例（complex64） | 上一轮 | 现在 | torch |
| --- | --- | --- | --- |
| solve 64×64 | 313 | 210 | 147 |
| qr 128×64 | 570 | 396 | 370 |
| qr 128×64 前向 + 反向 | 2,010 | 1,530 | 737 |

QR 反向仍是 torch 的 2 倍：为了高阶导数，反向里重新计算一次 QR（`_reconnect`），并由十余个
小算子组成。

## 已知边界与后续

- 未达报告的验收线：CPU `solve` 64×64 为 torch 的 2.6 倍；CUDA 上 QR 的前向加反向
  为 2.1 倍；复数 `norm` 在 GPU 上逐次同步仍为 4 倍（4 个 kernel）；CPU 上
  n ≥ 128 的 QR 受限于 OpenBLAS 的 LAPACK 本身。记为 KI-OPS-016。
- CUDA 上的 `solve`、`inv`、三角求解与 SVD 每次仍有一次主机同步（为了让奇异输入
  报错、或判断 `gesvdj` 是否收敛），与 torch 相同；没有做延迟到下一个同步点的检查。
- 发布用的 wheel 内容基线（`tools/release/baselines/`）需在下次出包时重新生成。
- 同步一个 `transpose` 会真正拷贝一次（torch 的 `permute` 是视图，不拷贝），所以
  这一列不可直接对比；CPU 上该拷贝 kernel 约 3 GB/s，未优化。
- 批量（rank > 2）的 `solve`/`svd`/三角求解仍走 CuPy/NumPy 的原路径。
- 第 6 项（GPU 峰值显存）需要报告方提供只依赖 Jittor 的复现。
- ROCm/NPU 未验证。
