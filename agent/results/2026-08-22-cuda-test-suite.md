# 完整 CUDA 后端、dtype、设备对拍与 OpInfo 门禁

- Status: Accepted for CUDA; NPU/ROCm unavailable
- Last reviewed: 2026-08-22
- Baseline: `866914d4`
- Owner: Jittor core and CUDA backend maintainers
- Review when: CUDA/CuPy/toolkit versions, CUDA gate contents, OpInfo registry,
  process-mode ownership, or known accelerator reductions change

## 结论

维护的 `nox -s cuda` 门禁已在真实 NVIDIA GeForce RTX 4090 上从空 Jittor、
CUDA 和 CuPy 缓存完整通过。门禁不再只运行基础 CUDA smoke，而是依次覆盖：

1. 完整 `tests/backends/cuda`；
2. 整数宽度与低精度 dtype；
3. 227 项 CPU/CUDA 前向和反向对拍；
4. Torch TF32 控制；
5. 227 项独立 OpInfo CUDA reference。

Jittor 真实加载 SM89 CUDA core 并执行张量探针。导入成功或 CPU fallback 未计为
设备支持证据。

## 环境

- Git baseline: `866914d4`
- Python: 3.11.15（预置 jt311 环境）
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- Driver: 595.84
- CUDA toolkit: 12.2.140
- CuPy: 13.6.0
- NumPy: 1.26.4
- pytest: 7.4.4
- Isolation: `$JITTOR_LAB_ROOT/_state/nox/envs/cuda/tmp/`, `cache_name=nox_cuda`
- JIT policy: `use_parallel_op_compiler=0`; the maintained session ran alone

测试使用分配到的空闲 RTX 4090，未触碰已有进程占用的其他设备。环境中没有
`npu-smi` 和 `rocminfo`，因此本报告不对 NPU 或 ROCm 作出结论。

## Verify-then-fix findings

### CUB extension

扩展 CUDA 后端目录后，首先复现
`tests/backends/cuda/test_cublas_test_op.py::TestCubTestOp::test` 失败：vendored
示例包含仅存在于上游测试树的 `test/test_util.h`。替换为确定性 host reference 和
直接 `cub::DeviceRadixSort` 后，又复现 `CubTestOp::jit_run` 通过 CUDA 输出指针
直接写入 host 值导致的段错误。

最终实现使用 CUDA runtime API 分配、复制并检查 radix-sort 缓冲区，并通过
`cudaMemcpyHostToDevice` 写入结果。定向结果：

- CUB target: `1 passed in 20.10s`；
- cuBLAS/cuDNN/CUB module: `3 passed in 7.31s`；
- 最终冷缓存 CUDA 后端目录包含同一 CUB 测试并通过。

### Device-parity process and failure isolation

共享 OpInfo registry 使用 Torch-facing signature，因此 native parity 最初在
`keepdim` 参数上失败。`test_device_parity.py` 现在归属 Torch-mode 进程，dtype
coverage 仍在独立 native pytest 进程运行。

窄整数 reduction 编译失败不能在聚合 parity 进程内保持 expected failure：Jittor
异步执行器会在下一个测试再次抛出失败 op。聚合 parity 因此只跳过这些精确路径，
并指向 `tests/ops/test_ops.py` 中的 strict CUDA xfail。最终门禁保留四项 strict
xfail，并证明紧随其后的普通操作仍会执行。

冷缓存运行还暴露了仅在先前 Jittor op 之后发生的 CuPy CUB C++17 determinant
编译失败。旧 linalg probe 使用较小矩阵，只覆盖 CuPy 的 C++11 路径。probe 现在
会在执行 Jittor graph 前物化与测试相同的 `(2,3,3)` 和 `(2,2,3,3)` batched
determinant。全新缓存的 `det -> diag -> diagonal -> digamma` 回归通过
`4 passed in 194.14s`，最终完整 parity 也运行了全部维护的 linalg 操作，而非跳过。

## Final maintained gate

```bash
CUDA_VISIBLE_DEVICES=<allocated-device> \
nvcc_path="$(command -v nvcc)" \
JITTOR_CI_PYTHON="$(command -v python)" \
use_parallel_op_compiler=0 \
python -m nox -s cuda
```

Cold-cache results:

| Stage | Result | Time |
| --- | --- | ---: |
| CUDA backend directory | 93 passed, 1 skipped, 1 xfailed | 23:51 |
| dtype coverage | 6 passed | 7:33 |
| CPU/CUDA device parity | 221 passed, 6 skipped, 12 warnings | 2:39:10 |
| Torch CUDA TF32 | 2 passed | 3.06s |
| strict CUDA OpInfo | 221 passed, 2 skipped, 4 xfailed | 9:50 |

Nox reported `Session cuda was successful in 3 hours`.

六项 parity skip 是 active known-issues ledger 中列出的四项窄整数 reduction 和两项
logical bool reduction。strict OpInfo 进程保留四项整数 strict xfail；实现修复会产生
XPASS 并要求清理 ledger。parity warning 是 FFT 结果转换到 float64 metric 时已有的
`ComplexWarning`。

## Supporting gates

```text
bash agent/scripts/check_repo_layout.sh
  repository layout OK

python -m pytest -q tests/structure
  218 passed in 76.70s
```

额外的 warm-cache native CUDA backend 加 dtype 运行通过
`99 passed, 1 skipped, 1 xfailed in 500.50s`；完整 warm-cache parity 在最终 cold
gate 前通过 `221 passed, 6 skipped, 12 warnings in 2617.53s`。

## Boundaries

- CUDA 仅对上述维护门禁范围 accepted，不代表每个可选依赖或下游项目均已安装。
- NPU 和 ROCm 保持未验证环境结果，而非框架通过。
- 独立二进制 PyTorch oracle 已在后续
  [parallel-range 与网络验证](2026-08-22-cuda-parallel-range-network-oracle.md)
  中通过；可选下游依赖仍是单独 todo 项。
- JIT 缓存和临时编译诊断位于 `$JITTOR_LAB_ROOT/_state/`，未向仓库添加 binary
  或 raw log。
