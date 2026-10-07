# 平台支持

本页回答一个问题：**某个操作系统、Python 版本或加速后端，是在哪里被验证过的。**
每一行都给出证据位置——CI 工作流、`docs/results/` 里的报告，或者「未验证」。

**未验证不等于不能用**，它等于没有人能替你保证。对一个没有门禁的组合，先自己跑
`python -m jittor.selftest` 和相关测试，再把它当成可用。

安装步骤不在这里，见
[README 的安装章节](https://github.com/Jittor/jittor#install)。

## 操作系统

| 平台 | 验证位置 |
| --- | --- |
| Linux x86_64 | 门禁。`.github/workflows/cpu.yml` 在 `ubuntu:24.04` 容器里跑 `nox -s smoke` 与 `nox -s full`；`structure.yml` 跑结构、lint、typing、打包与多版本 Python |
| Linux aarch64 | 只随昇腾门禁验证（`.github/workflows/npu.yml` 的自托管 ARM64 机器）。**没有独立的 aarch64 CPU 作业** |
| macOS（Intel） | 发布时验证。`release.yml` 的 `platform-validation` 在 `macos-15-intel` 上装规范 wheel 并跑 `python -m jittor.selftest`；**不跑测试树** |
| Windows x86_64 | 发布时验证，同上，runner 为 `windows-2022` |
| macOS（Apple Silicon） | 未验证。没有 runner，也没有报告 |

平台矩阵的唯一定义在 `.github/ci-baseline.env`（`CI_RELEASE_PLATFORM_MATRIX`），
各工作流消费它而不是各自重写。Jittor 发的是 `py3-none-any` 纯 Python wheel 加 JIT
源码资源，所以这三个平台安装的是**同一个产物**；发布流程还专门断言不会产出平台
wheel。

## Python 版本

`pyproject.toml` 声明 `requires-python = ">=3.7"`，classifier 覆盖 3.7–3.13。

| 版本 | 验证位置 |
| --- | --- |
| 3.11 | 门禁。所有维护的工具与测试会话都钉在 3.11（`CI_PYTHON_VERSION`），CPU/CUDA/NPU/结构/文档门禁都在它上面跑 |
| 3.7 | 门禁。`nox -s py37` 用真实 3.7 解释器编译全树（下界检查，不是全量测试） |
| 3.12 / 3.13 | 门禁。`nox -s py312`、`nox -s py313` 构建并跑已安装 wheel 的自检；3.13 这一条在 NumPy 2.x 下跑 |
| 3.8 / 3.9 / 3.10 | 声明支持，**没有专门的 CI 作业**。3.9 有一次真实使用记录：昇腾验证报告的机器是 Python 3.9.25（[Ascend 910B3 验证](../results/2026-08-28-ascend-910b-validation.md)） |

运行 `transformers` 需要 Python 3.10 以上（它用 `types.UnionType`），这是上游的要求，
不是 Jittor 的，见 [Torch 兼容](../compatibility/torch.md)。

## 后端

| 后端 | 版本范围 | 验证位置 |
| --- | --- | --- |
| CPU | g++ ≥ 5.4（Linux）、clang ≥ 8 + `libomp`（macOS），需要 OpenMP | 门禁，`cpu.yml` |
| CPU oneDNN relay | 只有 float32 | 随 CPU 门禁；dtype 范围与单次调用开销是开口项 `KI-BACKEND-011` |
| CUDA | 构建需要 `nvcc`；CUDA < 11 会额外取 cub，CUDA 11 优先选 cuDNN 8。`jittor[cuda12]` extra 钉 CUDA 12.2.2 运行时与 cuDNN ≥ 8.9.7 < 10（Linux x86_64） | 门禁，`cuda.yml`：自托管 RTX 4090（sm_89）+ CUDA 12.2（`CI_CUDA_VERSION`）。CUDA 11 与 13 的 wheel 家族**未提供** |
| 昇腾 NPU（ACL/HCCL），单卡 | CANN 9（门禁断言 `atc --version` 为 9.x），设备须报 Ascend 910B | 门禁，`npu.yml`（Linux ARM64、`/dev/davinci0`、fail-closed 的设备探针）。一次完整的冷缓存验证记在 [Ascend 910B3 验证](../results/2026-08-28-ascend-910b-validation.md)：CANN 9.0.0、驱动 25.5.1、`397 passed, 9 skipped` |
| 昇腾 NPU（ACL），Ascend 950PR / x86-64 | CANN 9.1 | **部分验证，无门禁**。`e5886b6f6`（2026-09-10）在真机上以 `backend_fallback=error` 零回退跑通 matmul、卷积前向反向、随机分布、CNN 与 transformer block 训练；同一提交上 `tests/backends/acl/test_acl.py` 为 32 passed / 11 failed。`docs/guides/ascend-910b.md` 里的一处同步开关测量也来自这台机器 |
| 昇腾 NPU，多卡 HCCL | — | **未验证**：需要 ≥ 2 张 910B3，见 [`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md) 的 `8.02` |
| ROCm | — | **未验证**。`nox -s rocm` 存在但没有 CI 工作流、也没有设备；HIP provider 的 `runtime/driver.cc` 至今没被编译器读过（`deferred-hardware.md` 的 `4.12`）。MIOpen 与 rccl 是**有意未实现**，会抛 `NotImplementedError` |
| 天数智芯 Corex | — | **未验证**。只有只读探测契约（`tests/backends/corex/test_corex_discovery.py`，离线假编译器），没有 nox session，没有设备。上机步骤见 [天数 Corex](corex.md)、`deferred-hardware.md` 的 `8.14` |
| MPI / NCCL 单机多卡 | — | `nox -s mpi` 与 `nox -s cuda` 的分布式用例，**无专门 CI 工作流**；CUDA 门禁的机器是单卡 |
| 跨机（≥2 节点） | — | **未验证**：rendezvous、启动器、checkpoint 与掉线门禁都在等两台机器（`deferred-hardware.md` 的 `8.15`–`8.18`、`10.22`） |

后端的 `grad()` 覆盖也不齐：60 条后端梯度里 36 条要硬件才能测，7 条还没有测试、
1 条未实现（开口项 `KI-BACKEND-013`）。

## Torch 兼容层针对的 PyTorch 版本

| 项 | 值 | 验证位置 |
| --- | --- | --- |
| 声明的 Torch API 级别 | `2.11.0`（`torch.__torch_version__`、`torch.version.__version__`） | `compat/torch/installers/cuda/bindings.py`；`torch.__version__` 仍报告 Jittor 自己的版本 |
| nightly 生态对拍的参照 torch | CPU 版 `2.7.1`，配 `transformers 4.46.3`、`diffusers 0.31.0`、`peft 0.13.2` | 门禁，`.github/workflows/ecosystem.yml`（`nox -s ecosystem`，七个用例跑两遍，比输出与每个梯度） |
| 真实模型差距表的参照 torch | `2.11.0+cu128`（cuDNN 9.19），Transformers 4.56.2、Diffusers 0.35.1 | [Torch 兼容层在真实模型上对 PyTorch](../results/2026-09-24-torch-compat-real-models.md) |
| 最近几次 SDPA / 端到端测量的参照 torch | `2.13.0+cu129` | [下游库性能台账](../performance/library-standing.md) |
| 适配器支持的库版本 | Transformers 4.56.2 / 5.5.3；TorchMetrics 1.7.4 | `adapters/pyproject.toml` 与 [`adapters/README.md`](https://github.com/Jittor/jittor/blob/master/adapters/README.md)；未识别的版本会抛 `UnsupportedAdapterVersion` |

**编译过的 PyTorch 扩展一概不支持**：`mmcv.ops`、`torch_npu` 这类自带针对 PyTorch
C++ ABI 编译的 `.so` 的包，Python 层的兼容层装不进来。这是架构边界，不是待办事项，
理由见 [Torch 兼容](../compatibility/torch.md)。

## 容器镜像

| 镜像 | 基础镜像 | 验证位置 |
| --- | --- | --- |
| `jittor` | `ubuntu:24.04` | 门禁，`containers.yml`：构建并冒烟启动默认的 notebook 服务 |
| `jittor-cuda` | `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04` | 同上 |

两个镜像共用根目录的同一份 `Dockerfile`，矩阵定义在 `.github/ci-baseline.env`
的 `CI_CONTAINER_MATRIX`。

用户可见的限制与各自编号见[已知限制](known-limitations.md)。
