# CUDA 13.0 工具链 wheel 与 uv 验证

- Status: 已在 Linux x86-64 / RTX 4090（SM 89）上验证；范围限于工具链与下文
  明确列出的库组合。这不是完整的 CUDA 13 数学库栈认证。
- Baseline: `dd1013dbb422c70147efe8f1c8ab2720403a2c92`，在 `ad47b5814` 上整合
  `origin/2.0-refactor` 并保留 AMP/SDPA 提交之后。
- Implementation: 包含本报告的提交；父级整合在登记本报告时应记录其最终合并 SHA。
- Maintainer: Jittor 维护者。在编译器/runtime/CCCL 升级、Python marker 变化，或
  其他驱动、架构或库组合下重新检查。

## 复现与修复

官方 PyPI wheel 被下载、按其发布的 SHA-256 摘要校验，并安装到独立的状态目录。相关的
[nvcc 发布](https://pypi.org/project/nvidia-cuda-nvcc/13.0.88/)、
[CCCL wheel 发布](https://pypi.org/project/nvidia-cuda-cccl/13.0.85/) 与
[NVIDIA CCCL 安装指南](https://nvidia.github.io/cccl/unstable/python/setup.html)
是主要参考。CCCL 的 Python API 包与 Jittor 使用的 `nvidia-cuda-cccl` C++ 头文件
wheel 不同；没有新增 Python CCCL API 依赖。

实际的安装布局是 `nvidia/cu13`，CUB/Thrust/libcu++ 头文件位于 `include/cccl` 下。
旧的 provider 立即以“requires nvcc 12.2”拒绝 nvcc 13.0.88。针对已安装 CUB 头文件的
一次实际编译器调用失败，因为 CUB 需要 C++17。初始的框架构建也因
`cudaDeviceProp.computeMode` 与 `clockRate` 在 CUDA 13 中被移除而失败。

新的 `cuda13` extra 把编译器、runtime、CRT、NVVM 与 CCCL 一起固定。provider 校验
它们的共享前缀、可执行编译器、设备编译器/libdevice、runtime、设备 runtime 归档与
CCCL 头文件。已安装但残缺或不匹配的 CUDA 13 wheel 工具链会报错，而不是静默选择
另一套安装。在没有显式/系统编译器时，Jittor 会发现 wheel 的 nvcc。CUDA 13 构建在
host 与 device 上都使用 C++17；CUDA 12.2 保持其既有方言与 fingerprint。

另外两处实际的链接失败也已解决：该 wheel 提供 `lib/` 与 `libcudart.so.13`，而
nvcc 默认的共享 runtime 参数要求 `lib64/` 与未版本化的 `libcudart.so`。CUDA 13
的 wheel 编译现在显式提供库目录与精确的 runtime SONAME，同时保留默认的设备
runtime 归档。设备属性在两个 CUDA 版本上都使用受支持的 attribute 查询。库设置
区分仅工具链 provider 与完整的 CUDA 12.2 库栈。

实际的通用 uv 解析暴露了一个既有的 Python 3.9 不兼容：
`nvidia-cudnn-frontend>=1.30` 要求 Python 3.10。其 extra marker 现在与该要求一致，
重新生成的 lock 可成功解析。

## 实际平台与库

Python 3.11.16、g++ 11.5.0、glibc 2.34、NVIDIA 驱动 580.178.04 与 SM 89。
没有修改系统工具链或共享 Python 环境。独立的原始日志、wheel 归档、venv 与 JIT 缓存
均保留在 checkout 之外。

| 组件 | Wheel / 实际版本 | 观测到的库/头文件来源 |
| --- | --- | --- |
| nvcc, CRT, NVVM | 13.0.88 | 私有 `nvidia/cu13` 编译器、CRT、NVVM/libdevice |
| Runtime | 13.0.96; API reports 13000 | 私有 `nvidia/cu13/lib/libcudart.so.13` |
| CCCL | 13.0.85 | 私有 `nvidia/cu13/include/cccl` |
| cuBLAS / cuBLASLt | `nvidia-cublas==13.0.0.19`; API 13.0.0 | 真实私有 `libcublas.so.13` / `libcublasLt.so.13` |
| cuDNN | 既有 cu12 wheel；API 92400 | 外部 `nvidia/cudnn/lib/libcudnn.so.9` |
| cuRAND | 既有 cu12 wheel 10.3.3.141；API 10303 | 外部 `nvidia/curand/lib/libcurand.so.10` |
| cuFFT | 既有 cu12 wheel 11.0.8.103；API 11008 | 外部 `nvidia/cufft/lib/libcufft.so.11` |
| cuSPARSE | 既有 cu12 wheel 12.1.2.141；API 12.1.2 | 外部 `nvidia/cusparse/lib/libcusparse.so.12` |
| NVRTC / nvJitLink | 既有 cu12 wheel 12.2.140；NVRTC API 12.2 | 外部 `nvidia/cuda_nvrtc` / `nvidia/nvjitlink` 库 |

请求的 nvcc 13.0.88，观察到的 nvcc 为 `[13, 0, 88]`，runtime API 13000，
provider fingerprint 为 `pipcu130_9bded6158c61`。`/proc/self/maps` 确认了已加载
路径；私有链接在工具链 include/library 目录中暴露外部库。加载这些库并不构成对
FFT、稀疏、随机、NVRTC 或 nvJitLink 运算的数值验证。

初始的 cu12 cuBLAS + cuDNN 混合配置通过矩阵乘法，但在卷积上中止。
`LD_DEBUG=libs` 显示 cuDNN 试图加载缺失的 `libcublasLt.so.13`。安装真实的、
经哈希校验的 cuBLAS 13.0.0.19 wheel 后，被测卷积得到解决。没有使用指向 CUDA 12
二进制的伪造 `.so.13` 别名。其他混合组合仍未验证。

## 执行证据

- 真实的 CUDA 13 core 编译与逐元素输出 `[3, 3, 3]` 通过。
- 三项新的 CUB 测试零跳过通过：6007 元素 float32 scan、
  `(5, 1031)` float64 scan 与 `(5, 6, 7)` 非末轴 scan。每项都把
  前向值与反向 scan 梯度与独立 float64 NumPy 比较，
  检查设备驻留并禁止/统计后端 fallback。它们既不需要 CuPy，也不需要另一个 Jittor
  scan 作为参考。
- 独立真实 PyTorch `2.11.0+cu128`、二进制 `_C` 扩展、CUDA 12.8 与同一物理 GPU
  在 Jittor 导入/loader 路径未设置的独立进程中产生矩阵/卷积输出与两个输入梯度。
  矩阵 shape `(17, 23) @ (23, 13)` 的三个输出最大绝对误差为零。
  卷积 `(2, 3, 9, 11)` 配 `(4, 3, 3, 3)` 权重与 padding 一，
  输出误差为零，输入梯度误差 `2.98e-8`，权重梯度误差
  `8.94e-8`。所有 Jittor 输出/梯度都驻留在 CUDA，fallback 增量为零，TF32 已禁用。
- 既有真实 pip-cu122 栈保留 fingerprint `pipcu122_a89fa4fed08c`。
  CUB、cuBLAS 梯度与 cuDNN 卷积 plan 回归：**23 passed, zero skips**，290.52 s，包含过期缓存编译。
- uv 0.13.1 执行了通用 lock 解析、实际的 PEP 517 wheel 构建、
  `jittor[cuda13]` 的依赖解析/安装，并通过 `uv run` 执行。
  它从经校验的官方 PyPI wheel 产物安装了全部五个精确的 13.0 工具链包；
  其余少量 runtime 依赖来自 PyPI。
  在 checkout 之外的已安装 wheel 运行中，`PYTHONPATH`、
  `JITTOR_SOURCE_ROOT` 与显式 nvcc 未设置，Jittor 找到了 wheel 编译器，
  重新编译了完整核心，并在 CUDA 上执行 CUB 前向/反向，
  设备驻留成立、fallback 增量为零、runtime API 为 13000。
- provider 文件系统/负向回归：**22 passed**，包括缺失
  CCCL/libdevice、漂移到 NVVM 13.4、分离前缀、缺失 CRT、
  不可执行的编译器与损坏元数据的传播。
- 三项 CUB 回归中最终的操作数驻留/dtype 断言在 CUDA 13（2.74 s）与 pip-cu122（2.04 s）上都通过。
- CPU `core`，native 与 Torch 两种模式：**322 passed, 42 skipped, one
  existing xfail**。两个 CPU core 与 oneDNN 在冷的外部
  runner 缓存中构建；这些耗时不是暖缓存性能测量。
- 相关构建/provider/preflight 消费者：**25 passed, two skipped**。
- 仓库布局与空白检查通过。完整结构运行：
  **1434 passed, four skipped, three failed**，362.15 s。实际失败
  是冻结基线的过期 ACL `DropoutACL` 梯度 manifest 条目、
  AMP/SDPA/本报告等待其 results 索引条目，以及生成的
  `MANIFEST.in` 需要这些报告条目。父级整合负责这些
  共享登记与已修复的 ACL manifest；它们没有被从这一较旧分支覆盖。CUDA 13
  provider 没有引入额外的结构失败。

复现时，用 pip 或 uv 把 `jittor[cuda13]` 安装到外部环境，并在该工具链的
include/library 路径中提供数学库的头文件/库。该 extra 有意只安装编译器/runtime/
CCCL 工具链。使用分开的 CUDA 13/CUDA 12.2 测试缓存，选择实际设备并运行：

```bash
python -m pytest -q tests/backends/cuda/test_cub_cumsum.py::TestCubToolchain
python -m pytest -q tests/backends/cuda/test_cublas_matmul_grad.py \
    tests/backends/cuda/test_cudnn_conv_plan.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
    tests/structure/build/test_cuda_wheel.py
```

CUDA 13.4、Windows、ARM64、其他 GPU 架构/驱动、完整的 CUDA 13 厂商库、NCCL 与
所有 CUDA kernel 家族都不由这些运行认证。uv 运行使用已安装的官方 wheel 产物与
单独的已安装 Jittor wheel，而不只是修改过的 lockfile 或仅从源码导入。
