---
name: cuda12-pip-stack-acceptance
description: 在干净的 uv 或 conda 环境里安装 jittor[cuda12]，让一套真实的系统 CUDA 保持可见（/usr/local/cuda、/usr/bin/nvcc、发行版库目录、PATH、LD_LIBRARY_PATH、CUDA_HOME），按场景验证编译器选择：合适的本机 nvcc 被使用，不合适或缺失时回退到 pip nvcc；并在真 GPU 上跑加法、matmul 前反向和 cuDNN 卷积，证明 libcudart、libcudnn 始终来自该环境的 site-packages/nvidia。改 cuda12 extra、编译器发现、CUDA 库查找或预加载顺序后用它验收。
---

# jittor[cuda12] 的干净环境验收

## 要证明什么

装了 `jittor[cuda12]` 后：编译器先用本机合适的 nvcc（12.2–13.x，且能配合 Jittor 的
C++ 编译器编出测试 kernel），没有就用 pip 的 nvcc 13；CUDA 运行库无论如何都来自 pip。

机器上没有系统 CUDA 时，这两条都无从证伪。真实缺陷只在系统 CUDA 可见时出现：
`jittor_core` 按名字依赖 `libcudart.so.12`，加载器先查 `LD_LIBRARY_PATH` 再查 core 的
`RUNPATH`，曾经让系统 cudart 抢先绑定；版本号合适的本机 nvcc 也可能根本编不过
（CUDA 12.4 拒绝 gcc 14，CUDA 12.x 与 glibc 2.41 的数学头文件冲突）。

## 系统 CUDA 怎么造（不需要 root）

`decoy_namespace.sh` 在 `unshare --user --map-root-user --mount` 的私有挂载命名空间里，
用 overlay 把 `$DECOY_CUDA` 挂成 `/usr/local/cuda`，把它的 cudart/cudnn/cublas/nvrtc
链进 `/usr/lib/x86_64-linux-gnu`，并设好 `LD_LIBRARY_PATH`、`CUDA_HOME`、`CUDA_PATH`。
按 `DECOY_MODE` 决定编译器：`system` 时 nvcc 排在 `PATH` 最前并链到 `/usr/bin/nvcc`；
`old-nvcc` 时 `--version` 谎报 11.8；`no-nvcc` 时 `/usr/local/cuda/bin` 为空。宿主机不受
影响，GPU 照常可见。系统 `nvcc`、`ptxas` 换成记录调用后再执行原程序的包装。

`$DECOY_CUDA` 用 NVIDIA redist 包（`developer.download.nvidia.com/compute/cuda/redist/`）
拼：`cuda_nvcc`、`cuda_cudart`、`cuda_cccl`（CUDA 13 另加 `cuda_crt`、`libnvvm`），加
`libcublas`、`cuda_nvrtc` 和 `cudnn/redist` 的 cuDNN，解到同一根目录，`lib/` 改名为
`lib64/`。放在 `$JITTOR_LAB_ROOT/cuda12-pip-stack/` 下。`system` 场景需要一套能配合
本机 C++ 编译器的工具链，`unusable` 场景需要一套版本合适但编不过的。

## 用法

```bash
export JITTOR_LAB_ROOT=<lab-root> CUDA_VISIBLE_DEVICES=<idle-gpu>
run=agent/skills/cuda12-pip-stack-acceptance/run_acceptance.sh
DECOY_CUDA=<usable-toolkit>   bash $run uv    <run> system
DECOY_CUDA=<unusable-toolkit> bash $run uv    <run> unusable
DECOY_CUDA=<any-toolkit>      bash $run uv    <run> old-nvcc
DECOY_CUDA=<any-toolkit>      bash $run uv    <run> no-nvcc
DECOY_CUDA=<usable-toolkit>   bash $run conda <run> system
```

- `uv`：`uv sync --locked --no-default-groups --extra cuda12`，以 editable 方式装本 checkout。
- `conda`：conda-forge 的 Python，再 `pip install "<checkout>[cuda12]"`（构建真实 wheel）。
- 每次运行用新的 `<run>`；状态、`JITTOR_HOME` 与日志在
  `$JITTOR_LAB_ROOT/_state/cuda12-pip-stack/<run>/`。多次运行串行，首次 JIT 不并发。
- `PYTHON_VERSION` 默认 3.11。运行时间较长，放后台执行并轮询输出文件。

## 判据（任一不满足即失败）

1. 环境里没有 `nvcc_path`、`JT_BUILD_NVCC_PATH`，`PATH`/`LD_LIBRARY_PATH`/`CUDA_HOME`
   都没有指向 pip 组件；`CUDA_VISIBLE_DEVICES` 显式给出。
2. 加法、matmul 及其两个梯度、conv2d 的结果用 `cuPointerGetAttribute` 证明在 GPU 内存里，
   数值与 numpy 一致；conv2d 捕获到 `cudnn_conv precision select`，即真的走了 cuDNN。
3. `jt.flags.nvcc_path`：`system` 场景必须是系统 nvcc，且调用记录里有真实编译；
   `unusable` 场景必须是 pip nvcc，系统 nvcc 只跑过版本探测和 `check.cu` 试编译；
   `old-nvcc` 只跑过 `--version`；`no-nvcc` 一次都没跑。
4. `/proc/self/maps` 里每个 CUDA 库都在 `site-packages/nvidia/` 下，cudart 和 cuDNN 都已
   映射，且没有编译器 wheel 带来的 `libcudart.so.13`。
5. `LD_DEBUG=files` 覆盖本次运行的全部进程（含编译子进程），没有任何进程从系统路径
   初始化 CUDA 运行库（系统 nvcc 本身从系统工具链运行是允许的）。

输出末尾的 `CUDA12_PIP_SMOKE=` JSON 记录路径与组件版本，结果报告引用它即可。
