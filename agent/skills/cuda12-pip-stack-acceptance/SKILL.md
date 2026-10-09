---
name: cuda12-pip-stack-acceptance
description: 在干净的 uv 或 conda 环境里安装 jittor[cuda12]，让一套真实的系统 CUDA 作为干扰项可见（/usr/local/cuda、/usr/bin/nvcc、发行版库目录、PATH、LD_LIBRARY_PATH、CUDA_HOME），在真 GPU 上跑加法、matmul 前反向和 cuDNN 卷积，并证明 nvcc、libcudart、libcudnn 都来自该环境的 site-packages/nvidia。改 cuda12 extra、编译器发现、CUDA 库查找或预加载顺序后用它验收。
---

# jittor[cuda12] 的干净环境验收

## 为什么需要干扰项

机器上没有系统 CUDA 时，「用的是 pip 的 CUDA」是唯一可能的结果，验收不说明任何事。
真实缺陷只在系统 CUDA 可见时出现：`jittor_core` 按名字依赖 `libcudart.so.12`，而动态
加载器先查 `LD_LIBRARY_PATH` 再查 core 的 `RUNPATH`，于是系统的 cudart 先被绑定，
pip 的那份随后作为第二个运行时载入。只看 `jt.flags.nvcc_path` 或日志都发现不了。

## 干扰项怎么造（不需要 root）

`decoy_namespace.sh` 在 `unshare --user --map-root-user --mount` 的私有挂载命名空间里，
用 overlay 把 `$DECOY_CUDA` 挂成 `/usr/local/cuda`，链接 `/usr/bin/nvcc`，把它的
cudart/cudnn/cublas/nvrtc 链进 `/usr/lib/x86_64-linux-gnu`，并把它放到 `PATH`、
`LD_LIBRARY_PATH`、`CUDA_HOME`、`CUDA_PATH` 最前面。宿主机不受影响，GPU 在命名空间里
照常可见。干扰工具链的 `nvcc`、`ptxas` 换成记录调用后再执行原程序的包装，所以
「用了但碰巧成功」也会被记下。

`$DECOY_CUDA` 必须是能用的完整工具链，版本最好和 pip 栈不同，泄漏时一眼可辨。可以用
NVIDIA redist 包（`developer.download.nvidia.com/compute/cuda/redist/`）拼一个：
`cuda_nvcc`、`cuda_cudart`、`cuda_cccl`、`cuda_nvrtc`、`libcublas` 和
`cudnn/redist` 的 cuDNN，解到同一个根目录，`lib/` 改名为 `lib64/`。放在
`$JITTOR_LAB_ROOT/cuda12-pip-stack/` 下。

## 用法

```bash
export JITTOR_LAB_ROOT=<lab-root> DECOY_CUDA=<decoy-toolkit-root>
export CUDA_VISIBLE_DEVICES=<idle-gpu>
bash agent/skills/cuda12-pip-stack-acceptance/run_acceptance.sh uv <run>
bash agent/skills/cuda12-pip-stack-acceptance/run_acceptance.sh conda <run>
```

- `uv`：`uv sync --locked --no-default-groups --extra cuda12`，以 editable 方式装本 checkout。
- `conda`：conda-forge 的 Python，再 `pip install "<checkout>[cuda12]"`（构建真实 wheel）。
- 每次运行用新的 `<run>`；状态、`JITTOR_HOME` 与日志在
  `$JITTOR_LAB_ROOT/_state/cuda12-pip-stack/<run>/`，两条路径串行跑，首次 JIT 不并发。
- `PYTHON_VERSION` 默认 3.11。

## 判据（任一不满足即失败）

`gpu_smoke.py` 与 `run_acceptance.sh` 检查：

1. 环境里没有 `nvcc_path`、`JT_BUILD_NVCC_PATH`，`PATH`/`LD_LIBRARY_PATH`/`CUDA_HOME`
   都没有指向 pip 组件；`CUDA_VISIBLE_DEVICES` 显式给出。
2. 加法、matmul 及其两个梯度、conv2d 的结果用 `cuPointerGetAttribute` 证明在 GPU 内存里，
   数值与 numpy 一致。
3. conv2d 捕获到 `cudnn_conv precision select` 日志，即真的走了 cuDNN。
4. `jt.flags.nvcc_path` 的真实路径在 `site-packages/nvidia/` 下，pip 栈已激活。
5. `/proc/self/maps` 里每个 CUDA 库都在 `site-packages/nvidia/` 下，cudart 和 cuDNN 都已映射，
   且没有编译器 wheel 带来的 `libcudart.so.13`。
6. 干扰工具链的 `nvcc`/`ptxas` 调用记录为空。
7. `LD_DEBUG=files` 覆盖本次运行的全部进程（含编译子进程），没有任何进程初始化来自
   干扰路径的 CUDA 库。

输出末尾的 `CUDA12_PIP_SMOKE=` JSON 记录路径与组件版本，结果报告引用它即可。
