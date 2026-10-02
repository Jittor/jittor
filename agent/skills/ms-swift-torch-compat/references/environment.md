# ms-swift CUDA 环境合同

## 调度与设备

所有 import 检查、模型运行、测试、JIT 和 benchmark 都必须进入用户指定的 Slurm 作业；不得在登录节点计算。当前任务使用 Slurm job `977`、节点 `cscg-qh13`、NVIDIA RTX 4090。每个运行日志都记录 `SLURM_JOB_ID`、`SLURM_STEP_ID`、hostname、CUDA 型号/UUID、可见设备和驱动/CUDA 版本。

```bash
srun --jobid=977 --overlap --mpi=none -N1 -n1 -c4 \
  bash --noprofile --norc -c 'hostname; echo "$SLURM_JOB_ID $SLURM_STEP_ID"; nvidia-smi'
```

每个 CUDA 进程只暴露本次需要的 GPU。CUDA 计算必须证实 Jittor `use_cuda=true`、Torch `device=cuda`、模型参数/输入/输出/梯度在目标 CUDA 设备；shim 用 `jt.runtime.scope(backend_fallback="error")` 和 `forbid_backend_fallbacks()`，观测整个阶段并要求计数为零。

## 双运行时

- oracle：独立原生 PyTorch CUDA 解释器；断言 `not hasattr(torch, "_torch_compat_install_context")`。
- candidate：同 ABI 的 Jittor/Torch shim 环境，导入顺序为先 `import jittor` 再 `import torch`，并断言 shim 标记存在。
- 两侧 ms-swift、Transformers、PEFT、tokenizers 等下游版本及实际模块路径必须核对一致。安装前检查依赖 pin，绝不让下游安装覆盖 oracle 的 torch。
- 上游 checkout 要记录 remote URL 和 commit SHA；模型、tokenizer、数据固定并离线运行（`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`）。

当前任务环境由用户 state 脚本准备。登录节点编排命令，作业 worker 内执行：

```bash
source /home/xinshen/_state/cuda-ms-swift/env.sh
export LD_PRELOAD=/usr/lib64/libpython3.9.so.1.0
export CPATH=/home/xinshen/_state/cuda-ms-swift/python39include
export PATH=/home/xinshen/projects/ms-swift-cuda/venv/bin:/home/xinshen/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin:$PATH
export JTCUDA_HOME=/home/xinshen/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux
export JT_BUILD_PYTHON_CONFIG_PATH=/home/xinshen/_state/cuda-ms-swift/python3.9-config
export JITTOR_HOME=/home/xinshen/_state/cuda-ms-swift/jittor-home-977b
export JITTOR_TORCH_SHIM=1
export JITTOR_TORCH_PROJECT_ROOT=/home/xinshen/projects/ms-swift-cuda/jittor
export PYTHONPATH=/home/xinshen/_state/cuda-ms-swift:/home/xinshen/projects/ms-swift-cuda/jittor/python:/home/xinshen/projects/ms-swift-cuda/ms-swift
```

每个彼此并行/不同实现版本使用独立 `JITTOR_HOME`，首次 JIT 编译串行进行。benchmark 不与首次编译或并发 unittest 共用 cache。原始产物统一放 `/home/xinshen/_state/cuda-ms-swift/` 下，不放进源码树。

## 运行键与证据

运行键至少包含同步 SHA、dirty diff SHA256、测试/runner SHA256、两个解释器及 ABI、PyTorch/Jittor/ms-swift/Transformers/PEFT 版本、CUDA/驱动/GPU、精度、case、权重和输入摘要、随机种子、优化器与步数、fallback 策略。报告每条命令、退出码、日志路径和状态。相同运行键且证据齐全可复用；实现、输入或依赖变更时重跑受影响范围。
