# CUDA 环境合同

本文件只定义可复用合同，不绑定某一次 job、节点或个人主目录。每次任务把实际值
写进结果报告和运行键。

## Slurm

由用户指定一个 GPU job；登录节点只编排，所有 import、JIT、模型、测试和 benchmark
通过 worker 执行。开始时记录：

```bash
hostname
echo "$SLURM_JOB_ID $SLURM_STEP_ID $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
```

每个并发实现使用不同的 `JITTOR_HOME`、`TMPDIR` 和日志目录；首次 JIT 串行，单测
和 benchmark 不共享首次编译缓存。模型、tokenizer、数据和 Hugging Face 缓存离线
固定，原始日志/权重/缓存放 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/<run>/`。

## 两侧解释器

- **oracle**：独立 native PyTorch interpreter；断言没有 Jtorch shim marker。
- **candidate**：Jittor interpreter，先导入 Jittor 再导入 `torch`；断言 shim marker、
  `use_cuda=true` 和目标设备。
- 两侧核对 Python ABI、Torch/Transformers/PEFT/ms-swift 版本和实际 module origin。
  安装下游包不能覆盖 oracle 的 torch；必要时用独立 package site。

candidate 的 CUDA 运行设置 `JITTOR_TORCH_SHIM=1`、目标 `JITTOR_HOME` 和
`JITTOR_TORCH_PROJECT_ROOT`，并在整个 case 外层使用
`jt.runtime.scope(backend_fallback="error")` 与 `forbid_backend_fallbacks()`。

## 多卡

除了 Jittor 的 `JT_NCCL_RANK`、`JT_NCCL_LOCAL_RANK`、`JT_NCCL_WORLD_SIZE`，每个
child 必须获得 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`。若 child 只看到一张 GPU，
`LOCAL_RANK=0`；否则它必须对应可见设备索引。NCCL include/lib 使用已安装的 CUDA
wheel 路径，不允许首次运行尝试联网下载。记录每个 rank 的 worker 日志、NCCL
初始化、all-reduce/barrier、退出码和 fallback 计数。
