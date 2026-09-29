# CUDA 验收合同

## 设备与运行键

所有 CUDA import、模型运行、测试、JIT 和 benchmark 都通过用户指定的 Slurm job 和
worker 执行。每条日志记录 job/step、节点、GPU 型号/UUID、驱动/CUDA、可见设备、
Jittor/PyTorch/ms-swift 版本、Python ABI、commit、模型/数据摘要、seed、精度、
optimizer/步数、`JITTOR_HOME` 和 fallback 策略。远端路径使用
`$JITTOR_LAB_ROOT/_state/<topic>/<run>/`，不把缓存或原始日志写进主仓库。

oracle 必须断言 `import torch` 是原生 PyTorch，candidate 必须断言 shim 标记存在；
两侧依赖版本和源码路径要相同。candidate 的整个阶段使用
`backend_fallback=error` 与 `forbid_backend_fallbacks()`，并要求
`fallback_count == 0`。多卡除 `JT_NCCL_*` 外还要导出 `RANK`、`WORLD_SIZE`、
`LOCAL_RANK`；每 rank 的 `LOCAL_RANK` 必须与它看到的单张 GPU 一致。

## L0-L5

| 层 | 证据 |
| --- | --- |
| L0 | native/shim import、模型/tokenizer/template/dataset/tuner/optimizer/trainer/engine 构造，版本、配置、状态键、设备和 dtype 一致 |
| L1 | 同权重同输入 CUDA 前向；容器、shape、dtype、finite、logits/hidden/loss 和 greedy token 一致 |
| L2 | 全 trainable 参数梯度、适用 input gradient、loss、optimizer state/update 和至少三步固定数据轨迹；不能只比较最终 loss |
| L3 | 同进程和全新进程 checkpoint/resume；模型、buffers、adapter、optimizer/scheduler、RNG、数据游标、global step、下一批和继续轨迹一致 |
| L4 | 真实 `swift` CLI、Python API、launcher、训练/推理或 server/client 入口，保留 worker、checkpoint、输出和子进程环境 |
| L5 | L0-L4 适用项通过后，真实目标尺寸预热后至少 10 次同步稳态；报告 native/Jtorch ratio、吞吐、显存口径和零 fallback |

前一层失败时后续层记为 `blocked`。缺失依赖、联网模型、NPU/TP/多机专属入口或
不可提供媒体 fixture 记 `blocked`/`not-applicable`，不能记 `pass`。

## 数值判定

CUDA 默认生态 forward `5e-3`、backward `2e-2`，必须在运行前固定。每个 case 还保存
最大绝对误差、使用全局比较 floor 的 scaled max、相对 L2、dtype、finite 检查和首个
分歧层。BF16 RMSNorm、attention、量化和 logits 不得看到结果后放宽门槛；若只在
BF16 失败，要保留 FP32、非 fused 和原生算子控制实验。
