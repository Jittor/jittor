---
name: ms-swift-cuda-torch-compat
description: 在真实 NVIDIA CUDA 上验证或修复 ms-swift 对 Jittor torch shim 的兼容性，按适配面矩阵和 L0-L5 记录可复现证据。
---

# ms-swift × Jittor CUDA 兼容

## 目标与范围

本 skill 面向 ms-swift 通过公开 torch API 使用 Jittor torch shim 的 CUDA 兼容性。首个
`ms_swift_lora_llama` 只是定位和回归入口，不能代表整个 ms-swift。开始时必须从当前
ms-swift checkout 的 `swift/`、`tests/`、`examples/`、requirements 和 CI 生成
surface manifest，按真实安装版本、可选依赖和用户声明确定范围。未执行的面必须写
`not-run`，资源或依赖问题写 `resource-blocked`，不能改写成通过。

本任务只把真实 NVIDIA CUDA 作为候选设备；CPU 只作为独立 PyTorch oracle。Ascend/NPU、
ROCm 和其他后端不继承本 Skill 的结论。

开始前阅读：

- [环境合同](references/environment.md)
- [验收合同](references/verification.md)
- [适配归属](../downstream-library-adaptation/SKILL.md)
- [transformers-torch-compat](../transformers-torch-compat/SKILL.md)
- [github-collaboration-commit](../github-collaboration-commit/SKILL.md)

## 覆盖面矩阵

至少盘点并逐项记录状态、首个断点、运行键和证据路径：

| 适配面 | 代表内容 |
| --- | --- |
| 入口与配置 | swift CLI/launcher、配置解析、runtime、设备、dtype、seed、保存/加载 |
| 模型 | 实际暴露的 causal LM、seq2seq/encoder、分类、embedding、reranker、reward、MoE、视觉/音频/多模态 |
| tuner | Swift LoRA、PEFT LoRA、QLoRA、prompt/prefix/adapters、冻结和 trainable 参数 |
| 训练 | SFT/pretrain/分类/embedding/reranker、gradient accumulation、checkpointing、AMP、AdamW/Muon 等 |
| 偏好与 RL | DPO/CPO/KTO/GKD/PPO/GRPO 及当前版本实际存在的其他流程；缺依赖单列 blocked |
| 分布式 | Jtorch launcher、单卡、单机双卡 NCCL、资源具备时的多机；FSDP/DeepSpeed/Ray 单列状态 |
| 推理与服务 | swift infer、Python pipeline、Transformers engine、batch/stream、server/client/deploy |
| 数据和序列化 | datasets/tokenizers/collator、safetensors、checkpoint 元数据、adapter export |
| 评测/采样/插件 | eval、sampling、metrics、custom model/dataset/template、callback/web UI |
| 性能 | 真实尺寸训练或推理稳态、吞吐/延迟、显存口径和 fallback_count |

每个组至少选择一个纯 Transformers case、一个 ms-swift 私有组件 case 和一个公开入口；
多模态、RL、Megatron、服务等不能由 LoRA tiny case 代替。没有模型、数据、可选依赖、
外部服务或 CUDA kernel 时，保留明确的 blocked/not-run 原因。

## 工作流程

1. 阅读仓库 AGENTS.md、协作手册、环境合同、验收合同和最近结果；检查 dirty worktree、
   运行中的测试与缓存锁。保留本地改动，不自动 stash，不丢弃改动。
2. 获取并记录目标远端 SHA；固定 ms-swift/Jittor commit、解释器和 Python ABI、依赖版本、
   模型/数据/初始权重、seed、dtype、优化器、步数、GPU、Slurm job、节点和独立
   `JITTOR_HOME`。登录节点只编排和读写日志，所有 CUDA 计算、JIT、测试和 benchmark
   必须在 Slurm worker 上执行。
3. 先在独立 native PyTorch CUDA oracle 裸跑，再在 Jtorch candidate 复现。oracle 不得有
   shim 标记，candidate 必须有 shim 标记；模型、输入、checkpoint 和数据离线固定。
4. 失败按三行归属：能力、autograd、CUDA kernel 或 optimizer 修 Jittor core/backends/cuda；
   torch 名称、签名、返回结构、device/dtype/state 语义修 `jittor.compat.torch`；
   可迁移且确属 ms-swift 私有 glue 才进 adapter；runner/fallback/比较错误修 harness。
   不修改 ms-swift 源码绕过失败。
5. 每个面先最小复现，再按 L0-L5 推进；相同运行键和完整证据可复用。小问题直接修并补
   最小回归；无法解决的问题记录首个失败层、日志、影响、workaround 和解除条件，然后
   继续不依赖它的下一项。

## CUDA 设备合同

candidate 必须证明参数、输入、输出和梯度真实位于 CUDA，并报告
`device=cuda`、`use_cuda=true`。从 import/构造到结果保存使用
`backend_fallback=error`、`forbid_backend_fallbacks()`，完整窗口
`fallback_count == 0`。首次并行 JIT 使用不同 `JITTOR_HOME`/cache，首次编译串行；
benchmark 不与编译和单测共用缓存。

多卡必须使用安装好的 NCCL include/lib，设置 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`
以及 Jittor rank 变量；每个 child 的 `CUDA_VISIBLE_DEVICES` 与 local rank 一致。保存
准确命令、退出码、stdout/stderr、Slurm job/node/GPU、缓存路径和原始产物。

## L0-L5 验收

前一层失败时后续层为 blocked，不得跳过写 pass。

| 层 | 必须证据 |
| --- | --- |
| L0 构造 | native 与 shim 真实 import；模型、tokenizer、template、dataset、tuner、optimizer、trainer/engine 构造成功，状态键/device/dtype 一致 |
| L1 前向 | 同权重同输入 CUDA 前向；容器、shape、dtype、有限值、logits/hidden/loss 和适用 greedy token 一致 |
| L2 反向/更新 | 全部 trainable 参数和适用输入梯度、optimizer state/update、至少三步固定数据轨迹一致 |
| L3 恢复 | 同进程与全新进程恢复模型/adapter、buffers、optimizer、scheduler、RNG、dataloader 游标、global step 和后续轨迹 |
| L4 公开入口 | 真实 swift CLI、Python API、launcher 或适用 server/client 端到端，保留 worker、checkpoint 和推理输出 |
| L5 性能 | L0-L4 适用项通过后，真实尺寸预热并同步测量至少 10 次稳态；报告 latency/throughput、native/Jtorch ratio、显存和零 fallback |

CUDA 默认 forward 容差为 `5e-3`、backward 为 `2e-2`；BF16、RMSNorm、attention、
量化和 logits 另外报告最大绝对误差、相对 L2、dtype 和首个分歧层，不能事后放宽门槛。
tiny case 只能证明注册/API 面，不能替代真实 checkpoint、公开入口或真实尺寸性能。

## 报告与提交边界

原始日志、缓存、checkpoint 和环境放在
`$JITTOR_LAB_ROOT/_state/<topic>/<run>/`，不放进仓库。结果报告必须包含 surface
manifest、同步 SHA、dirty diff 摘要、运行键、job/node/GPU、依赖、准确命令、每个 case
的 L0-L5、首个失败、修复归属、误差、速度、显存、fallback 和 blocked/not-run 原因。

结束前执行 `git diff --check`、布局检查和受影响结构测试。提交时遵循
`github-collaboration-commit`：个人分支、功能粒度、明确路径；不提交缓存、日志、模型
或未完成实验产物。
