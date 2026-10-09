# DeepSpeed 0.17.6：单机双 Ascend NPU 训练验收

- Status: 单机双 NPU、FP32、ZeRO-1/2/3 已验证。
- Date: 2026-10-09。
- Baseline commit: 官方 `2.0-refactor@7a18abf295668d9b19da5fa1657f5606e84b65a0`；候选基于 PR #18039 的 `669c10795cf4948472ed78ddac9d5de4919e5095`，并包含本 PR 的修复。
- Owner: @shaoniannizhendou。
- Review when: DeepSpeed 固定版本或源码散列、`jittor.compat.transaction.owned_runtime_hook` / `active_transaction`、`module_patcher.patch_method`、`jittor.distributed.get_hccl_world_info`、HCCL 通信、Torch 兼容层参数数据重绑定或优化器梯度接口发生变化时复验。

验收使用 DeepSpeed 0.17.6、Transformers 4.56.2、原始 Qwen3-0.6B 权重（SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`）、FP32 AdamW。单节点两个 Ascend NPU 各运行一个进程，HCCL world size 为 2；Jittor 端设置 `JT_BACKEND_FALLBACK=error`，模型和输入均离线固定。参考端为独立的 PyTorch 2.7.1 与 torch_npu 2.7.1.post4 进程。

## 结果

| ZeRO 阶段 | 四步 Qwen loss 对照：两 rank 共 8 点最大绝对差 | 正式训练单步中位数（rank 0 / rank 1） | Jittor 回退增量 |
| --- | ---: | ---: | ---: |
| 1 | `1.811981201e-5`（job 2086） | `0.9661 / 0.9649 s`（job 2128） | `0 / 0` |
| 2 | `1.525878906e-5`（job 2086） | `0.9579 / 0.9576 s`（job 2095） | `0 / 0` |
| 3 | `1.525878906e-5`（job 2086） | `3.5193 / 3.5196 s`（job 2094） | `0 / 0` |

作业 2086、2128、2095、2094 均以状态 `COMPLETED`、退出码 `0:0` 结束。正式训练每 rank 完成 10 次预热和 5 次采样，六份报告均为 `passed`、输入梯度非零。计时范围为清梯度、前向、反向、优化器更新及同步；这些是实测值，不作为性能上限声明。ZeRO-1/3 启用显存峰值统计；ZeRO-2 统计模式在双卡作业 2087、2088 中触发 `nano_vector.h:41: slice overflow`，因此 job 2095 关闭该统计，未验收 ZeRO-2 峰值显存，详见 `KI-MEM-004`。关闭统计后，同一组 2、3 号卡上双 rank 的 16 步训练均通过。使用 4×4 Toy 模型的双卡检查点作业 2127 在各阶段训练两步、保存并用新 engine 恢复，六个 rank/阶段的恢复前后 loss 绝对差均为 0；作业状态为 `COMPLETED`、退出码 `0:0`。

## 复验条件

在隔离的单机双卡环境中，先确认两张分配卡无本任务残留进程；每个 rank 使用独立 JIT 缓存。参考端与 Jittor 端使用相同权重、固定输入和优化器，各训练四步并逐 rank/step 比较 loss。正式训练按 `DS_ZERO_STAGE=1,2,3` 分别运行，设置 `DS_L5_WARMUPS=10`、`DS_L5_SAMPLES=5`，检查六份报告的设备、HCCL world size、梯度非零与 `fallback_delta=0`；ZeRO-2 计时不启用 `profile_memory_enable`，不检查峰值显存。启动形式为 `srun --nodes=1 --ntasks=2 --gres=gpu:2 --kill-on-bad-exit=1 <独立验收脚本>`；适配器激活与配置见 `adapters/jittor_adapters/deepspeed/README.md`。

本结论只覆盖上述单机双 NPU、FP32、固定模型及配置。HCCL reduce-scatter 当前由 all-reduce 加设备端切片实现，因此此处不作通信性能或显存节省声明；多机未纳入验收。