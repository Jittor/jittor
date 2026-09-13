# Llama 3.1 70B No Robots 正式 SFT 曲线与资源对比

- 状态：PASS（真实 CUDA 的代表性正式短训练；不是全参数 70B SFT 或总体 L5）
- 验收日期：2026-09-07（Asia/Shanghai）
- 分支与提交：`feature/cgq_transformers@fba9982421c42038f2a8391e7e96cf26a9e70a4b`
- 已推送共同基线：`508f816b81bdacba5ebdb72ce82b9bab3181b2ec`
- Transformers / PyTorch / Jittor：`4.56.2` / `2.6.0+cu124` / `1.3.11.0`
- 硬件：6 张 NVIDIA A800 80GB，以进程内逻辑 rank 记录设备身份
- 维护者：Jittor Transformers compatibility
- 复查条件：FSDP2 分片/生命周期、Llama 实现、数据处理或训练 harness 变化

## 结论

在同一份 Llama 3.1 70B 权重、`HuggingFaceH4/no_robots` SFT 样本、tokenizer、padding、seed、
optimizer、BF16、序列长度和有效 batch 下，原生 PyTorch 与 Jittor Torch adapter 都完成了
60 个 optimizer step、3 个完整 epoch 和全部 80 层 forward。两条 loss 曲线都下降且几乎重合：

| 指标 | PyTorch | Jittor |
| --- | ---: | ---: |
| epoch 1 mean | 1.9181586750 | 1.9186480756 |
| epoch 2 mean | 1.8038813457 | 1.8052478577 |
| epoch 3 mean | 1.7209118025 | 1.7214694288 |
| 起止 loss | 2.250263095 / 1.894611121 | 2.246716608 / 1.895229856 |
| 起止绝对降幅 | -0.3556519747 | -0.3514867524 |
| 降幅比（Jittor/PyTorch） | - | 0.9883 |
| 拟合线性斜率 | -0.0056555497 | -0.0056496181 |

本轮验收比较曲线下降，不要求逐 step loss 相等。图中细线是 raw loss，粗线是 5-step rolling
mean，竖线是 epoch 边界：

`$JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1/loss-curves.png`

## 固定协议

- Checkpoint：本地 `meta-llama/Llama-3.1-70B@349b2ddb53ce8f2849a6c168a81980ab25258dac`，
  index SHA256 `2abe0910e23770a30ccf9b1b91804c64831c47f9c98defaa5293aa999433fc2b`，权重
  `141107412992` bytes。
- 数据：`HuggingFaceH4/no_robots@e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`，预处理 120 条、
  3 epochs、sequence length 64；预处理 JSONL SHA256
  `53b182c99a5bafebd61f765103664ae04624b02e07106798dc3d594a7fc37e30`。数据集许可证为
  `CC BY-NC 4.0`，仅适合遵守该许可的非商业用途。
- right padding、每卡 batch 1、有效 batch 6、seed `20260907`、eager attention。
- SGD `lr=0.001`、momentum 0、weight decay 0；无 LoRA、autocast、GradScaler 或 CPU offload。
- 完整 80 层执行 forward；只训练最后 decoder block、final RMSNorm 和 `lm_head`，11 个
  trainable tensors（`1,906,335,744` 参数），其余 712 个 tensor 冻结。
- 正式曲线 run 使用 `tensor-check-steps=0`，只保存 loss、样本 ID、计时、吞吐和显存标量；
  完整 step-0 梯度/更新存在性来自同一 harness 的上一轮 5-step 代表性实验，不能把本轮
  说成重新保存了全部梯度数组。

## 时间拆分

| 阶段 | PyTorch | Jittor | 说明 |
| --- | ---: | ---: | --- |
| launcher wall | 2424.604 s | 5271.966 s | Jittor 正式 run；另有串行预热 1436.341 s |
| process wall | 2419.595 s | 5269.655 s | 进程启动到结束 |
| model construction/load/shard | 42.877 s | 730.422 s | Jittor `checkpoint_read_assign=531.387 s`、`fsdp_shard=148.187 s` |
| training loop | 2368.634 s | 4413.291 s | 包含 60 steps 及同步/曲线比较 |
| cold step | 39.801 s | 1784.525 s | Jittor 首个真实 step 含大量惰性 CUDA JIT；不能拆成纯编译秒数 |
| steady mean（step 1-59） | 39.314 s | 43.851 s | Jittor / PyTorch = `1.115x` |

Jittor 的 clean-cache 总墙钟不能直接当成稳定态训练速度。PyTorch 没有对应的 Jittor core/NCCL
串行预热和惰性 kernel 编译；因此总 launcher 比值主要反映启动固定成本。模型加载分项显示，
Jittor 的瓶颈集中在 safetensors 分片读取/assign 与 FSDP2 shard 阶段，可能涉及 Python shim
holder 替换、逐 tensor assignment、同步和 lazy materialization；没有 profiler 时不把它归因到
单一函数。

## 显存拆分

| 口径 | PyTorch | Jittor |
| --- | ---: | ---: |
| framework overall peak | 27.171 GiB | 67.854 GiB |
| step 0 framework peak | 27.163 GiB | 31.409 GiB |
| step 1-59 framework peak | 27.171 GiB（基本不变） | 33.239-67.854 GiB（持续上升） |
| independent `nvidia-smi` max | 33.946 GiB（GPU 8） | 68.493 GiB（各卡） |

显存图和阶段图：

`$JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1/resource-profiles.png`

这里的 Jittor 高显存不是只由 cold step 造成：step 0 是约 31.4 GiB，随后每步高水位增加，
到 step 59 达到约 67.85 GiB，独立采样也达到 68.49 GiB。当前最谨慎的解释是 Jittor lazy
graph、FSDP all-gather/reshard 产生的中间 Var，以及 allocator/cache 的生命周期在 torch
shim 训练循环中没有像 PyTorch FSDP2 那样及时回收；`jt.gc()` 已被调用，但没有阻止该高水位
增长。框架峰值来自 `torch.cuda.max_memory_allocated()` 风格 shim 观测，物理峰值来自独立
`nvidia-smi` 采样，两者都是真实 CUDA 观察值，但 allocator 统计语义并非完全同口径。这个
显存随 step 累积的问题应作为后续 adapter/Jittor 性能缺陷继续定位，不能外推为所有模型均有
同样比例的显存差异。

## 原始证据与复现

原始结果、日志、采样 CSV、隔离 `JITTOR_HOME` 和图均未版本化，保存在：

`$JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1/`

`$JITTOR_LAB_ROOT/_state/transformers_70b_sft/no-robots-formal-60step-r1/`

正式自动队列命令：

```bash
export JITTOR_CUDA_CANDIDATES="<comma-separated candidate devices>"
$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python \
  $JITTOR_LAB_ROOT/transformers_70b_sft/run_formal_pair_when_idle.py \
  --state-root $JITTOR_LAB_ROOT/_state/transformers_70b_sft/no-robots-formal-60step-r1 \
  --output-root $JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1 \
  --candidates "$JITTOR_CUDA_CANDIDATES" \
  --idle-memory-mib 128 --idle-utilization-percent 5 --stable-checks 3 \
  --poll-seconds 15 --steps 60 --max-layers 80 --sequence-length 64 \
  --seed 20260907 --learning-rate 0.001 --master-port 29627
```

重新绘图：

```bash
$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python \
  $JITTOR_LAB_ROOT/transformers_70b_sft/plot_loss_curves.py \
  --pytorch $JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1/pytorch \
  --jittor $JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1/jittor \
  --output $JITTOR_LAB_ROOT/transformers_70b_sft/runs/no-robots-formal-60step-r1/loss-curves.png \
  --smooth-window 5 \
  --title 'Llama 3.1 70B — No Robots SFT (120 samples, 3 epochs)'
```

## 限制

这是固定 batch、64 token、60 step 的 BF16 短训练，只训练最后 block、norm 和 lm head；没有
测试集效果，也不是全参数 70B optimizer parity。Jittor 首次 JIT/cold step、分片加载成本和
显存随 step 的高水位增长仍是本实验的已知限制。正式 run 没有重新保存全部梯度数组；更广
的 dtype、长上下文、Trainer/Accelerate/PEFT、ROCm/NPU 和其他模型仍需独立验证。
