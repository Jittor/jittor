---
name: jittor-transformers-perf
description: Reproducible CUDA and Ascend performance analysis for Transformers on import-jittor-as-torch versus real PyTorch. Use for operator, Transformer-block, HF-model, SDPA, gradient-management, or optimizer-step benchmarks where Jittor lazy execution and backend/version isolation must be handled correctly.
---

# Jittor Transformers 性能基准

本 skill 的可复用源码位于 `scripts/`，运行状态与结果默认写到
`${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}/jittor_transformers_perf/`。
目标是得到能解释、能复现的
Jittor vs 真 PyTorch 数据，而不是把 JIT、H2D 或 lazy graph 漏执行当成性能结论。

## 环境

- Jittor：`/home/zy/miniconda3/envs/jt311/bin/python`，仓库源码 `python/jittor`。
- PyTorch oracle：`/home/zy/rt_venv/bin/python`。运行前必须断言 `torch.__file__`
  不含 Jittor shim。
- 所有命令通过 `agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh`，
  并为并行任务设置唯一 `cache_name` 与 `CUDA_VISIBLE_DEVICES`。缓存、日志和
  结果必须留在 lab 工作区，不得写入 Jittor 主仓库。

## 不可省略的计时规则

1. 预分配多个不同输入 slot；计时前触碰所有 slot，排除首次 H2D residency。
2. 计时内保留每次输出；训练还要保留每次梯度，最后统一同步。只保留最后一个输出会让
   Jittor lazy executor 漏执行前面的迭代，产生假快。
3. 首次 JIT/extension 编译不计入稳态；同时记录 cold build 时必须单独标注。
4. SDPA 必须区分 PyTorch `default`、forced `math` 和 forced `flash`。Jittor math
   fallback 不能直接与 PyTorch default fused kernel 归因成“softmax 慢”。
5. 训练结果必须检查每个目标梯度 finite 且非零。显式 cuBLAS op 若缺反向，会产生看似
   很快的静默零梯度。
6. HF 对比必须使用同一 Transformers 版本。`benchmark_hf_tiny_models.py` 会把真
   PyTorch/torchvision 固定在 `rt_venv`，再加载 jt311 的 Transformers 4.56.2。
7. allocator 指标只能作为 harness 工作集方向性数据；严格峰值需另用 NVML/进程级采样。
8. SDPA 同时报告 `--sync-mode per_call` 延迟与 `queued` 吞吐。前者每步同步，后者保留
   全部输出/梯度后统一同步；二者不可互相替代。per-call 结果还分别记录 graph/build
   与 sync 中位数，用于区分 Python/grad 构图和设备执行开销。

## 工具路由

- 通用算子初始基线：`benchmark_transformer_bottlenecks.py`
- forward/backward 算子与 Transformer block：`benchmark_training_hotspots.py`
- 同版本 Tiny Llama/BERT/ViT：`benchmark_hf_tiny_models.py`
- GELU/softmax kernel 与 build/profile：`probe_softmax_gelu.py`
- 大词表、LayerNorm 边界正确性：`audit_kernel_changes_gpu.py`
- clip/scaler 消融：`probe_clip_grad_norm.py`、`benchmark_grad_management.py`
- 完整 SGD/AdamW step：`benchmark_optimizer_step.py`
- SDPA layout/Flash 物化：`probe_sdpa_layout_materialization.py`
- Ascend Qwen3 整模推理：`benchmark_qwen3_ascend.py`

Ascend 整模对拍必须为 Jittor 和原生 `torch_npu` 使用独立 Python 进程、相同
Transformers 版本和相同本地 checkpoint。运行 Jittor 一侧前设置
`JITTOR_TORCH_SHIM=1`，并在加载权重后显式迁移到 NPU；两侧都通过
`ASCEND_RT_VISIBLE_DEVICES` 选择已分配设备。首次 JIT 单列，稳态 prefill 和
generation 在每次样本后同步；单 token 与多 token decode 都要测，不能用一次
generation 成功推断 KV-cache 后续步可用。Jittor 计时结束后还会在日志捕获范围内
各执行一次 prefill 和 generation，发现 CPU fallback 时直接失败。可用
`--logits-output` 保存末 token logits，再在进程外比较 argmax、Top-K 和全量误差。
Jittor 侧可用 `--profile-output /path/to/report.json` 在所有计时结束后额外执行一次
generation，并保存聚合 profiler row；该次运行不进入稳态样本。profile 不能在
原生 PyTorch backend 使用。`--pipeline-ops` 必须通过成对 A/B 决定，旧图上的最佳
阈值不能直接作为新算子组合的默认值。

`benchmark_training_hotspots.py` 的 SDPA 模式支持可配置 batch/head/sequence/head-dim、
causal 与同步口径。Jittor `math` 显式绕过 native loader，`flash` 要求且校验 official
backend，`direct` 使用预物化 BSHD 输入，`default` 验证生产 dispatch。非 required、
无 dropout 的训练默认在 `B*H*Lq*Lk < 2^24` 时选择 math；可用
`JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES` 调整，设为 0 则禁用该阈值。

## 常用命令

```bash
JITTOR_LAB_ROOT=${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
CUDA_VISIBLE_DEVICES=2 cache_name=hf_tiny_gpu2 \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  /home/zy/miniconda3/envs/jt311/bin/python \
  "$SCRIPT_ROOT/benchmark_hf_tiny_models.py" \
  --backend jittor --model bert --phase forward --repeats 20
```

```bash
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
JITTOR_FLASH_ATTN_JITTOR_SRC=/path/to/flash-attention \
JITTOR_FLASH_ATTN_HEAD_DIMS=64 JITTOR_FLASH_ATTN_DTYPES=fp16 \
CUDA_VISIBLE_DEVICES=2 cache_name=sdpa_train_gpu2 \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  /home/zy/miniconda3/envs/jt311/bin/python \
  "$SCRIPT_ROOT/benchmark_training_hotspots.py" \
  --backend jittor --case sdpa --phase fwd_bwd --dtype float16 \
  --sdpa-backend default --sync-mode per_call \
  --batch 4 --heads 12 --length 1024 --head-dim 64 --repeats 12
```

```bash
JITTOR_LAB_ROOT=${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}
SCRIPT_ROOT=agent/skills/jittor-transformers-perf/scripts
CUDA_VISIBLE_DEVICES=6 cache_name=softmax_boundary_gpu6 \
  "$SCRIPT_ROOT/run_perf_env.sh" \
  /home/zy/miniconda3/envs/jt311/bin/python \
  "$SCRIPT_ROOT/audit_kernel_changes_gpu.py" \
  --task softmax-case --length 50257 --rows 2 --dtype float32
```

详细结果优先查 `agent/results/transformers/`；数值精度对拍仍使用
`agent/skills/jittor-torch-diff/`。
