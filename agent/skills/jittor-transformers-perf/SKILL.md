---
name: jittor-transformers-perf
description: Reproducible CUDA performance analysis for Transformers on import-jittor-as-torch versus real PyTorch. Use for operator, Transformer-block, HF-model, SDPA, gradient-management, or optimizer-step benchmarks where Jittor lazy execution and backend/version isolation must be handled correctly.
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

## 工具路由

- 通用算子初始基线：`benchmark_transformer_bottlenecks.py`
- forward/backward 算子与 Transformer block：`benchmark_training_hotspots.py`
- 同版本 Tiny Llama/BERT/ViT：`benchmark_hf_tiny_models.py`
- GELU/softmax kernel 与 build/profile：`probe_softmax_gelu.py`
- 大词表、LayerNorm 边界正确性：`audit_kernel_changes_gpu.py`
- clip/scaler 消融：`probe_clip_grad_norm.py`、`benchmark_grad_management.py`
- 完整 SGD/AdamW step：`benchmark_optimizer_step.py`
- SDPA layout/Flash 物化：`probe_sdpa_layout_materialization.py`

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
