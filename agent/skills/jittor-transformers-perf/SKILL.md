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

- 原生与独立 Torch frontend 的 NPU 矩阵乘：`benchmark_npu_frontends.py`（见下节）
- 通用算子初始基线：`benchmark_transformer_bottlenecks.py`
- forward/backward 算子与 Transformer block：`benchmark_training_hotspots.py`
- 同版本 Tiny Llama/BERT/ViT：`benchmark_hf_tiny_models.py`
- GELU/softmax kernel 与 build/profile：`probe_softmax_gelu.py`
- 大词表、LayerNorm 边界正确性：`audit_kernel_changes_gpu.py`
- clip/scaler 消融：`probe_clip_grad_norm.py`、`benchmark_grad_management.py`
- 完整 SGD/AdamW step：`benchmark_optimizer_step.py`
- SDPA layout/Flash 物化：`probe_sdpa_layout_materialization.py`
- Ascend Qwen3 整模推理：`benchmark_qwen3_ascend.py`

### 独立前端 NPU 最小基准

`benchmark_transformer_bottlenecks.py` 仍使用历史 `import jittor as torch` 与命名空间
替换，不能证明独立 Torch frontend。新的 `benchmark_npu_frontends.py` 每次只运行一个
模式，compat 显式调用 `activate(independent_namespace=True)`，检查独立 Tensor 身份。
它不通过旧 `run_perf_env.sh` 激活环境；先按昇腾指南加载 CANN，在相同已分配设备上
用以下两个独立进程顺序执行（外部设置 `ASCEND_RT_VISIBLE_DEVICES`）：

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
perf_run="$JITTOR_LAB_ROOT/_state/ascend-validation/matmul-$(date +%Y%m%d-%H%M%S)"
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_npu_frontends.py
perf_python="${JITTOR_CI_PYTHON:-python3}"
JITTOR_HOME="$perf_run/native-cache" cache_name=npu_matmul_native \
  "$perf_python" "$perf_script" --mode native --output "$perf_run/native.json"
JITTOR_HOME="$perf_run/compat-cache" cache_name=npu_matmul_compat \
  "$perf_python" "$perf_script" --mode compat --output "$perf_run/compat.json"
```

解释器需已安装当前核心和独立 `jittor-torch` 兼容发行版；源码 `PYTHONPATH`
本身不会安装独立包。`jittor.compat.shim` 是安装后的公开入口，不应改为 native alias。
用与测试相同的已安装解释器设置 `JITTOR_CI_PYTHON`；例如 task venv 的 `bin/python`。
先在对应的隔离缓存中只验证导入与身份，再启动计时：

```bash
JITTOR_TORCH_SHIM=0 JITTOR_HOME="$perf_run/compat-cache" \
  "$perf_python" -c 'import jittor as jt; from jittor.compat.shim import activate; activate(independent_namespace=True); import torch; assert torch is not jt and torch.Tensor is not jt.Var; print(torch.__file__, torch.Tensor.__module__)'
```

模板默认 float32、256×256 矩阵、3 组输入、
10 次预热和 50 次计时；用 `--m --k --n --slots --warmup --repeats --seed` 修改。
每组输入先与 NumPy float64 累积的矩阵乘对拍，检查设备驻留，预热后逐次同步计时，
最后再对拍。整个计算处于零回退守卫内；任何 ACL、数值或驻留失败都不输出成功报告。
记录每次延迟、median、p95、回退计数、设备快照、Python/包来源、提交和 dirty 补丁
摘要；原始 JSON 只允许写到 lab 下且在 checkout 外。保存运行前驱动/固件/CANN 版本
快照，并按 [NPU 记录模板](../../../docs/development/npu-validation-templates.md) 填写报告。

本入口测量单次前向的 Python 构图到同步完成延迟，不含 D2H、首次编译或训练。
保持各组 shape、精度和设备负载一致，多轮交错重跑后再形成比较结论；微基准结果
不能外推到整个模型。此入口的代码审查/语法检查不代表真实 NPU 性能已经验证。

Ascend 整模对拍必须为 Jittor 和原生 `torch_npu` 使用独立 Python 进程、相同
Transformers 版本和相同本地 checkpoint。运行 Jittor 一侧前设置
`JITTOR_TORCH_SHIM=1`，并在加载权重后显式迁移到 NPU；两侧都通过
`ASCEND_RT_VISIBLE_DEVICES` 选择已分配设备。首次 JIT 单列，稳态 prefill 和
generation 在每次样本后同步；单 token 与多 token decode 都要测，不能用一次
generation 成功推断 KV-cache 后续步可用。Jittor 每次 prefill 和 generation 都在
`forbid_backend_fallbacks()` 中执行并同步；原生策略设为 `error`，退出时核对尝试计数，
即使内部吞掉回退异常也会失败，不再依赖日志文字。计时结束后另外各执行一次验证。可用
`--logits-output` 保存末 token logits，再在进程外比较 argmax、Top-K 和全量误差。
Jittor 侧可用 `--profile-output /path/to/report.json` 在所有计时结束后额外执行一次
generation，并保存聚合 profiler row；该次运行不进入稳态样本。profile 不能在
原生 PyTorch backend 使用。`--pipeline-ops` 必须通过成对 A/B 决定，旧图上的最佳
阈值不能直接作为新算子组合的默认值。两侧都可用
`--python-profile-output /path/to/profile.prof` 在计时后额外执行一次 generation，
保存可由 `python -m pstats` 读取的 cProfile 数据；profiler 会放大墙钟时间，只用它
比较调用结构和热点，不把该次执行写入性能样本。

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

详细结果优先查 `docs/results/transformers/`；数值精度对拍仍使用
`agent/skills/jittor-torch-diff/`。
