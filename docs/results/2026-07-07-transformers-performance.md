# Jittor Transformers 性能瓶颈分析（2026-07-07）

✅ 已完成 GPU 首轮分析与低风险加速补丁。

## 目标

分析 `import jittor as torch` 跑 transformers 时相对 PyTorch 的速度瓶颈，优先 GPU 跑通并量化，再评估低风险加速点。

## 工作文档

用户要求中间产物放在 `./<project_name>` 下；本次使用 `${JITTOR_LAB_ROOT}/jittor_transformers_perf/`：

- `benchmark_transformer_bottlenecks.py`：micro/attention/MLP benchmark。
- `run_perf_env.sh`：固定 jt311、JTCUDA、项目内 runtime/cache 的运行环境。
- `results/`、`logs/`：测试结果和日志。

## 当前结论

- 只读源码分析显示：CUDA GEMM 已有 `CUBLAS_COMPUTE_32F_FAST_TF32` 底层分支，但 torch shim 的 `torch.backends.cuda.matmul.allow_tf32` 之前没有接到 CUDA，只接了 ACL/HF32。
- `scaled_dot_product_attention` 在未命中 flash-attn 时会落到 `bmm + mask + softmax + bmm`，会物化 scores；训练态、mask/dropout/fp32 场景大多走此路径。
- Python/shim 层还有训练同步风险：`loss.backward()` 预同步、`clip_grad_norm_` `.item()`、GradScaler per-grad `.item()`、MoE grouped_mm fallback 的 `.numpy().tolist()`。

## 本轮改动

✅ 已实现并验证：

- 新增 `jt.flags.cuda_allow_tf32`，只用于 CUDA float32 matmul/bmm 的 TF32 compute。
- `torch.backends.cuda.matmul.allow_tf32` 与 `torch.set_float32_matmul_precision("high"/"medium")` 接到该 flag；默认仍为 `0`，保持严格 fp32。
- 不复用 `use_tensorcore`，避免顺带改变 fp16/bf16 GEMM 的累加/compute 语义。

## Corrected Benchmark

脚本修正点：计时段保留所有输出，避免 Jittor lazy 图只执行最后一次输出造成 matmul 假快；`slots >= repeats`，避免同输入 CSE。

结果文件：

- `${JITTOR_LAB_ROOT}/jittor_transformers_perf/results/summary.md`
- `${JITTOR_LAB_ROOT}/jittor_transformers_perf/results/summary.json`
- 原始 JSONL：`bottlenecks_corrected_{torch,jittor}_tf32{off,on}.jsonl`

| case | torch fp32 ms | torch TF32 ms | jittor fp32 ms | jittor TF32 ms | jittor/torch fp32 | jittor/torch TF32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul_big_batched_16x1024 | 0.8565 | 0.4635 | 0.7437 | 0.4878 | 0.87x | 1.05x |
| matmul_small_batched_32x512 | 0.1942 | 0.1321 | 0.2033 | 0.1386 | 1.05x | 1.05x |
| softmax_32x256x1024 | 0.0762 | 0.0763 | 0.1480 | 0.1815 | 1.94x | 2.38x |
| gelu_32x256x1024 | 0.0752 | 0.0751 | 0.2019 | 0.2034 | 2.69x | 2.71x |
| relu_32x256x1024 | 0.0747 | 0.0747 | 0.0801 | 0.0791 | 1.07x | 1.06x |
| layernorm_16x128x768 | 0.0151 | 0.0156 | 0.0367 | 0.0358 | 2.42x | 2.30x |
| sdpa_math_8x16x128x64 | 0.0401 | 0.0410 | 0.1252 | 0.1272 | 3.13x | 3.10x |
| mlp_16x128x768_3072 | 0.5590 | 0.3204 | 0.7266 | 0.5045 | 1.30x | 1.57x |

## 结论

✅ 已修/改善：

- CUDA TF32 opt-in 后，Jittor GEMM 已基本追到 PyTorch TF32：大 batched matmul `1.05x`，小 batched matmul `1.05x`。
- MLP 端到端从 `0.7266ms` 降到 `0.5045ms`，但仍比 PyTorch TF32 `1.57x`，说明 `matmul + bias + gelu + matmul + bias` 的非 epilogue/fusion 开销仍在。

🔴 主要剩余瓶颈：

- `softmax`：Jittor fp32 约 PyTorch `1.94x`，TF32 不相关。
- `gelu`：约 `2.7x`，组合 elementwise/fusion 仍不如 PyTorch。
- `layernorm`：约 `2.3-2.4x`，训练/通用路径不是完整 fused CUDA forward/backward。
- `sdpa_math`：约 `3.1x`，fallback 会物化 scores，缺训练态/带 mask fused SDPA。

## 验证

✅ `python/jittor/test/test_torch_compat_cuda_tf32.py`：1/1 OK。

✅ 独立进程数值差异：同一 1024x1024 matmul，TF32 off sum `-12415.37`，TF32 on sum `-12413.40`，`max_abs=0.04896`，说明 cuBLAS FAST_TF32 生效。

🟡 NPU 未复验：本次机器为 NVIDIA 4090；`allow_tf32` 仍会设置 `jt.acl_allow_hf32`，但需在 910B 上确认 ACL 行为未回归。

## 后续建议

1. 训练态/带 mask SDPA：优先做 fused SDPA 或接 flash-attn varlen/mask 路径，这是 transformers attention 最大剩余差距。
2. GELU/LayerNorm fused kernel：尤其 MLP 中 `linear -> bias -> gelu` 的 epilogue 或 pattern fusion。
3. cublasLt：支持 heuristic、workspace、bias/GELU epilogue，可继续压缩 MLP 差距。
4. 训练同步：单独量化 `loss.backward()` pre-sync、`clip_grad_norm_` `.item()`、GradScaler per-grad `.item()`，再决定是否加开关或 device-only 实现。
