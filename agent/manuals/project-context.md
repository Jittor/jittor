# Jittor Project Context

- Status: Current-state index, not a history log
- Last reviewed: 2026-10-06
- Baseline reviewed: `1a6e203fc`
- Owner: Jittor core maintainers
- Freshness expires: 2027-01-06
- Review when: the release line, a top-level goal or an indexed location changes

Find where things are here, then read only the document your task needs.

## What the project is

Jittor is a JIT deep-learning framework built on meta-operators: Python builds a
graph, the C++ core in `src/` fuses and compiles kernels, and backends live under
`backends/<name>/` (CPU, CUDA, ROCm, ACL/Ascend, Corex, and `comm/{mpi,nccl,hccl}`).
The Torch compatibility layer is the separate `jittor-torch` project in `compat/`
(published as `jittor.compat`, including the `import torch` shim); downstream
library adapters live in `adapters/`. The goal is a maintainable, Torch-grade
framework that keeps the JIT/meta-operator design, under the gates in
[Torch compatibility principles](../../docs/compatibility/principles.md).

## Current state

The 2.0 restructuring has landed and the tree is in pre-release cleanup: the
refactor-period board, plans and handoff documents are gone, and task status now
lives in GitHub issues and pull requests. The accepted layout is in
[repository layout](../../docs/development/repository-layout.md); packaging targets
Python 3.7-3.13; [`noxfile.py`](../../noxfile.py) and
[`tools/run_test_suite.py`](../../tools/run_test_suite.py) are the maintained gate
surface (tiers `core` / `smoke` / full, see `AGENTS.md`). User-facing breaking
changes: [2.0 migration notes](../../docs/releases/2.0.md).

## Where information lives

| Kind | Location |
| --- | --- |
| Rules for agents and maintainers | [`AGENTS.md`](../../AGENTS.md), [collaboration](collaboration.md) |
| Entry index of manuals and skills | [agent-index.md](agent-index.md) |
| Environment, isolation, backend prerequisites | [environment.md](environment.md) |
| Open defects and limitations | [known-issues.md](known-issues.md) |
| Work waiting for hardware | [deferred-hardware.md](deferred-hardware.md) |
| Source architecture and module boundaries | [source architecture](../../docs/development/source-architecture.md), [development index](../../docs/development/index.md) |
| Test system and gate tiers | [test system](../../docs/development/test-system.md) |
| Mechanism notes (numerics, placement, profiling) | [notes index](../../docs/notes/index.md) |
| Torch compatibility | [compatibility index](../../docs/compatibility/index.md) |
| Backend guides (Ascend, Corex, MPI) | [guides index](../../docs/guides/index.md) |
| Performance method | [benchmarking](../../docs/performance/benchmarking.md) |
| Reproducible verification results | [results index](../../docs/results/index.md) |
| Reusable verification tools | `agent/skills/` (listed in [agent-index.md](agent-index.md)) |
| Task status, handoff, evidence | GitHub issues and pull requests |

## Active focus areas

## Active work areas

### Framework and compatibility

Native capabilities belong in their framework domain; spelling/adaptation lives
in `jittor.compat.torch`. Torch shim runtime and deployment are canonically owned
by `jittor.compat.shim`; `jittor.torch_shim` is only a same-object legacy alias,
and the deployed top-level `torch` package preserves that identity.
Project-specific patches do not belong in core. The Triton bridge is canonically
owned by `jittor.compat.triton`.

### Dtypes and autograd

Native complex64 supports the maintained arithmetic, reduction, structural,
FFT, linear-algebra bridge, and first-order gradient surface. Complex128,
second-order complex autograd, and several backend kernels remain explicit
limitations. See [native complex dtype](../../docs/notes/complex-dtype.md).

### Tests and performance

The test architecture uses independent forward references, numerical gradients,
and CPU-to-accelerator parity. See [test system](../../docs/development/test-system.md).
The complete CPU repository gate is
[`tools/run_test_suite.py`](../../tools/run_test_suite.py), which owns separate
native and Torch-mode processes, JIT caches, temporary directories, and process
mode variables. Current AArch64 verification passes native `768 passed, 738 skipped`, Torch `1591 passed, 536 skipped`, and clean structure `232 passed, 2 skipped`; see the [current CPU suite](../../refactor-wip/results/2026-09-01-current-cpu-suite.md) and the [ARM CPU stability report](../../refactor-wip/results/2026-08-30-arm-cpu-suite-stability.md).
The maintained CUDA gate also passes on a real RTX 4090 and covers the complete CUDA backend
directory, dtype coverage, CPU/CUDA device parity, TF32 controls, and strict
OpInfo CUDA references. The maintained CPU gate also passes with a fail-closed
independent binary PyTorch oracle, and compact ResNet18, ViT, GPT-2, and
diffusion UNet forward/backward parity passes on CPU and CUDA. ROCm, most
optional downstream dependencies, full training, and performance remain
separate gates. On a real 910B3, the maintained Ascend gate passes `397 passed, 9 skipped`; float16/float32 `arg_reduce` backward and float32/integer `prod` execute without CPU fallback.
Transformers 4.56.2 Qwen3-8B float32 loads all 8,190,735,360 parameters; SDPA,
greedy `arg_reduce`, and mask `all` run on ACL without CPU fallback. A native-shape `empty`
fast path brings 0.6B decode to 15.90 token/s versus native `torch_npu` 16.19 token/s.
Qwen3-0.6B BF16 SDPA passes zero-fallback generation at 14.92 token/s versus native 15.31 token/s.
Qwen3-0.6B FP32 eager forward/loss/backward also passes zero-fallback at `1.07x-1.12x` native `torch_npu`. Transformers 5.5.3 BF16 completes forward, backward, and AdamW without CPU fallback; explicit fused AdamW matches CANN/PyTorch for two fixed-gradient steps. BF16 embedding/RMSNorm/RoPE training kernels pass independent real-NPU references. After correcting Python-scalar promotion, RMSNorm rounding order, and BF16 SiLU, all 29 hidden states and logits match native `torch_npu` elementwise for the maintained one-step input. Eliminating 57 no-op full-slice gradients and lowering 112 continuous last-axis gradients to cached-zero CANN Cat preserves the exact snapshot and brings the current same-device protocol from `1.195x` to about `1.063x`; direct CANN RoPE reaches `0.988x` but is rejected because its logits and gradient trajectory differ. Cross-framework long training parity and the exact-path performance gate remain open. See the [training report](../../refactor-wip/results/transformers/2026-08-30-qwen3-ascend-training.md).
See the [Ascend guide](../../docs/guides/ascend-910b.md), [validation report](../../refactor-wip/results/2026-08-28-ascend-910b-validation.md), [arg-reduce](../../refactor-wip/results/2026-08-30-npu-arg-reduce-backward.md)/[product](../../refactor-wip/results/2026-08-30-npu-product-reduction.md) follow-ups, [Qwen3 inference report](../../refactor-wip/results/transformers/2026-08-28-qwen3-ascend-performance.md), complete [CPU](../../refactor-wip/results/2026-08-22-complete-cpu-test-suite.md)/[CUDA](../../refactor-wip/results/2026-08-22-cuda-test-suite.md) reports, and the [parallel follow-up](../../refactor-wip/results/2026-08-22-cuda-parallel-range-network-oracle.md).
The current fail-closed optional CUDA base gate passes 16 TorchMetrics,
MMCV/MMEngine, PEFT, TensorDict, and FlashAttention-adapter tests from one
retained cache; TorchMetrics is split by domain so cold compilation does not
consume one monolithic test timeout. See the
[optional CUDA report](../../refactor-wip/results/2026-08-24-optional-compat-cuda-gate.md).
Compact ResNet18, ViT, GPT-2, and diffusion UNet now also pass three-step SGD
loss, complete trainable-parameter, and shared-buffer trajectories on CPU and
real CUDA. A bilingual native-Jittor ResNet tutorial executes the same three-step
training/state-restore workflow in the maintained offline CPU notebook gate. See the
[common-network trajectory report](../../refactor-wip/results/2026-08-26-common-network-training-trajectories.md).
The same report records the real-scale follow-up: UNet is accepted at `0.79x`,
ConvNet improved to `1.08x`, and ViT remains open at about `1.33x` because its
dominant CUDA GEMMs lag the PyTorch reference.
Performance work uses isolated caches, synchronization, and exact commit labels.
The ecosystem harness verifies twelve Transformers/Diffusers/PEFT/ms-swift/MMCV/MMEngine CPU/CUDA cases; its NPU scope verifies Diffusers UNet2D, MMCV/MMEngine, and ms-swift LoRA Llama forward and every gradient against `torch_npu` with zero CPU paths.
Diffusers correctness and maintained float32 performance are accepted at `0.964x` native `torch_npu`; the tiny OpenMMLab NPU cases now pass at `0.927x/0.796x`. See the [Diffusers](../../refactor-wip/results/2026-08-30-diffusers-ascend-parity-performance.md) and [OpenMMLab](../../refactor-wip/results/2026-08-30-mmcv-mmengine-ascend-parity.md) reports. The tiny ms-swift LoRA case uses fused float32 causal SDPA training and passes at `0.969x`; see the [ms-swift Ascend report](../../refactor-wip/results/2026-08-31-ms-swift-ascend-parity-performance.md).
On a real 910B3, the locked verl core algorithms pass exact loss/gradient parity
against `torch_npu` for vanilla PPO, GSPO, SAPO, GPG, geometric-mean, CISPO, and
GRPO with zero CPU fallback; only GPG passes the NPU micro-performance protocol,
while full NPU workers/FSDP2/rollout/PPO remain open. The CPU/CUDA gate passes
four-rank NCCL/FSDP2 and tiny Qwen3 PPO; see the [Ascend core report](../../refactor-wip/results/2026-09-02-verl-ascend-core-algorithms.md)
and [CUDA PPO report](../../refactor-wip/results/2026-08-24-verl-weight-transfer.md).
The external NPU vLLM adapter on current HEAD passes public `vllm.LLM.generate` for Qwen3-0.6B with exact four-token parity, zero CPU fallback, and no loaded `torch_npu`/`vllm_ascend`. Preserving BF16 parameters and grouping the maintained CANN serving operations, including the exact Q/K RMSNorm and RoPE sequence, reduce its pooled warm-request median from about `0.615s` to `0.36330s`. The current comparable native `vllm-ascend` baseline is `0.38998s`, so restricted single-request, short-context, unquantized TP=1 correctness and performance are accepted; broader serving coverage remains open. See the [vLLM Ascend report](../../refactor-wip/results/2026-08-31-vllm-ascend-jittor-bootstrap.md).
CUDA vLLM 0.24 TP=2 follow-up (2026-09-27): [communication, model checks and boundaries](../../docs/results/2026-09-27-vllm-multigpu.md). Qwen3-0.6B vLLM real-CUDA inference
now runs about 20.5% faster than its real-PyTorch reference on the maintained
4-token protocol; TRELLIS.2 improved from about 1.20x to 1.093x slower, so its
performance gate remains open. CUDA masked SDPA now reuses the safe softmax
kernel instead of building redundant row-valid graphs. CUDA training LayerNorm
and standard RMSNorm now use fused forward/backward capabilities. The default
math-attention path remains `1.13x/1.22x` slower for GPT-2/Llama; with explicitly
configured native FlashAttention and float32-to-fp16 cast, GPT-2 reaches
`0.90-0.94x`, while Llama retains a conservative `3-4%` gap. See
[benchmarking](../../docs/performance/benchmarking.md) and the
[ecosystem parity/performance report](../../refactor-wip/results/2026-08-23-ecosystem-parity-performance.md),
the [verl/vLLM/TRELLIS current-baseline report](../../refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md),
the [CUDA masked SDPA report](../../refactor-wip/results/2026-08-23-cuda-masked-sdpa.md), and the
[Transformer normalization follow-up](../../refactor-wip/results/2026-08-26-transformers-training-normalization.md).

### Agent-operable optimization

The proposal for structured compiler observation and bounded optimization is
research only; no autonomous mutation path is implemented. See
[agent-operable framework optimization](../../refactor-wip/research/agentic-optimization.md).

## Before running work

Sync with the target remote branch as `AGENTS.md` requires and record the SHA;
isolate the run per [environment](environment.md); search
[known-issues.md](known-issues.md) and the [results index](../../docs/results/index.md)
for existing evidence; reproduce minimally before editing. Update this page only
when its current-state summary or a location changes.
