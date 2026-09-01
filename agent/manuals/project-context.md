# Jittor Project Context

- Status: Current index, not a history log
- Last reviewed: 2026-09-01
- Baseline reviewed: `406f56e1`
- Owner: Jittor core maintainers
- Freshness expires: 2026-11-12
- Review when: a modernization stage lands, a top-level goal changes, or an
  indexed contract becomes stale
This file is the short handoff entry for repository work. Read only the linked
document relevant to the task; do not append experiment transcripts or completed
history here.

## North star

The project is moving Jittor toward a maintainable, Torch-grade framework while
preserving its JIT/meta-operator design. Changes are accepted only when they meet
the correctness, real-device, composition, maintainability, and evidence gates
defined in [Torch compatibility principles](../../docs/architecture/torch-compatibility-principles.md).

Operational rules:

- verify before fixing;
- prefer a clear unsupported error over a silent wrong result;
- exercise every advertised backend rather than relying on CPU fallback;
- keep one canonical implementation and explicit compatibility aliases;
- isolate caches and record reproducible evidence.

## Current repository state

The accepted repository shape and packaging boundaries are documented in
[Jittor repository layout](../../docs/architecture/repository-layout.md). The
current tree has:

- normal packages for `jittor.nn`, `jittor.misc`, `jittor.pool`, and
  `jittor.compat.torch`;
- compatibility domains under `jittor.compat`, including FSDP2 and Triton;
- repository tests under root `tests/`, examples under `examples/`, tools under
  `tools/`, and ASV benchmarks under `benchmarks/`;
- `pyproject.toml` as package/tool configuration and `noxfile.py` as the
  maintained gate surface;
- one installed smoke test at `python -m jittor.selftest`;
- durable design documents under semantic `docs/` directories;
- Python 3.7-3.13 packaging, with real 3.12/3.13 wheel gates and NumPy 2.x on 3.13.

Source ownership and future moves must follow
[source architecture](../../docs/architecture/source-architecture.md). Runtime compiler
resources under `python/jittor/src/` and `python/jittor/extern/` remain physical-path contracts.

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
limitations. See [native complex dtype](../../docs/architecture/complex-dtype.md).

### Tests and performance

The test architecture uses independent forward references, numerical gradients,
and CPU-to-accelerator parity. See [test system](../../docs/testing/test-system.md).
The complete CPU repository gate is
[`tools/run_test_suite.py`](../../tools/run_test_suite.py), which owns separate
native and Torch-mode processes, JIT caches, temporary directories, and process
mode variables. Current AArch64 verification passes native `768 passed, 738 skipped`, Torch `1591 passed, 536 skipped`, and clean structure `232 passed, 2 skipped`; see the [current CPU suite](../results/2026-09-01-current-cpu-suite.md) and the [ARM CPU stability report](../results/2026-08-30-arm-cpu-suite-stability.md).
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
Qwen3-0.6B FP32 eager forward/loss/backward also passes zero-fallback at `1.07x-1.12x` native `torch_npu`. Transformers 5.5.3 BF16 completes forward, backward, and AdamW without CPU fallback; explicit fused AdamW matches CANN/PyTorch for two fixed-gradient steps. BF16 embedding/RMSNorm/RoPE training kernels pass independent real-NPU references. After correcting Python-scalar promotion, RMSNorm rounding order, and BF16 SiLU, all 29 hidden states and logits match native `torch_npu` elementwise for the maintained one-step input. The exact full step is currently `1.070x` native; direct CANN RoPE reaches `0.988x` but is rejected because its logits and gradient trajectory differ. Cross-framework long training parity and the exact-path performance gate remain open. See the [training report](../results/transformers/2026-08-30-qwen3-ascend-training.md).
See the [Ascend guide](../../docs/guides/ascend-910b.md), [validation report](../results/2026-08-28-ascend-910b-validation.md), [arg-reduce](../results/2026-08-30-npu-arg-reduce-backward.md)/[product](../results/2026-08-30-npu-product-reduction.md) follow-ups, [Qwen3 inference report](../results/transformers/2026-08-28-qwen3-ascend-performance.md), complete [CPU](../results/2026-08-22-complete-cpu-test-suite.md)/[CUDA](../results/2026-08-22-cuda-test-suite.md) reports, and the [parallel follow-up](../results/2026-08-22-cuda-parallel-range-network-oracle.md).
The current fail-closed optional CUDA base gate passes 16 TorchMetrics,
MMCV/MMEngine, PEFT, TensorDict, and FlashAttention-adapter tests from one
retained cache; TorchMetrics is split by domain so cold compilation does not
consume one monolithic test timeout. See the
[optional CUDA report](../results/2026-08-24-optional-compat-cuda-gate.md).
Compact ResNet18, ViT, GPT-2, and diffusion UNet now also pass three-step SGD
loss, complete trainable-parameter, and shared-buffer trajectories on CPU and
real CUDA. A bilingual native-Jittor ResNet tutorial executes the same three-step
training/state-restore workflow in the maintained offline CPU notebook gate. See the
[common-network trajectory report](../results/2026-08-26-common-network-training-trajectories.md).
The same report records the real-scale follow-up: UNet is accepted at `0.79x`,
ConvNet improved to `1.08x`, and ViT remains open at about `1.33x` because its
dominant CUDA GEMMs lag the PyTorch reference.
Performance work uses isolated caches, synchronization, and exact commit labels.
The ecosystem harness verifies twelve Transformers/Diffusers/PEFT/ms-swift/MMCV/MMEngine CPU/CUDA cases; its NPU scope verifies Diffusers UNet2D, MMCV/MMEngine, and ms-swift LoRA Llama forward and every gradient against `torch_npu` with zero CPU paths.
Diffusers correctness and maintained float32 performance are accepted at `0.964x` native `torch_npu`; the tiny OpenMMLab NPU cases now pass at `0.927x/0.796x`. See the [Diffusers](../results/2026-08-30-diffusers-ascend-parity-performance.md) and [OpenMMLab](../results/2026-08-30-mmcv-mmengine-ascend-parity.md) reports. The tiny ms-swift LoRA case uses fused float32 causal SDPA training and passes at `0.969x`; see the [ms-swift Ascend report](../results/2026-08-31-ms-swift-ascend-parity-performance.md).
Other ecosystem cases and the following verl results remain CPU/CUDA evidence, not NPU claims. Current verl core
algorithm/FSDP2 gates also pass on CPU/CUDA. The maintained framework FSDP2
gate additionally passes single-node four-rank NCCL sharding, collectives, and
sharded SGD on real CUDA; tiny Qwen3 end-to-end verl PPO also passes native
single-node four-rank FSDP2, while multi-node and native 0.6B remain open.
See the [verl weight-transfer and PPO report](../results/2026-08-24-verl-weight-transfer.md).
The external NPU vLLM adapter on current HEAD passes public `vllm.LLM.generate` for Qwen3-0.6B with exact four-token parity, zero CPU fallback, and no loaded `torch_npu`/`vllm_ascend`; only single-request, short-context, unquantized TP=1 correctness is accepted. Preserving BF16 parameters, CANN SwiGLU, a transpose-free single-token RoPE path, direct position-table lookup, and CANN multi-output split reduce its maintained warm-request median from about `0.615s` to `0.41749s`, but it remains slower than the `0.364996s` native baseline. See the [vLLM Ascend report](../results/2026-08-31-vllm-ascend-jittor-bootstrap.md).
Qwen3-0.6B vLLM real-CUDA inference
now runs about 20.5% faster than its real-PyTorch reference on the maintained
4-token protocol; TRELLIS.2 improved from about 1.20x to 1.093x slower, so its
performance gate remains open. CUDA masked SDPA now reuses the safe softmax
kernel instead of building redundant row-valid graphs. CUDA training LayerNorm
and standard RMSNorm now use fused forward/backward capabilities. The default
math-attention path remains `1.13x/1.22x` slower for GPT-2/Llama; with explicitly
configured native FlashAttention and float32-to-fp16 cast, GPT-2 reaches
`0.90-0.94x`, while Llama retains a conservative `3-4%` gap. See
[benchmarking](../../docs/performance/benchmarking.md) and the
[ecosystem parity/performance report](../results/2026-08-23-ecosystem-parity-performance.md),
the [verl/vLLM/TRELLIS current-baseline report](../results/2026-08-23-verl-vllm-trellis-current-baseline.md),
the [CUDA masked SDPA report](../results/2026-08-23-cuda-masked-sdpa.md), and the
[Transformer normalization follow-up](../results/2026-08-26-transformers-training-normalization.md).

### Agent-operable optimization

The proposal for structured compiler observation and bounded optimization is
research only; no autonomous mutation path is implemented. See
[agent-operable framework optimization](../../docs/research/agentic-optimization.md).

## Before running work

1. Read [collaboration rules](collaboration.md).
2. Configure a portable, isolated run from [environment](environment.md).
3. Search the [active known-issues ledger](known-issues.md) and
   [`agent/results/`](../results/README.md) for existing evidence.
4. Confirm the branch, exact commit, dirty state, and target backend.
5. Run the smallest reproduction before editing.

## Validation map

| Change | Minimum maintained evidence |
| --- | --- |
| Repository/docs only | layout checker, structure tests, relative-link check |
| Python API/refactor | focused tests, public identity/import tests, structure gate |
| Core operator/autograd | independent forward reference, gradcheck, CPU regression |
| Accelerator behavior | real-device execution and CPU/device parity |
| Packaging/runtime resource | sdist/wheel audits and installed selftest |
| Performance claim | correctness gate plus reproducible ASV comparison |
| Distributed behavior | focused process test and the relevant MPI/FSDP2 gate |

Use [`noxfile.py`](../../noxfile.py) for the maintained `lint`, `format`,
`typing`, `structure`, `py37`, `cpu`, `cuda`, `npu`, `rocm`, `mpi`, and
`benchmark` sessions.

## Open issues

The canonical list is [known-issues.md](known-issues.md). Highest-risk active
items include pending ROCm verification of corrected negative integer floor
division, the parallel compiler crash hypothesis, NPU reduction and FFT gaps,
and remaining NPU dtype/ROCm verification of corrected floating NaN comparisons.

Do not add a second bug narrative here. Add or update the ledger entry with an
owner, executable evidence, workaround, and exit condition.

## Recording progress

- Update this index only when its current-state summary or links change.
- Put durable architectural decisions under `docs/`.
- Put compact, reproducible verification and performance conclusions in a dated
  `agent/results/YYYY-MM-DD-topic.md` report.
- Keep raw logs, generated source, caches, wheels, profiles, and large benchmark
  data under `$JITTOR_LAB_ROOT/_state/`.
- A report names the exact commit, environment, commands, results, limitations,
  and any unversioned artifact hashes.

The Git history is the completed-work ledger. Do not rebuild a chronological
commit diary in this file.
