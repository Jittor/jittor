# Jittor Project Context

- Status: Current index, not a history log
- Last reviewed: 2026-08-26
- Baseline reviewed: `b37098f7`
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
- durable design documents under semantic `docs/` directories.

Source ownership and future moves must follow
[source architecture](../../docs/architecture/source-architecture.md). Runtime
compiler resources under `python/jittor/src/` and `python/jittor/extern/` remain
physical-path contracts.

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
mode variables. Its 2026-08-22 baseline passes both sessions. The maintained
CUDA gate also passes on a real RTX 4090 and covers the complete CUDA backend
directory, dtype coverage, CPU/CUDA device parity, TF32 controls, and strict
OpInfo CUDA references. The maintained CPU gate also passes with a fail-closed
independent binary PyTorch oracle, and compact ResNet18, ViT, GPT-2, and
diffusion UNet forward/backward parity passes on CPU and CUDA. NPU, ROCm,
optional downstream dependencies, full training, and performance remain
separate gates. See the
[complete CPU suite report](../results/2026-08-22-complete-cpu-test-suite.md) and
[complete CUDA suite report](../results/2026-08-22-cuda-test-suite.md), plus the
[parallel-range and network-oracle follow-up](../results/2026-08-22-cuda-parallel-range-network-oracle.md).
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
Performance work uses ASV, isolated caches, synchronized measurements, and exact
commit labels. The same-version ecosystem harness now verifies twelve
Transformers/Diffusers/PEFT/ms-swift/MMCV/MMEngine CPU and CUDA cases, while
real-scale training performance remains incomplete. Current verl core
algorithm/FSDP2 gates also pass on CPU/CUDA. Qwen3-0.6B vLLM real-CUDA inference
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
items include pending NPU/ROCm verification of the corrected negative integer
floor division, the parallel compiler crash hypothesis, NPU reduction atomic gaps,
and NPU/ROCm verification of the corrected floating NaN comparisons.

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
