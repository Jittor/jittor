# Jittor Project Context

- Status: Current index, not a history log
- Last reviewed: 2026-09-12
- Baseline reviewed: `integrate/cgq-transformers-2.0-refactor@54cd55cc`, with
  `origin/2.0-refactor@cb823778` merged in `c2bdd467` and retaining the validated
  compatibility work from `origin/cgq_transformers@5bf4d374`
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
defined in [Torch compatibility principles](../../refactor-wip/architecture/torch-compatibility-principles.md).

Operational rules:

- verify before fixing;
- prefer a clear unsupported error over a silent wrong result;
- exercise every advertised backend rather than relying on CPU fallback;
- keep one canonical implementation and explicit compatibility aliases;
- isolate caches and record reproducible evidence.

## Current repository state

The accepted repository shape and packaging boundaries are documented in
[Jittor repository layout](../../docs/development/repository-layout.md). The
current tree has:

- normal packages for `jittor.nn`, `jittor.misc`, `jittor.pool`, and
  `jittor.compat.torch`;
- compatibility domains (`jittor.compat`) owned by the top-level `compat/` independent `jittor-torch` project;
- repository tests under root `tests/`, examples under `examples/`, tools under
  `tools/`, and ASV benchmarks under `benchmarks/`;
- `pyproject.toml` as package/tool configuration and `noxfile.py` as the
  maintained gate surface;
- one installed smoke test at `python -m jittor.selftest`;
- durable design documents under semantic `docs/` directories;
- Python 3.7-3.13 packaging, with real 3.12/3.13 wheel gates and NumPy 2.x on 3.13.

Follow [source architecture](../../docs/development/source-architecture.md): CUDA resources
live under top-level `backends/`, including `comm/{mpi,nccl,hccl}`; shared core
lives in `src/` and is packaged as `jittor/src`. Package-local `src/extern` are removed.

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
The complete CPU repository gate is [`tools/run_test_suite.py`](../../tools/run_test_suite.py),
which owns separate native/Torch processes, JIT caches, temporary directories, and process-mode
variables. CUDA, ROCm, Ascend, optional downstream, full-training, and performance checks remain
separate gates; see the dated reports linked from `refactor-wip/results/`.
The current refactor integration retains the fixed Transformers 4.56.2 text matrix: 4 encoder,
10 decoder, and 3 encoder-decoder implementations are 17/17 PASS on fixed public checkpoints
with real A800 CUDA. The public artifacts were generated from the validated compatibility
commit `5bf4d374`; the later refactor delta through `cb823778` is confined to backend/structure
and benchmark changes, so the existing real-device HF CUDA evidence remains applicable. The evidence
includes tokenizer, forward, cache, generation, state compatibility, and encoder
Jittor-to-PyTorch round-trip; the GPT-NeoX top-p cutoff tie remains an explicit numerical
boundary. See the [current refactor 17-model L4 report](../../refactor-wip/results/transformers/2026-09-11-transformers-refactor-17-model-l4-cuda.md), the [historical strict L4 report](../../refactor-wip/results/transformers/2026-09-06-transformers-text-core-l4-cuda.md), and the [earlier matrix](../../refactor-wip/results/transformers/2026-09-03-transformers-text-core-matrix-cuda.md).
The same worktree retains the FSDP2 metadata-lifetime fix and the bounded six-A800 BF16 Llama 3.1 70B short-SFT evidence. The current 17-model report is FP32 correctness evidence, not a claim of L5 performance, low precision, long-context training, or broad downstream coverage. See the [FSDP2 memory-lifetime report](../../refactor-wip/results/transformers/2026-09-07-fsdp2-memory-lifetime-fix-cuda.md), [representative Llama 70B SFT report](../../refactor-wip/results/transformers/2026-09-07-transformers-llama31-70b-sft-cuda.md), and [formal No Robots resource report](../../refactor-wip/results/transformers/2026-09-07-transformers-llama31-no-robots-sft-cuda.md).
Current Ascend/Qwen3, CPU, CUDA, and optional ecosystem conclusions are maintained in their dated
reports under `refactor-wip/results/`; they are not duplicated in this index.
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
[ecosystem parity/performance report](../../refactor-wip/results/2026-08-23-ecosystem-parity-performance.md),
the [verl/vLLM/TRELLIS current-baseline report](../../refactor-wip/results/2026-08-23-verl-vllm-trellis-current-baseline.md),
the [CUDA masked SDPA report](../../refactor-wip/results/2026-08-23-cuda-masked-sdpa.md), and the
[Transformer normalization follow-up](../../refactor-wip/results/2026-08-26-transformers-training-normalization.md).

### Agent-operable optimization

The proposal for structured compiler observation and bounded optimization is
research only; no autonomous mutation path is implemented. See
[agent-operable framework optimization](../../refactor-wip/research/agentic-optimization.md).

## Before running work

1. Read [collaboration rules](collaboration.md).
2. Configure a portable, isolated run from [environment](environment.md).
3. Search the [active known-issues ledger](known-issues.md) and
   [`refactor-wip/results/`](../../refactor-wip/results/README.md) for existing evidence.
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
  `refactor-wip/results/YYYY-MM-DD-topic.md` report during the refactor.
- Keep raw logs, generated source, caches, wheels, profiles, and large benchmark
  data under `$JITTOR_LAB_ROOT/_state/`.
- A report names the exact commit, environment, commands, results, limitations, and any unversioned artifact hashes.

The Git history is the completed-work ledger. Do not rebuild a chronological
commit diary in this file.
