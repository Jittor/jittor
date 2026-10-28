# H3 BF16 attention: preserving the original 1% gradient contract

Status: short-sequence attention primitives pass the original budget; the complete H3 forward passes, the complete AD still fails; no H3 L2/L3, CPU or CUDA breadth is granted.

Date: 2026-10-16. Owner: Torch/CUDA attention maintainers.
Baseline commit: `aa9e669f514432a3aae77840d1604458c1b9f3ff`, depending on the forward batch
`67969d159e4f9033bac2ffc864a66a1b2a6a41c1`. This batch first integrated remote
`bc4dc0bc01a3930d3fad22326da0f9b8c70d444f`, then, after freezing the verified final state, integrated
`00ceb26fd395707fb33649cc43709af045110a9a`; formal measurement uses only the clean `aa9e669f5`.

Review when: native attention/opmath, the CuDNN frontend, Torch SDPA routing, Function saved-value semantics,
Omni's private CUDA/Triton operations, installation provenance validation or the pinned upstream version changes.

## Problem and fix scope

The previous batch's CuDNN forward removed the H3 default Flash oracle's forward difference, but of the
12 input gradients across the four genuine attention fixtures only 11 satisfied the original scaled-relative `.01`: seed 19 refiner dQ was
`0.01818181818181818`. The old protected math path gave 9/12, with a maximum error of
`0.32462121212121214`. The old data and report keep their original identity and are not replaced by this batch's results.

Independent binary Torch forcing CuDNN on the same card reproduces the same dQ difference. The CuDNN frontend
graph actually compiled in this batch places the scale before the BF16 rounding of dS; in an external per-stage genuine CUDA comparison, the combination that first rounds dS to BF16 and
applies the scale after the FP32 Q/K GEMM reproduces this default Flash gradient reference byte for byte.
That comparison is attribution only; the complete H3 oracle always requests default attention and never switches to MATH or CuDNN.

The new branch keeps the genuine CuDNN forward and uses the shared native attention gradient composition: FP32 probabilities/dP,
row term from the rounded output, dS rounded to BF16, FP32 product for Q/K and then scaled, then BF16
gradients returned; dV uses the BF16-rounded probabilities. **This is a CuDNN forward plus native composite backward, not fused
CuDNN backward, and it grants no performance level.** The split applies only within the short BF16, mask-free,
non-causal, zero-dropout, non-GQA admission domain already bounded by the previous batch; no new shape admission is added.

The default native path keeps the original probability and gradient formulas and operation order; an independent AST folding comparison and actual CPU/CUDA regressions both verify
this. Non-BF16 paths and the rejection domain continue on their original route. Original inputs are saved through an independent differentiable holder, covering q/k/v
rebinding individually and sequence-major layouts.

## Genuine CUDA results

Platform is RTX 4090 D / sm89, driver 580.178.04. The reference is genuine Torch `2.13.0+cu130`,
with CUDART `13000` and cuDNN `92000` actually loaded; the candidate is the Jittor Torch facade, with CUDART `12020`,
cuDNN `92400` actually loaded, actual nvcc `12.2.140` / GCC `11.5`, frontend `1.30.0`.
The candidate's Torch API/media metadata is compatibility metadata, not a declaration that genuine Torch is installed.
The two sides' actual dependency stacks are bound separately; no identical CUDA/cuDNN distribution stack is claimed.

| Check | Measured result |
| --- | --- |
| Four original fixtures, seeds 7/19, sequences 2/6, head_dim 8 | New path: 4 outputs + 12 input gradients all byte equal; original `.002/.01` all pass |
| Original-path before/after | Old math 9/12 → new path 12/12 gradient arrays; no old pass becomes a new failure |
| 12 boundary groups, seeds 7/19, head_dim 8/16/24, query 1/64, key 64 | 48/48 arrays pass the original `.002/.01`; 46 byte equal |
| Boundary nonzero errors | seed19/d16/q64 dQ `0.002336448598130841`, dK `0.0005081300813008131`, both below the original `.01`; the rest byte equal |
| Actual provider | CuDNN `_forward` executes, the native shared gradient helper executes, CuDNN `_backward` does not execute; genuine BF16 CUDA residency, fallback delta 0 |

The `.02` of the existing NumPy boundary tests is still only a provider contract; it cannot replace the public `.01` comparisons above, nor
the complete H3 `.002`. The mask, causal, dropout, GQA, long sequence and single-key rejection domains keep their original routes and original budgets.

## Complete H3 on a clean commit

Pinned Omni `446c2b5474dabffccf9a0034836eeab77c62b4b8`, vLLM
`98dff2a81d747d1dba01a47f939f48c3526d4206`. The actual `MiniMaxH3DiTModel` keeps
1 trunk / 1 refiner / hidden32, 33268 scalars, 36 parameters / 37 initial states, initial
seed 3, input seeds 7/19, and the original default attention. The complete installation provenance guard and the pinned C6 source-built
Omni wheel are enabled as-is; that wheel is not called an official PyPI release.

The unobserved forwards on both CUDA sides each produce 6 arrays, all byte equal, and the original
`max(abs(diff))/max(1e-8,max(abs(reference))) <= .002` all pass. The 37 initial states are byte-identical to the original
reference. The independent and unobserved runs on each side agree on all 6 outputs / 37 states, and all 16 internal pairs of
Q/K/V/output agree. The actual 4 calls hit genuine Flash and genuine CuDNN layout 7 respectively;
22 mapped library/JIT files on the reference and 92 on the candidate were each re-hashed, and the complete model's 43 state/output
pointer records verify genuine CUDA, with candidate fallback count 0.

The 4 AD workers for eval/train on both sides each write 14 genuine arrays and then strictly return rc1. Each seed requests gradients for
36 parameters and 4 inputs, yet **32/36 parameter gradients and all 4 input gradients** are missing; only the 4 final output-head
parameters have gradients. The complete 8-row expected/observed ledger preserves that failure; partial head gradients or the
identical missing state cannot grant L2/L3. CPU genuine reports `No CUDA GPUs are available` at default model construction;
the candidate refuses at the genuine Gloo availability check. The CPU installation/platform packets on both sides are retained;
there is no CPU counterpart, and no CUDA breadth is counted.

## Regression and recheck

After the final private installation fix: core 322 passed / 42 skipped / 1 xfailed; CPU attention
19 passed / 2 CUDA-only skipped; the new genuine CUDA suite 8 passed / 0 skipped; structure +
compat structure + Omni/platform host 1701 passed / 4 skipped; Ruff, layout and diff checks passed.
The same production math file previously had another 41 CUDA-configured native adjacent tests (including explicit CPU HVP checks),
7 compatibility-layer and 3 FP32 CUDA owner tests passing. The default native `.002/.01`, FP16/FP32,
mask/GQA, rebinding and FP64 HVP contracts were all left unrelaxed.
The unit tests ran on the same file bytes frozen before the commit, and their actual HEAD/dirty state is stored in the original command packet;
the complete H3 and final production comparisons ran on clean `aa9e669f5`, and the old unit tests' commit identity is not rewritten.

The earlier 3 structural failures are listed separately and preserved: the new grad implementation lacked a reference registration, and 2 nested xdist subprocesses
did not explicitly load the plugin. After narrowly supplementing the genuine CUDA/NumPy test registration and fixing the external launcher, the original failing node passed 28, and
a subsequent full structure run passed. Another bare import-spec issue came from this task's private compat/adapters
editable installation; after the fix the bare import actually resolves to genuine Torch binary; it is recorded separately from the 3 failures above.
Two control-plane failures — a mistyped SHA argument and a CPU expected-error-text check — are also preserved; the actual workers were not rewritten.

The maintained regression uses this commit's `compat/tests/torch/test_sdpa_bf16_backend_precision.py`. First prepare, per the environment manual, an independent candidate prefix of
the current source, genuine CuDNN/frontend and a deployed facade, then run in an independent directory:

```bash
h3_unit="${JITTOR_LAB_ROOT:?}/_state/h3-sdpa-recheck/unit"
mkdir -p "$h3_unit/home" "$h3_unit/jittor-home" "$h3_unit/tmp"
env HOME="$h3_unit/home" JITTOR_HOME="$h3_unit/jittor-home" TMPDIR="$h3_unit/tmp" \
  CUDA_VISIBLE_DEVICES="${H3_CUDA_VISIBLE_DEVICE:?}" \
  JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda JITTOR_TEST_DEVICES=cuda \
  JT_BACKEND=cuda JITTOR_CUDA_WHEEL_STRICT=1 JT_CUDA_KERNEL_MATH=strict \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_PLUGINS=pytest_timeout,xdist.plugin \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="python:${H3_FACADE:?}:adapters" \
  "${H3_CANDIDATE_PYTHON:?}" -m pytest -q \
  compat/tests/torch/test_sdpa_bf16_backend_precision.py
```

The raw evidence is **unversioned**, located at `$JITTOR_LAB_ROOT/_state/omni-backward-precision/20261016/`.
`public-gradient-production-v1/`, `boundary-production-v1/`, `formal-forward-audit-v1.json`,
`formal-gradient-ledger-v1.json` and `formal-observer-audit-v1.json` respectively hold the genuine values above and the independent recomputations.
The complete source/wheel/METADATA/RECORD/payload verification binds 1607 framework files; the complete Git seal
binds 2912 mode/blob/content entries, including symlink target text.

Final `evidence-manifest-v1.json` SHA256:
`13c906638fe36a8e24884e2740241fdecf6595b56646133b527a5bd625f119d0`,
9352 artifact paths / 5,763,671,636 bytes. The seal does not include its own still-open stdout log.
The independent full re-hashing and the script are preserved in
`$JITTOR_LAB_ROOT/_state/omni-backward-seal-review/20261016/`; when rechecking, copy that script into a new external directory,
write the output to a new directory, and never overwrite the sealed result. The original `67969d` seal, source and raw data always keep their original identity.
