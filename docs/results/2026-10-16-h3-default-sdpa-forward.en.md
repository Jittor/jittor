# H3 default BF16 SDPA forward precision

- Status: forward improvement verified; public primitive gradient and full-model AD gaps remain open. No whole-model L2/L3 or CUDA breadth admission.
- Date: 2026-10-16
- Measured source: `67969d159e4f9033bac2ffc864a66a1b2a6a41c1` (clean)
- Preserved attribution baseline: `57b5f57ae220d1d8af1e6880ffcde6e362969110`
- Owner: native attention, CUDA attention and Torch frontend maintainers
- Review when: attention selection, cuDNN/frontend/runtime, the pinned H3 recipe, or the genuine Torch default provider changes.

## Scope and unchanged budget

This measures the actual upstream `MiniMaxH3DiTModel`: one main block, one refiner,
hidden dimension 32, 33,268 parameter scalars, 36 parameters and 37 initial state
entries. Omni is pinned to `446c2b5474dabffccf9a0034836eeab77c62b4b8` and vLLM to
`98dff2a81d747d1dba01a47f939f48c3526d4206`. Initial weights use seed 3; inputs use
seeds 7 and 19. The six outputs are video, audio and loss for each input seed.

The original forward metric remains
`max(abs(candidate-reference)) / max(1e-8, max(abs(reference))) <= 0.002` for
each array. Absolute error or `allclose(atol=.002, rtol=.002)` is a different
criterion. No oracle attention override, smaller model or changed input is used
for the formal comparison.

## Attribution and implementation

On source57, all 37 initialized states match the genuine runtime byte for byte,
but four of the six outputs fail the original scaled budget. The scaled errors
are `.001200732/.005837109/.017018842` for seed 7 and
`.001815902/.004493984/.002656590` for seed 19, in video/audio/loss order.

A real operator observer preserves the outputs. Genuine default Torch uses
Flash SDPA on BF16 `[1,4,2,8]` refiner and `[1,4,6,8]` main-attention tensors.
The first refiner Q/K/V inputs agree exactly, but its protected FP32-opmath
output differs by a BF16 ULP; subsequent main-attention inputs already differ.
Independent CUDA primitives reproduce genuine fused forward rounding using
unnormalized BF16 probability products. Genuine MATH and normalized BF16
probabilities produce different outputs. Forcing MATH was diagnostic only.

The production change reuses the existing native cuDNN provider. A strict
boolean native `allow_reduced_precision` option defaults to false, preserving
the native public `.002` forward / `.01` gradient opmath policy. The Torch
frontend requests this option; it adds no attention mathematics. New admission
is restricted to unmasked, noncausal, dropout-free BF16 rank-four tensors with
matching batch/head counts, identical K/V shapes, head dimension 8/16/24,
query length 1..64 and key/value length 2..64. Mask, causal, dropout, GQA,
longer sequences and single-key calls retain their prior protected route.
A single key has exactly zero analytical Q/K gradients; no cancellation floor
is introduced to claim fused parity there.

## Clean-source formal results

On the clean measured source, all six default-attention outputs are **byte
equal** between candidate and genuine Torch, with scaled error zero. All 37
initial state entries also match the original source57 reference. The genuine
six-array NPZ hash remains
`e11269e381af6b183917886a9b71be847b957ba9bccd3081c4b3eb1365005bb9`;
its initial-state NPZ remains
`5e0e32c5633363e79de48076c6faf59df6a2e691c69a768c2b458a5e8c7d077e`.

Separate observed runs preserve each side's six unobserved outputs. All 16
internal Q/K/V/output array pairs agree byte for byte. Genuine executes four
Flash forward calls; candidate executes four actual native cuDNN forward calls,
using physical sequence-major layout 7 and BF16 operands. The forward probe
records 43 actual CUDA pointers on each side; candidate fallback delta is zero.

Full eval/train AD auditing remains incomplete on **both** runtimes. Each
worker explicitly requests gradients for all 36 parameters and four floating
inputs, writes its actual arrays, and then exits nonzero under the strict
completeness check. For each seed and model mode, only four terminal-head
parameter gradients exist; 32 parameter gradients and all four input gradients
are missing. Each worker writes 14 arrays (six forwards and eight available
head gradients across both seeds). Missing gradients are not replaced by zeros,
and head-only gradients are not whole-model training evidence.

The CPU probes also remain negative: upstream default raises `No CUDA GPUs
available`; candidate raises `Fallback Gloo backend is not available`. Both
write complete installation/platform packets before the original failure.
These do not provide a CPU numerical counterpart or CPU `jtrun` coverage.

## Public gradient boundary

The new provider tests use the existing cuDNN BF16 `.02` contract. **This does
not establish the Torch public default's original `.01` gradient contract.**
A separate four-fixture before/after CUDA comparison retains the original
`.01` metric for every dQ/dK/dV array:

| Fixture | Old protected-math dQ | New cuDNN dQ | New dK | New dV |
| --- | ---: | ---: | ---: | ---: |
| seed7 refiner | .025103401 | .007352941 | .006849315 | 0 |
| seed7 main | .004629630 | .003086420 | .005681818 | 0 |
| seed19 refiner | .324621212 | **.018181818** | .001470588 | 0 |
| seed19 main | .004360465 | .005813953 | .004739336 | 0 |

Nine of twelve old gradient arrays and eleven of twelve new arrays pass `.01`;
there is no old-pass-to-new-fail gradient array. Seed19 refiner dQ remains open.
All four new primitive outputs are byte equal to default Flash. This table is
forward/primitive evidence, not a waiver of the public gradient gap or the H3
missing-gradient ledger. A subsequent backward investigation uses its own
source and artifacts; it must not relabel this measured source.

## Platform and source proof

The card is an RTX 4090 D, SM89, driver 580.178.04. Genuine is independent binary
Torch **2.13.0+cu130**, binary SHA256
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`.
Live mappings report genuine CUDART 13000 / cuDNN 92000, and candidate CUDART
12020 / cuDNN 92400. Candidate uses NVCC 12.2.140, GCC 11.5 and the actual
NVIDIA cuDNN frontend 1.30.0 wheel. The available compiler in the genuine
platform packet is also NVCC 12.2; it does not identify Torch's build toolkit.
The two actual library stacks are recorded separately, not called one toolkit.

Both runtimes independently validate complete installed Omni/vLLM payloads,
METADATA, RECORD and original wheel hashes. The Omni wheel is a **local
source-built** artifact, not a PyPI release:
`c6a43c816c95f99f751dd9aa18c36ff94461d508da002a2a328c543e14b2c0de`.
The fixed artifact guard and source closure remain active. Candidate's 13
source-bound deployed facade files and compatibility metadata are explicitly
identified as compatibility implementations, not genuine Torch/media/Flash
packages. Frontend wheel hash is
`98d3b5e3ef9c951a69ca36978f7b421e744895a810167577d1f57f3926a0202f`,
with all 897 installed payload members independently rehashed.

## Regression and preserved negative evidence

The unchanged native CUDA opmath/MemoryEfficient/cuDNN suite passes 41 tests;
new CUDA policy tests pass 5, nearby compatibility tests pass 7 and FP32 provider
tests pass 3, all without skips. New CPU opmath tests pass 10; nearby CPU
MemoryEfficient tests pass 9 with two CUDA-only skips. The maintained CPU core
gate passes 322, with 42 skips and one existing xfail. Final structural checking
passes 1,698 with four skips. Ruff, layout and diff checks pass. These unit runs
record their then-current base SHA plus exact seven-file WIP hashes, matching
the final production bytes; only the formal H3 workers claim the clean measured
SHA.

All failed attempts remain separate: inherited CUDA backend in the first CPU
unit invocation; two initial route-observation assertions; an editable core/
compat namespace installation failure; and the first genuine CUDA platform
probe lacking an actual compiler on PATH. Explicit environment correction,
compat-mode core installation and real provider-call observation resolve these
without changing model math or numerical budgets. The loader token changes
from `(1,0,0)` to `(1,1,2)` during actual discovery, and cache hits do not republish
`LAST_FUSED_KERNEL`; tests verify real `_forward` calls rather than manufacture
that marker. The first seal included its still-open control log; the final v2
seal rehashes completed artifacts and excludes its own live log.

## Reproduction and sealed evidence

All paths below are **unversioned** under `$JITTOR_LAB_ROOT`:

- State: `_state/omni-forward-fix/20261016/`.
- Immutable measured worktree: `worktrees/omni-forward-fix`, clean source679.
- Formal drivers: `run-formal-v2.py`, `run-remaining-formal-v1.py`; each case has
  exact argv/environment, pre-execution source hashes, terminal status and log.
- `formal-forward-audit-v1.json`: independent full installed-source and six-array
  recomputation; `formal-gradient-ledger-v1.json`: eight runtime/mode/seed rows.
- `formal-observer-audit-v1.json`: real providers, physical operands, live mapped
  libraries, 12 unchanged final outputs and 16 internal byte-equal pairs.
- `public-gradient-budget-audit-v1.json`: independent recomputation of all 32
  before/after comparisons from 48 saved arrays, retaining `.002/.01` budgets.
- Final `evidence-manifest-v2.json`: 9,180 artifacts / 4,476,538,455 bytes and
  2,907 current Git blobs, including symlink target text. SHA256:
  `816360ba7f1975b4b7620c698e6e2f6219e1cc7c0c8dfcba331d25f09362d365`.

To independently recompute forward evidence against the original measured tree,
use a host interpreter with NumPy and write a new external output:

```bash
python "$JITTOR_LAB_ROOT/_state/omni-forward-fix/20261016/audit-formal-v1.py" \
  --root "$JITTOR_LAB_ROOT/worktrees/omni-forward-fix" \
  --oracle "$JITTOR_LAB_ROOT/_state/omni-forward-fix/20261016/formal-genuine-cuda-v2" \
  --candidate "$JITTOR_LAB_ROOT/_state/omni-forward-fix/20261016/formal-candidate-cuda-v2" \
  --output "$JITTOR_LAB_ROOT/_state/<new-independent-audit>.json"
```

The original source57 attribution, setup failures, source679 arrays and missing
AD ledger remain distinct. This report does not claim full H3 training,
performance, engine/pipeline support, CPU support or CUDA breadth admission.
