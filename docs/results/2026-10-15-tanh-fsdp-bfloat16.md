# Saved-output tanh gradients and native FSDP BF16 verification

- Status: validated on CPU and two real CUDA/NCCL ranks at source
  `cc45f12a216af639faf1e7bd3fc2ff916ce16653`.
- Owner: Jittor core maintainers.
- Remote baseline: `344dba6cd259259050c64b41541fdff281788b22`, integrated as
  `1d5fc1480401cf6e22220ff803f058f4013a0231` before final verification.
- Recheck when: unary differentiation, Linear accumulation, mixed precision,
  collective reduction, the compiler or the independent Torch reference changes.

The native tanh backward now differentiates its saved output. This removes the
remaining BF16 gradient discrepancy in the fixed native FSDP fixture and keeps
first and second derivatives finite at saturated FP16 inputs. The production
change is confined to `src/ops/unary_op.cc`; the gradient engine, AMP policy and
optimizer mathematics are unchanged.

## Finding and repair

The previous backward recomputed `cosh(x)` and divided by its square. For a
rounded BF16 or FP16 tanh output, this differs from the derivative implemented
by independent Torch. At FP16 inputs of magnitude 20, the recomputed expression
also produced two NaNs in the thirteen-input second-derivative probe on both
CPU and CUDA.

The controlled layer probe used the original oracle weights at steps zero and
four and both ranks' fixed inputs. Linear outputs, tanh outputs, predictions,
and the gradients entering tanh matched bitwise. The first discrepancy was the
gradient leaving tanh for the first Linear. A separate AMP level-three probe
left all 340 BF16 trajectory arrays unchanged, ruling out that proposed repair.

The replacement builds the ordinary differentiable graph
`dout * (1 - out * out)` through the saved output. CPU FP16/BF16 uses FP32
opmath and rounds the result once; CUDA preserves the low-precision scalar
rounding boundaries observed in the actual Torch CUDA kernel. These boundaries
are observably different: the frozen BF16 grid case at input -1 produces
`-0.040283203125` on Torch CPU and `-0.0400390625` on Torch CUDA. The regression
keeps separate device gold values. FP32 and FP64 retain their own dtypes.

[The new regression](../../tests/ops/test_tanh_backward_precision.py) checks
FP32, FP16 and BF16 first and second derivatives at AMP levels zero and three,
including rounded saturation, with independent binary Torch values. A further
FP64 test verifies the analytic second derivative. Low-precision grid results
match the corresponding real device oracle bitwise; every checked value must
have the exact shape and be finite, with backend fallback forbidden.

## Independent distributed calibration and fixed protocol

The independent runtime was binary Torch `2.11.0+cu128`, git
`70d99e998b4955e0049d13a98d77ae1b14db1f45`. Its real FSDP2 used two CUDA ranks,
BF16 parameters, FP32 reduce dtype, FP32 outputs and actual DTensor collection.
All **396** arrays equal the dense independent Torch oracle bitwise, including
outputs, losses, input/parameter gradients, parameters and optimizer state.
Thus the earlier residual was an implementation difference for this fixture.

The reusable
[calibration tool](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/native_fsdp_torch_calibration.py)
was committed separately as `eca1d57cb`. It imports the same NumPy-only fixture
and real Torch FSDP2; it introduces no second implementation of sharding or a
new numerical budget. Run it with an independent binary Torch interpreter:

```bash
<binary-torch-python> -m torch.distributed.run --standalone --nproc_per_node=2 \
  agent/skills/jittor-distributed-verification/native_fsdp_torch_calibration.py \
  --source-root "$PWD" \
  --output-dir "$JITTOR_LAB_ROOT/_state/native-fsdp/calibration"
```

The native fixture and thresholds have the same bytes as the original
[accepted source](2026-10-15-native-fsdp.md). Its fixture SHA-256 remains
`5b675b8c250fde9bbddc42d9ffc1f7c5c68caa342bdd80631a196eee75bcdd7a`.
It uses Linear(5, 7), tanh, Dropout and Linear(7, 3), four parameters and 66
logical elements. Each rank owns 35 padded elements. Five steps exercise SGD
with momentum and AdamW, each with two parameter groups, in FP32 and BF16.

Elementwise budgets remain FP32 `atol=2e-6, rtol=2e-5` and BF16
`atol=4e-4, rtol=3e-2`. The per-parameter update L2 budgets remain
`2e-6 + 2e-5 * reference_norm` and `1e-6 + 0.03 * reference_norm`, respectively.
Each reference update must exceed ten absolute floors. Every actual parameter
and step rejects an injected zero gradient and zero update.

## Same-input results across the three preserved sources

The following row is AdamW, rank zero, step five, `fc1.bias` gradient. Its
reference L2 norm stays **0.001057407702319324** throughout.

| Source | Gradient error L2 | Relative L2 error | Exact BF16 gradient arrays |
| --- | --- | --- | --- |
| Original `ee24fac2b` | 0.00011868552246596664 | 11.2241969% | 11 / 80 |
| Shared Linear repair `251fc6bc7` | 0.00007667446334380656 | 7.2511731% | 35 / 80 |
| Saved-output tanh `cc45f12a2` | 0 | 0% | 80 / 80 |

The original elementwise absolute tolerance admitted the small gradient's
absolute error. Its report and raw arrays retain the original 11.224% finding;
the intermediate 7.251% result is also preserved. The final result proves all
80 measured BF16 parameter-gradient arrays bitwise equal to both independent
Torch references for this fixed fixture.

The complete parameter trajectories retain small FP32 update differences:
maximum BF16 AdamW parameter relative L2 error is `1.0407533e-7`. Maximum BF16
per-step update relative L2 error is `5.3062839e-7`. The smallest measured BF16
update norm is `0.00014121785352472216`, over 141 absolute floors; the absolute
term supplies at most 19.1% of the update error budget. Both precisions pass all
**160 zero-gradient and 160 zero-update negative checks**. Per-parameter,
per-step norms, errors, floors and budgets remain in the raw audit.

## Acceptance and scope

| Check | Result |
| --- | --- |
| CPU unary, FP64, native FSDP and checkpoint neighbors | 65 passed, 20 accelerator skips |
| CUDA unary and FP64 neighbors | 37 passed, zero skipped |
| Existing CUDA Linear second-gradient node | 1 passed, 5 other nodes deselected |
| CPU core, native and Torch modes | 322 passed, 42 skipped, 1 expected failure |
| Complete structure gate | 1472 passed, 4 skipped, zero failures |
| Native CUDA parity | 7 passed per rank, zero skipped |
| Save after three steps and live continuation | 7 passed per rank, zero skipped |
| Fresh launcher restore and two further steps | 7 passed per rank, zero skipped |
| Layout, Ruff and whitespace | Passed |

The unit/core/structure gates used exactly the subsequently committed runtime
bytes. All three final CUDA phases ran with clean, fixed `cc45f12a2` HEAD.
An initial optional SciPy dependency failure and a missing explicit pytest
plugin configuration were corrected externally before the successful reruns;
the rejected attempts remain in raw logs and do not count as passes. Shared
dependency installations were unchanged.

The final 24 candidate NPZ files contain **1,968 arrays**. All **368 arrays**
from the fresh launcher equal uninterrupted native training exactly, including
Dropout outputs, gradients, parameters, moments and each rank's Python, NumPy,
native CPU and CUDA RNG streams. Dropout probability is 0.25 in save/resume;
independent Torch parity uses probability zero. The two ranks' RNG streams are
checked independently and differ from one another.

Native runs used CUDA 12.2.140, NCCL 2.18.3 and strict CUDA kernel math. Both
native precision tiers are `highest`; independent Torch disables TF32 and uses
`highest` matmul precision. Instrumentation verifies actual BF16 GEMM operands
and the FP32 accumulator's CUDA allocation before pointer queries, BF16
all-gathers, FP32 reduce-scatter, and FP32 masters/moments. The three phases
execute **384 all-gather and 384 reduce-scatter local calls** across both ranks.
Every phase has zero fallback and imports neither Torch nor its compatibility
frontend. Rank-local checkpoint annotations cover only genuinely sharded state
and RNG; replicated gain/control drift is rejected.

This establishes the fixed two-rank CUDA training and fresh-restore scope above.
Other backends, changing world size, arbitrary model families, peak memory and
throughput are outside this measurement. Other GPU processes were present, so
these results support correctness conclusions only.

## Main-tree integration verification

The runtime repair was integrated byte for byte on `3ba5bb771`. CPU core passed
322 tests with 42 declared skips and one existing xfail. Native CPU tanh/TP
owners passed 17 tests with four accelerator skips; the seven tanh tests also
passed after explicitly scoping their exact gold values to strict kernel math.
Actual CUDA unary, low-precision and FP64 neighbors passed all 37 tests without
skips. The first default-math attempt exposed two FP16 exact-gold mismatches
and four missing real SciPy `erfinv` failures; those logs remain unversioned.
The test now declares strict math, and a private existing SciPy installation
supplies the neighbor's real dependency. Production arithmetic was unchanged
by this test/environment correction. Other unary neighbors ran in default math.

Actual CUDA/NCCL TP revalidation passed 13 tests per rank without skips, after
serial warmup of independent rank caches. This includes FP32/BF16 training,
optimizer state, full gradients and fresh native checkpoint restore. All 431
recorded TP source hashes remained unchanged throughout the integration run.
These are integration checks; historical FSDP measurements retain `cc45f12a2`.

The root audit independently rehashed all 801 follow-up files, checked all
1,968 candidate arrays for finite values, compared all 80 padded BF16 gradient
shards with dense independent Torch, and compared all 368 fresh-resume arrays
bitwise. Raw integration state stays under
`$JITTOR_LAB_ROOT/_state/tanh-integration/` and
`$JITTOR_LAB_ROOT/_state/native-tp-tanh-revalidation/`.

## Preserved evidence

Raw data stays under
`$JITTOR_LAB_ROOT/_state/native-fsdp-linear-recheck/20261015/`; final measurements
are in `tanh-final/`, separate from the Linear-only arrays and all earlier
probes. `SOURCE.json` binds 34 source files. `STAGES_COMPARISON.json` retains all
80 gradient rows per historical source. The final audit independently verifies
324 artifacts, 21 completed logs and five runners against
`tanh-final/artifacts/SHA256SUMS.json`, whose SHA-256 is
`2b405f496e679f131d20ab493cf83dc8e830a3c5b24a44f6ba593bb4417d1556`.

The original `ee24fac2b` seal remains
`9901d1ed7dd834dae4d11b4afdb501a37d2cdc3bc692019daad2d6a0c27c9a27`;
its 236 artifacts, 13 logs and five runners were independently rehashed and
all match. The intermediate Linear-only seal's 312 artifacts, five runners
and twelve other logs also match. Its own stdout log was still being appended
when that historical seal hashed it; that single log mismatch is recorded, and
both original files are retained. The final seal hashes only completed logs.

`FOLLOWUP_SHA256SUMS.json` records the historical sealing caveat and binds all
801 follow-up artifact/log/runner files to their complete bytes. Its SHA-256 is
`0589274e5982de851438c7134ecdd326ca2aa9ab7a184f66e4e95f38529b2195`.
The new audit does not overwrite the original tensor evidence or thresholds.
