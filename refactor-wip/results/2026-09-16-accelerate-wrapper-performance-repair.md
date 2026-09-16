# Accelerate wrapper performance repair

- Status: verified within scoped FP32 CPU/CUDA optimizer, batching and compatibility-autocast validation
- Date: 2026-09-16
- Baseline commit: `feature/cgq_transformers@4d424432`
- Repair commit: `1048a6ec` for FP32 performance; introducing AMP followup commit for the scoped autocast repair
- Owner: Jittor core maintainers
- Review when: optimizer arithmetic, Tensor scalar promotion/placement, Accelerate wrapping, or TensorDataset batching changes

The objective is to bring Accelerate-wrapped Jittor training closer to the same
native Jittor workload. The comparison separates native Jittor, the independent
Torch frontend, frontend plus prepared model/optimizer, `Accelerator.backward`,
and the prepared DataLoader. PyTorch is an independent numerical oracle, not
the speed target for the wrapper overhead.

## Diagnosis and repair

The fixed FP32 MLP uses batch 16 and dimensions 1024/2048/1024, identical initial
weights and preallocated inputs, AdamW, highest FP32 matmul precision, and
disabled TF32. Cold JIT, numerical snapshots, and profiling are outside the
throughput window. Every optimizer update and state dependency is consumed;
each round ends with a device synchronization. GPU contention is recorded in
the unversioned raw results, so separate native runs are directional controls;
the same-process ABBA comparison establishes the repair's local contribution.

Four extra profiled Accelerate steps attribute 56 ms to the AdamW adapter and
54 ms to its canonical `adam_update`. The overloaded Tensor arithmetic reaches
340/216 `_promoting_binary` calls, 5792 installation-context lookups, and 3654
frontend precision-policy reads. This identifies repeated frontend promotion
and placement queries inside optimizer arithmetic. It does not establish that
each elementwise AdamW operation launches a separate CUDA kernel: the earlier
saved profile contains fused multi-operation optimizer graphs.

SGD independently shows the same hotspot: warmed full steps take
6.24--6.44 ms, with 4.06--4.19 ms spent submitting optimizer work. Four
extra profiled steps spend 43 ms in 16 canonical `sgd_update` calls,
55 ms in 228/152 promoting-binary calls, 45 ms in 4075 context lookups,
and 41 ms in 2674 precision-policy reads.

The repair keeps the canonical native Adam and SGD expressions and selects native
arithmetic only for homogeneous FP32 update tensors, identical placements,
Python numeric coefficients, and a matching scalar-factory backend/device.
Only concrete registered frontend Tensor/Parameter types are eligible;
native Vars and application subclasses retain their existing operator dispatch.
Only CPU/CUDA backends are eligible; all other backends retain existing dispatch.
Typed stopped rank-0 scalar constants preserve the frontend's rounding and
native strict FP32/AMP policy. Unsupported fast-path inputs retain their
existing arithmetic. Parameter gradient lifetime, per-parameter step counts,
decay, epsilon, and state serialization are unchanged by this dispatch.
The shared dispatcher lives in `optim/base.py`; adapters and FSDP continue
calling canonical update functions. The improvement therefore applies to the
public compatibility path, rather than an experiment-specific Accelerate patch.

The missing-gradient SGD regression also identifies an existing correctness
bug: compatibility `zero_grad(set_to_none=True)` removes an empty group's
gradient list, but native SGD indexed that list unconditionally. A frozen
baseline replay produces the same `KeyError: 'grads'`. Native SGD now treats
an absent list as empty, so groups with no gradients skip updates while other
groups train. Five-step tests cover a group with no gradient on its first step,
later missing gradients, resumed gradients, unchanged momentum buffers and
per-parameter counters. Nonzero dampening preserves the existing native
first-step policy; it is not presented as a new PyTorch parity claim.

The prepared DataLoader's extra host cost is default collation: five extra
profiled fetches spend 133 ms overall, 115 ms in collation, 111 ms in stack,
81 ms in concatenation, and 65/160 setitem calls. Moving an already resident
batch to its device is negligible here. The repair batches only exact
TensorDataset/default-collator combinations with concrete Tensor
or Parameter inputs. Contiguous inputs use the canonical native GetitemOp in
a matching frontend placement scope; noncontiguous inputs use public vector
indexing. Custom datasets, subclasses, collation and invalid indices retain
the original per-sample/default-collation path. Its directed tests cover
batch values, gradients and fallback guards. The final nine-case suite also
spies on native Getitem calls and requires default collation to be bypassed;
the guarded optimization is restricted to CPU/CUDA placements.
Final CPU/CUDA tests explicitly confirm native batch-gather eligibility,
so passing fallback behavior cannot be mistaken for fast-path coverage.

## Evidence

Same-process optimizer ABBA runs use 6 warmup steps followed by 24 measured
steps per round; the final loader ABBA uses 4 warmup and 20 measured steps.
Each round restores identical initial weights and inputs. A rounds
substitute the frozen baseline canonical update or batch fetch; B rounds use
the repaired implementation. Measurements below are milliseconds per step:

| Round | AdamW Full Loop | SGD Full Loop | Prepared Loader With Repaired AdamW |
| --- | --- | --- | --- |
| A | 9.223 | 6.292 | 11.630 |
| B | 4.859 | 3.842 | 8.304 |
| B | 4.850 | 3.832 | 8.264 |
| A | 9.208 | 6.451 | 11.505 |

The AdamW table uses guarded source
`bd431b740c74902c4c0064ac0366272d0cbe1180e984a93f163acba47c8667e9`.
The SGD table uses the final shared dispatcher and SGD source
`03f299384f084acaa88e47cf9a4717b5565431d795978e3c6333c34d711746ee`;
the subsequent absent-gradient fix changes only empty groups, which do not
occur in this performance workload. The loader table uses final source
`0c39609829bec692241a9c2d34f8fd2ae21a27fdcaf5960ad624fcb8fe2e2492`
and the final shared Adam dispatcher. Loader submission time falls from
6.41--6.48 to 0.48--0.50 ms per step; the session reports zero fallback.
Its candidate forward phase shows considerable GPU contention, so the total
loader-loop measurements should not be used to claim exact native parity.

The application-level comparison trains the same frozen ResNet-18 on all
50,000 real CIFAR-10 training samples, 391 batches of size 128 including the
tail, with SGD learning rate 0.1, momentum 0.9 and decay 5e-4. Inputs are
preloaded onto CUDA before measurement, the model is in training mode and
the full epoch consumes all updates. Warmup/JIT is measured separately,
there are no metrics or host snapshots inside the epoch, and synchronization
occurs at the epoch boundary.
An actual-iteration fairness check confirms direct and prepared loaders both
yield all 50,000 samples in 391 batches, with a tail of 80 and identical order.

| ResNet-18 Configuration | Full Epoch Seconds |
| --- | --- |
| Frozen baseline, Accelerate model/optimizer/DataLoader | 58.247 |
| Repaired, Accelerate model/optimizer/DataLoader | 37.445 |
| Repaired, direct frontend with the same DataLoader | 38.272 |

All three start from identical frozen weights (maximum initial error zero),
have the same first loss 2.4602913856506348, update all 62 parameters and
report zero backend fallback. The GPU has external shared load, so this
comparison finds no clear additional wrapper cost after repair; it does not
establish that wrapping is faster than direct execution. Host data/optimizer
phase values are submission/blocking wall time, not isolated CUDA kernel
durations. The native Jittor direction control completes the epoch in
36.271 s, using manual batches rather than a prepared DataLoader. This
also places the repaired wrapped run near native execution, but the strict
wrapper comparison is the repaired direct run with the same DataLoader.

The final small-MLP SGD control is 3.38--3.41 ms direct and
3.71--3.76 ms with full wrapping, leaving about 0.35 ms locally. Its four
extra diagnostic steps reduce canonical SGD cumulative time from 43 to 7 ms
and promoting-binary calls from 228/152 to 52/40. Remaining shared host costs
include 2331 context queries and 1698 precision-policy reads (both 17 ms
cumulative, overlapping), dtype getters and module dispatch. Profiling also
includes construction, cleanup, garbage collection and extra synchronization;
these are excluded from throughput and must not be interpreted as steady
training bottlenecks. This bounded repair does not add a global context cache.

The maintained directed tests cover multi-step Adam/AdamW parameters and both
moment states, nondefault betas and epsilon, changing per-group learning rates,
missing gradients, strict FP32 graphs, and explicit CPU parameters with a CUDA
runtime default, plus Parameter subclasses with overridden multiplication
and subtraction. SGD covers no momentum, momentum, Nesterov, weight decay,
native dampening policy and changing per-group learning rates.
The final production optimizer/batching suite passes 37 tests on CPU with
one accelerator-only skip. On real CUDA the 29 optimizer tests, nine batching
tests and two Accelerate integration tests all pass (40/40, 25.51 s).
Both sessions report zero fallback and verify actual placement. The batching
spy confirms one native Getitem fetch and zero default-collation calls.
The independent real-PyTorch CPU batching oracle passes eight numerical/API
cases, with the Jittor-only native-dispatch spy skipped (nine collected).
These gates include accumulation, resume and missing-gradient state behavior;
independent real-Torch optimizer oracle scope, native regression and structure
results are recorded separately. Three optimizer tests mixed native Vars with
Torch-only `.grad`/`.backward()` interfaces; their Torch-style portions now
construct explicit frontend tensors, while native direct/split backward
coverage is retained. Exact frozen-baseline replay confirms the stale tests
failed before this repair; the broader legacy family installer/placement and
dtype-string failures are likewise unchanged and remain outside this scope.

The native optimizer regression passes 37/37 tests in its independent native
process: closed-form SGD/Adam/AdamW rules, low-precision dtype preservation,
gradient plumbing and CPU save/resume. CPU/CUDA update cases execute under
the declared test-device sweep. The layout checker passes. The complete
structure gate executes all 1333 tests in the required Torch-compatibility
mode: 1329 pass and four fail. Targeted control against an exact `4d424432`
source snapshot confirms the four pre-existing failures:

- `test_docs_structure.py::test_every_page_is_reachable_and_every_toctree_entry_exists`: three tracked result pages are already absent from toctrees in HEAD; the current checkout additionally contains an existing untracked onboarding page.
- `test_packaging_structure.py::test_manifest_covers_runtime_trees_without_cache_payloads`: HEAD's root manifest omits five existing documentation files, and the compat manifest omits an existing dist-info resource.
- `ops/test_misc_structure.py::test_public_surface_and_private_ownership`: the old expected public-name set omits `sinc`; the unchanged native misc tree and frozen test reproduce the same failure.
- `runtime/test_flag_scope_contract.py::test_no_test_leaves_a_jittor_flag_changed`: HEAD's `test_conv1d_parameter_sync.py:52` assigns `use_cuda` without restoration.

The local `structure_head_control.json` preserves these exact baseline
failures. Unrelated architecture and existing untracked documentation are
outside this performance repair; this is not a full-structure green result.
CPU child-process compilation and CUDA compilation occupy separate native
configuration caches, and the warmed CUDA binary remains unchanged.
No ROCm/NPU performance claim is made.

Frozen FP32 performance-commit source hashes:

| File | SHA-256 |
| --- | --- |
| `optim/base.py` | `c779fcc292efc8860fceb0806b18b10fee86352e601b32c0cdaf72b4623e60a3` |
| `optim/algorithms/adam.py` | `ec2d62eff7da73a91b9e358ab2114ebc7dca8be2c2b3b3323c470be080aa10c2` |
| `optim/algorithms/sgd.py` | `35a4c76d3b01cdf93cc3f9776d47bfd3b620fa71c135efa87706d1bd001a478b` |
| `compat/torch/installers/data.py` | `0c39609829bec692241a9c2d34f8fd2ae21a27fdcaf5960ad624fcb8fe2e2492` |

## Scoped compatibility-autocast followup

The followup to `1048a6ec` addresses a separate correctness defect exposed by
an independent AMP oracle: native AMP can demote FP32 pointwise arithmetic,
mean and Adam's square root while compatibility autocast is active. The
frontend now preserves input dtype for Tensor add/subtract/multiply/division
operators and mean on CPU/CUDA. Functional add, mul and the multiply alias
call their original canonical native functions inside a local AMP-disabled
scope, retaining alpha/out behavior and native/custom-subclass protocols.
Canonical Adam math alone also uses a local AMP-disabled scope; closures and
backward retain their caller's autocast context. The shared FP32 optimizer
dispatcher falls back whenever native AMP is enabled. Raw native AMP and
other backends retain their existing policies.

The final integrated regression passes 46 tests with two skips on CPU
(9.03 s) and all 50 tests on real CUDA (15.35 s), covering optimizer,
batching, Accelerate and the final ten-case AMP test file. The sessions
report zero fallback and verify actual placement. Functional tests cover
raw native Vars, custom Parameter arithmetic overrides, aliases and alpha/out
protocols. A final production probe uses no ablation flags and confirms FP32
functional arithmetic, mean and division, identical three-step AdamW/SGD
trajectories inside and outside autocast, and the default loss scale 65536.
Against independent real-Torch CUDA, parameter trajectories match exactly
and Adam moment-state maximum error is 1.86e-9. The scaled gradients match
exactly (weight 170.625, bias 1365). Maintained tests also exercise genuine
overflow skip/backoff followed by a finite update and closure-context
preservation. Finite-only smoke tests are not used as numerical parity evidence.

The affected optimizer/API owner and import-boundary structure subset passes
all 31 cases (4.01 s), and the layout checker passes. The full 1333-case
structure result and four proven baseline failures above remain the full-gate
record; the followup subset is not presented as a new full-structure green
run. A final warmed FP32 SGD sanity run measures 3.374 ms direct and 3.819 ms
wrapped with zero fallback, showing no clear performance regression relative
to the earlier direct/full control under shared GPU load. It is a short
sanity check, not a replacement for the FP32 ABBA attribution experiments.

This is a bounded autocast correction, not a complete AMP parity claim.
Half-precision Python weak-scalar overflow and the general public sqrt policy
remain open as
[KI-TORCH-AMP-001 and KI-TORCH-AMP-002](../../agent/manuals/known-issues.md#ki-torch-amp-001-half-precision-weak-scalars-can-overflow-before-arithmetic).
The current shim GradScaler uses a Python float scale; independent Torch
uses a typed zero-dimensional scale. This followup does not modify the scaler.
Weak Python-scalar promotion and Torch's CPU/CUDA typed-scale paths must be
compared separately, and a future weak-scalar repair must jointly verify the
shim scaler's overflow detection, skipped updates and backoff. Canonical
Adam's local scope avoids the sqrt mismatch without claiming a public sqrt
repair.

Final AMP followup source hashes:

| File | SHA-256 |
| --- | --- |
| `optim/base.py` | `50f362f352a2672cc5eec852c55c54f311f2e38c4abbf167e01dcff36679d96e` |
| `compat/torch/optimizer_api.py` | `c08bc20d88b06927871b5047c2579154d1309015297c34d37d8b362ca219bf17` |
| `tensor/method_api.py` | `d7bbd2ac0e0de5b247d2267202d30728d24655ed5a730723047f11d85b9ee40a` |
| `tensor/methods.py` | `29e932f1e4fd970f74503c0da3ba142f9957c09446358c957b2e2ca0dbf07caa` |
| `tensor/shape_api.py` | `ab11c0602d3fb46c6525535ee8752eda377000d1f04b2fba4be562a8f55eedbf` |
| `test_accelerate_amp.py` | `510e8398e49fb86642bfe07d2ff566947186daacb6eb2e3d13167e30c8f38096` |

## Reproduction

Unversioned experiment scripts and compact JSON/NPZ/profile outputs live under
`${JITTOR_LAB_ROOT}/Accelerate/perf_repair/`; baseline source snapshots and raw
logs live under `${JITTOR_LAB_ROOT}/_state/Accelerate/perf_repair/`. The isolated
Jittor and real-PyTorch environments live under
`${JITTOR_LAB_ROOT}/transformers_compat/`. Run serially on one allocated CUDA
device, with separate caches for tests and benchmarks, and assert the real
PyTorch oracle is not the shim. Both environments use the same downstream pins.

The local `mlp_probe.py` harness accepts `--backend native|shim|torch`, `--device cpu|cuda`,
`--optimizer adamw|sgd`, and `--modes direct,prepared,full,loader,full,direct`.
`--adam-abba --modes full,full,full,full` substitutes the frozen baseline update
in A rounds inside the same process. `--profile` records an extra round outside
throughput. `optimizer_probe.py` verifies precomputed-gradient parameter/state
trajectories against the independent oracle. The local
`reproduction_manifest.md` records captured MLP/ResNet argument sets, harness
SHA-256 and result-artifact SHA-256; raw device/resource snapshots remain
unversioned. Native/structure commands and result counts are recorded there;
structured JUnit results are kept in the state directory.

These are manual attribution experiments. The current ASV
`OptimizerStepBenchmarks` uses native optimizers and native Vars, so it does not
exercise this Torch-frontend repair; a maintained compat ASV gate remains open.
