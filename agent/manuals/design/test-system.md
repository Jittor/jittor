# Test-system refactor — modern, PyTorch-style unit-test architecture

> Status: 🟡 in progress (2026-06-28). Foundation built + validated; **op_db = 227 ops**
> (Rounds 9–11 added 37 core ops + standalone dtype/RNN/distributions modules), all green
> on CPU+CUDA. torch_compat migration + coverage report still in flight. Replaces the ad-hoc
> per-file scaffolding the audit catalogued, WITHOUT dropping existing coverage.

## Why (the audit's verdict)

A fan-out audit of all 196 `python/jittor/test/*.py` files found the suite
*under-measures the bug class that matters most here* — **silent-wrong backward**:

- **Forward-only epidemic**: ~40 files test forward only, concentrated exactly on
  the riskiest gradients (every activation, every norm, pad, interpolate, reduce,
  attention, scatter). A forward-correct op with a wrong gradient sails through.
  The marquee example: `test_torch_compat_norm` is forward-only, yet every norm bug
  fixed (`d4c7927a`/`98dfaf04`/`48024e98`) was a *backward* small-variance one.
- **Copy-pasted scaffolding**: the torch-availability guard (~35 files), a local
  `check_equal` with divergent tolerances (~12), hand-rolled numerical-grad (~11),
  and the `if jt.has_cuda: use_cuda=1` device sweep (~99).
- **Silent-pass / tautological oracles**: `try/except-print` swallowing assertion
  failures; backward "thorough" tests that only check `grad(key*key)==2x`
  (self-consistent, can't catch a wrong index-routing grad); fully-disabled files
  that still count as "tests".

## The design (mirror PyTorch — do NOT invent bespoke mechanisms)

User directive: reference how authoritative projects (PyTorch) organize tests; no
freelancing (an earlier "golden file" idea was scrapped). Real torch 2.12.1 source
is readable at `~/rt_venv/lib/python3.11/site-packages/torch/testing/_internal/`.

Layout under `python/jittor/test/`:

```
_internal/
  common_utils.py            JittorTestCase(assertEqual, make_tensor, parametrize),
                             dtype groups, tolerance policy, device detection
  common_device_type.py      instantiate_device_type_tests, @ops/@dtypes/@onlyCPU/
                             @onlyCUDA/@onlyNPU/@skipCUDAIf  (cpu/cuda/npu via has_acl)
  gradcheck.py               gradcheck / gradgradcheck / numerical_vjp (FD vs jt.grad)
  opinfo/core.py             OpInfo, SampleInput, UnaryUfuncInfo, BinaryUfuncInfo,
                             ReductionOpInfo, DecorateInfo/skip/xfail
  opinfo/definitions/        per-domain OpInfo files (auto-discovered); _refs.py shared
  common_methods_invocations.py   op_db = aggregate of all definitions/*.py
test_ops.py                  TestCommon (fwd vs numpy ref) + TestGradients (gradcheck),
                             generated over op_db × device × dtype
test_norm.py                 float32 small-variance backward STABILITY (the gap gradcheck
                             can't see; float32 analytical vs float64 numerical_vjp)
test_regression.py           one named lock per fixed §4 silent-wrong bug (commit cited)
test_kernel_traps.py         §4-B inherent-behavior gotchas (0-d, isnan, neg-dim,
                             dtype lattice, float64 narrowing)
```

### Oracle taxonomy (why a green test means something)

- **Forward**: compared to an INDEPENDENT numpy `ref` in each OpInfo (not jittor-vs-jittor).
- **Backward**: `gradcheck` — numerical (central-difference) vs analytical (`jt.grad`)
  Jacobian, in **float64 on CPU**. Because the forward is already pinned to numpy, a
  correct gradcheck *transitively proves the backward matches torch*, with no
  live-torch dependency. This is exactly how PyTorch verifies derivatives.
- **float32 stability** (norm cancellation): a *separate* test — float32 analytical
  grad vs a float64 numerical reference at small variance. gradcheck runs in float64
  where the cancellation does not occur, so it cannot catch this; `test_norm.py` does.

### Key adaptations to jittor

- Device is a global flag (`jt.flag_scope(use_cuda=...)`), not a tensor attr, so the
  generated per-device method runs its body inside the right flag scope and receives
  the device label. ACL is distinguished from CUDA via `has_acl` → labelled `npu`.
- `gradcheck` runs on **CPU only** (`@onlyCPU`): the derivative *formula* is
  device-independent, and float64 matmul is unsupported on this CUDA/cuBLAS path
  (`CUBLAS_STATUS_NOT_SUPPORTED`). Forward parity still runs on every device.
- `op_db` is built by auto-discovering `definitions/*.py` — adding ops never edits a
  central file (no merge conflicts when several domains are added at once).

## Validation (all on this box: jt311 + CUDA + ~/rt_venv real torch)

- **op_db = 163 operators** (14 domains, expanded from 27 via two parallel authoring
  workflows). `test_ops`:
  - **forward 163/163** vs numpy ref (CPU+CUDA) after triaging the initial failures
    (test-authoring fixes, 1 harness fix, real findings).
  - **gradcheck: 139/163 ops have a backward test**; all differentiable ops pass
    (the 24 without are non-differentiable — comparisons/argmax/integer/complex-IO —
    plus cholesky symmetric-gauge & split multi-output skipped with honest notes,
    both backward-verified elsewhere). Suite **all green**.
  - Second-round domains closed the audit's named gaps: **SDPA** (+causal), **embedding**
    (+a standalone `padding_idx` grad-zero check for §4 `311eedf6`), **interpolate**
    (nearest/bilinear) / **grid_sample** / **affine_grid**, **pad** (constant/reflect/
    replicate/circular), **fft/ifft/rfft/irfft** — all backward-bug-prone, now tested.
  - gradcheck proven to PASS correct ops and FAIL a deliberately-broken backward, plus
    gradgradcheck (which surfaced the norm 2nd-order gap).
- `test_norm`: 6/6 — the four small-variance backward-stability locks pass at tol 1e-3
  (would fail if the stable jt.Function backward regressed) + running_var Bessel.
- `test_regression`: 11/11 silent-wrong locks. `test_kernel_traps`: 8/8 (1 documented
  expectedFailure for the index_select bug). Coverage report: `_internal/report.py`.

## Findings surfaced by the new suite (the point of it)

1. **`jt.index_select` ignores negative dim** (silent-wrong): `index_select(x[2,3,4],
   dim=-1, [0,2])` → shape `(2,3,4)` instead of `(2,3,2)`; dim>=0 correct. Negative-dim
   normalization missing (cf. the dim>0 fix `3eb7bc78`). Locked as
   `expectedFailure` in `test_kernel_traps.py` — flip to a hard assert when fixed.
2. **`layer_norm`/`group_norm` have no 2nd derivative**: gradgradcheck shows analytical
   0 vs numerical ≠ 0 — the stable norm backward is a non-differentiable `jt.Function`
   (the known "native 2nd-order autograd / jvp" gap). Declared `supports_gradgrad=False`.
3. **`jt.array` silently narrows** float64→float32 AND int64→int32 — a real trap for any
   double-precision reference; pin dtype explicitly (`jt.array(a, dtype=...)`). Locked in
   `test_kernel_traps.py`; gradcheck/make_tensor pin dtype because of it.
4. **`jt.floor_divide` truncates toward zero** (C semantics: `-5//3 == -1`) where
   numpy/torch floor toward −∞ (`-2`). A torch-parity gap; the OpInfo samples only
   non-negative dividends and records the divergence (fix belongs in the op).
5. **`jt.equal` is whole-tensor equality** (returns one bool, like `torch.equal`), not
   elementwise — the elementwise op is `a == b` (`torch.eq`). Easy to confuse.
6. **fp16/bf16 elementwise backward upcasts the gradient to float32** (`test_low_precision.py`):
   matmul preserves the low-precision grad dtype, but elementwise backward returns a
   float32 grad where torch keeps fp16/bf16 — breaks `param.grad.dtype == param.dtype`
   in mixed-precision training. Forward dtype is correct, so forward-only tests miss it.
   Locked as `expectedFailure`.
7. **`jt.median` is broken** (`misc.py:321-322`): `_, x = jt.argsort(x, dim)` expects
   native argsort's `(idx, val)` 2-tuple but the torch-compat override returns indices
   ONLY → "too many values to unpack"; plus an off-by-one (`dim-1`). `median` OpInfo is
   retained but `skip`-ped with the reason so it reactivates on fix. (`kthvalue` is fine.)
8. **`mean` over an empty axis returns 0, not nan** (jittor avoids 0/0). Minor parity
   divergence, `expectedFailure`.
9. **jittor scalar/fused arithmetic carries more precision than strict float32**:
   `(x + 1e8) - 1e8` keeps a `1.0` that IEEE single (torch/numpy) loses to 0.
   `expectedFailure`.
10. **sub-32-bit integer reduce fails to COMPILE on CUDA** (found by the full device-parity
    run): `sum`/`prod`/`max`/`min` over `uint8`/`int8` emits `atomicAdd(uint8*, uint8)` /
    `atomicMax(uint8*, …)`, which CUDA has no overload for → JIT compile error. (int8/16
    *max/min* were fixed in `eb3c8bee`; reduce-*add* and 8-bit are still broken.) The
    `*_int_reduce` device-parity tests are `expectedFailure` with this note. NB: under the
    *parallel* compiler this error surfaced as collateral in unrelated tests (pad_constant)
    — the verifier now compiles serially so each error is attributed to its own op.

## How to run

```bash
JT=/home/zy/miniconda3/envs/jt311/bin/python
PYTHONPATH=$PWD/python $JT -m jittor.test.test_ops -v          # op battery (cpu+cuda)
PYTHONPATH=$PWD/python $JT -m jittor.test.test_norm            # norm backward stability
PYTHONPATH=$PWD/python $JT -m jittor.test.test_regression      # §4 silent-wrong locks
PYTHONPATH=$PWD/python $JT -m jittor.test.test_kernel_traps    # §4-B traps
JITTOR_TEST_DEVICES=cpu ...                                    # restrict device matrix
cache_name=cardN CUDA_VISIBLE_DEVICES=N ...                    # parallel-run isolation
```

## Round 2 — low-level / core coverage (2026-06-27)

Directive: "cover more cases, especially the low-level / core parts — those can't have
bugs." The #1 gap was **device-specific backward** (gradcheck is CPU-only, but the
project's worst bugs are device kernels: CUDA scatter `880cd6ad`, int reduce
`eb3c8bee`, setitem negindex `58e95b73`). Added:

- **`test_device_parity.py`** — for every op, run forward AND backward on identical
  inputs on CPU and the accelerator and assert they match (float32). CPU is the oracle;
  a divergence means the accelerator kernel is wrong (how the scatter bug was found).
  Verified **48 ops 0-divergence** + a targeted risk sweep (gather/scatter/index/
  getitem/reindex/int-reduce/matmul). Skips with a report on a CPU-only build.
- **op_db → 185 ops**: `+22` core meta-operators — **getitem ×11** (slice/int/step/
  mask/fancy/ellipsis/newaxis/negative, incl. `getitem_fancy_negative` regression-locking
  `58e95b73`), **reindex/reindex_reduce + broadcast/repeat/unsqueeze-via-reindex** (the
  fusion primitives conv/pool lower to), **integer-dtype reduce ×6** (sum/prod/max/min
  over int + all/any over bool — the int8/16 reduce kernel). All forward + gradcheck green.
- **`test_setitem_core.py`** (14 tests, CPU+CUDA) — the setitem/scatter kernel:
  negative-index backward, scatter-add/max/min with duplicate indices (locks
  `58e95b73`/`880cd6ad`). All green on both devices.
- **`test_type_system.py`** (32 tests) — dtype-promotion lattice (all pairs), cast
  methods (`.long()`==int64, `.double()`==float64…), NanoString, and the preserved
  `@skip`-locked semantic-diffs. All green.
- **`test_autograd_engine.py`** (9 tests, CPU+CUDA) — the autodiff *engine*: diamond-graph
  accumulation, broadcast-back (grad sums over broadcast/reduced axes), stop_grad
  mid-graph, grad through view chains, multi-`backward()` accumulation, higher-order
  through a fused expression, matmul-chain. Analytic references. All green.
- **`test_fusion_correctness.py`** (6 tests, CPU+CUDA) — jittor *is* a fusion compiler;
  the audit found its tests only parse log lines. This computes each expression with
  fusion ON vs OFF (`no_fuse=1`) vs numpy, fwd+bwd — a fused-vs-unfused divergence
  isolates a fusion miscompile. Covers elementwise chains, the normalization fusion,
  where-fusion, cumsum-fusion, and the `x==x`/inf-nan-ternary trap patterns. All green.
- **op_db → 190**: `+22` core meta-operators (**getitem ×11** incl. `getitem_fancy_negative`
  regression-locking `58e95b73`; **reindex/reindex_reduce + via-reindex** fusion primitives;
  **integer-dtype reduce ×6**) `+5` ordering ops (**sort/topk** with the index-routing
  *backward* — the audit's tautological-oracle gap — argsort/kthvalue/median). All
  forward + gradcheck green (median skipped, see finding #7).
- **`test_setitem_core.py`** (14 tests, CPU+CUDA) — setitem/scatter kernel: negative-index
  backward, scatter-add/max/min with duplicate indices (locks `58e95b73`/`880cd6ad`).
- **`test_edge_cases.py`** (24 tests, CPU+CUDA) — empty tensors, non-contiguous views,
  extreme/inf values, degenerate ranks (5-D, single-element), vs numpy. (2 documented
  `expectedFailure` divergences: mean-over-empty, scalar-add precision.)
- **`test_optim_core.py`** (16 tests) — SGD/Adam/AdamW/RMSprop update rules vs the
  closed-form math (momentum, weight-decay vs decoupled, bias correction, state). Green.
- **Harness hardening**: `common_methods_invocations` now raises on a duplicate OpInfo
  `full_name` — the generic templates key tests by name, so a collision would *silently
  drop* one op's tests (a silent-wrong in the test system itself); now it's a loud error.
- **Full device-parity run** (`test_device_parity.py`, all 185 prior ops, CPU-vs-CUDA
  fwd+bwd): every op's accelerator kernel matches the CPU oracle EXCEPT it surfaced
  **finding #10** (sub-32-bit integer reduce fails to compile on CUDA — a real bug, now
  `expectedFailure`). The cupy-backed linalg ops (det/slogdet/inv/...) are skipped (cupy
  CUDA compile unavailable in this venv, not a jittor kernel bug); multi-output ops
  (split/slogdet) needed a harness fix; the verifier now compiles serially so a kernel
  compile error is attributed to its own op rather than a sibling. **No silent-wrong
  divergence** (a kernel computing the wrong number) was found — the one real issue is a
  loud compile failure.

## Round 9 — breadth: core integer/special/structure ops (op_db 190 → 210)

Filled the largest remaining gaps in the operator registry with **20 jittor-CORE ops**
(not torch-compat shim) that had zero coverage, each dropping straight into the OpInfo
machinery (forward-vs-numpy + gradcheck + CPU-vs-CUDA device-parity for free). Three new
definition modules + one dedicated backward module:

- **`definitions/bitwise_ops.py`** (8 ops) — `bitwise_and/or/xor/not`, `left_shift`,
  `right_shift`, `logical_xor`, `logical_not`. Pure C++ INTEGER kernels (`binary_op.cc`),
  the classic accelerator silent-wrong spot, previously untested. `supports_autograd=False`;
  swept across the full integral-width set on CPU and CUDA. **CUDA matches CPU bit-exact
  (0.0 error)** on every one. Resolved one test trap: jittor's *binary* bitwise on `uint8`
  promotes the output to `int8`, so the sample builder keeps `uint8` values non-negative
  (sign-bit two's-complement coverage stays on the SIGNED widths, which match numpy exactly).
- **`definitions/special_unary.py`** (5 ops) — `erfinv`, `lgamma`, `digamma` (the
  differentiable special functions the Gamma/Dirichlet distributions depend on; `lgamma`
  /`digamma` are `jt.Function`s called via `.apply`), `deg2rad`, `rad2deg`. Forward oracle
  is **SciPy** (independent impl). Finding: these ship ~1e-7-accurate kernels, so float64
  finite-difference gradcheck is round-off-dominated — `digamma`'s FD gradcheck is
  therefore `skip`ped with a recorded reason, NOT silently passed; its backward is verified
  analytically instead (next module). `erfinv`/`lgamma` FD-gradcheck pass cleanly once run
  in the real `@onlyCPU` (use_cuda=0) context (the earlier crashes were the known float64-
  cublas-on-CUDA limitation, not a backward bug).
- **`definitions/structure_ops.py`** (7 ops) — `tril`/`triu` (triangular masks, gradchecked),
  `cross`/`kron`/`tensordot` (bilinear, both operands gradchecked + gradgrad), `cummax`/
  `cummin` (prefix-scan, forward + parity). All green; `tensordot`'s full-contraction
  sample exercises the jittor-no-0-d-scalar convention (ref `atleast_1d`, same as reductions).
- **`test_special_grad.py`** (4 tests, CPU+CUDA) — `erfinv`/`lgamma`/`digamma` backward vs
  the SciPy CLOSED FORM (`sqrt(pi)/2·exp(erfinv²)`, `digamma`, `polygamma(1)`), plus
  `erfinv` 2nd-order. The proper oracle the FD gradcheck can't be for limited-precision
  kernels; gives `digamma`'s backward real, independent coverage.

Verify-then-fix discipline held: the round's first run produced 4 red signals (uint8
bitwise, cross 1-D shape, tensordot 0-d, the special-fn gradchecks) and **all four were
test/oracle artifacts, not jittor bugs** — fixed in the samples/refs, not the core. The
only genuine kernel behavior surfaced (uint8→int8 bitwise output promotion) is a documented
dtype quirk. Net: **+20 ops, all green on CPU+CUDA, no new core bug** — the new kernels are
now locked against regression.

## Round 10 — breadth: core nn functional + predicates (op_db 210 → 219)

Continued the breadth pass with 9 more jittor-core ops, same OpInfo-for-free pattern:

- **`definitions/nn_activations_extra.py`** (5 ops) — `log_sigmoid`, `hardtanh`, `glu`,
  `normalize`, `cosine_similarity`. Differentiable; all forward-only suspects whose
  backward is the risky part (glu mis-pairing the gated halves, normalize/cosine coupling
  every component through the quotient rule). Forward vs numpy, gradcheck + (where smooth)
  gradgrad, CPU-vs-CUDA parity — all green. `hardtanh` gradgrad off (piecewise clamp).
- **`definitions/predicates.py`** (3 ops) — `isnan`, `isinf`, `isfinite`. Bool, no grad.
  Samples deliberately CONTAIN NaN/±Inf (make_tensor never does), so this is the first
  op_db coverage of the special-value classification kernels. **CUDA classifies NaN/Inf
  bit-identically to CPU (0.0 error)** — the notorious accelerator NaN/Inf divergence point
  is verified clean (complements the fusion suite's `isnan`-through-`x==x`-fold check).
- **`meshgrid`** (added to `structure_ops.py`) — 'ij' multi-grid broadcast, stacked to one
  Var for comparison; forward + parity.
- **`definitions/search_histogram.py`** (2 ops) — `searchsorted` (binary-search kernel,
  with the `right` side flag) and `histc` (histogram bin assignment). Deterministic
  data-dependent kernels whose off-by-one boundary logic a forward-vs-numpy oracle pins
  exactly; int/count output, no grad. CUDA == CPU bit-exact.

All 11 green on CPU+CUDA, no new bug. Session breadth total: **op_db 190 → 221 (+31 core ops)**
across `bitwise_ops` / `special_unary` / `structure_ops` / `nn_activations_extra` /
`predicates` / `search_histogram` + `test_special_grad.py`.

## Round 11 — conv/pool variants + dtype-sweep + stateful subsystems (op_db 221 → 227)

Pushed breadth into the areas the user named (底层/核心): the 1-/3-D conv-pool kernels, the
dtype lattice, and the stateful RNN / distributions subsystems.

- **`definitions/conv_pool_extra.py`** (6 ops) — `conv1d`, `conv3d`, `max_pool1d`,
  `avg_pool1d`, `max_pool3d`, `avg_pool3d`. The 1-D/3-D conv & pool kernels are SEPARATE
  codegen paths from the already-covered 2-D ones. Forward pinned to an explicit numpy
  cross-correlation / windowed reference (small auditable regime: groups=1, dilation=1, a
  few stride/padding values), gradcheck + gradgrad, CPU-vs-CUDA parity — all green.
- **`test_dtype_coverage.py`** (6 tests) — the per-dtype VALUE lattice that `test_ops`
  (mostly float32/64) and `test_low_precision` (grad *dtype* only) don't pin:
  (i) **integer width sweep** — add/sub/mul/maximum/minimum/bitwise/negative/abs across
  uint8/int8/int16/int32/int64, exact vs numpy AND CPU-vs-CUDA bit-identical (every width);
  (ii) **low-precision forward** — fp16/bf16 elementwise/matmul/reduce/softmax vs an fp32
  reference and CPU-vs-accelerator, within half-precision tolerance. All green.
- **`test_rnn_recurrence.py`** (11 tests, CPU+CUDA) — RNN(tanh/relu)/LSTM/GRU, incl.
  multi-layer + bidirectional, with an INDEPENDENT numpy recurrence oracle (the legacy
  `test_rnn.py` compares against `import torch`, which under the jt-torch shim is jittor vs
  itself). Forward == numpy recurrence using the layer's own extracted weights; backward =
  jittor analytic `d/dinput` vs a float64 FD of the numpy reference. Confirms gate order
  (LSTM i,f,g,o; GRU r,z,n), the `b_hh` bias, and through-time unrolling. All green.
- **`test_distributions_grad.py`** (4 tests, CPU+CUDA) — `rsample` reparameterization
  gradients (the VAE/VI-critical, silently-detachable path): `Normal`/`LogNormal` exact
  (`d/dmu=1`, `d/dsigma=eps`, …), `Gamma` pathwise MC `d/dconc E[X] → 1/rate`, and a
  finite-&-nonzero guard for `Gamma`/`Beta`/`Dirichlet` (the detach/zero-grad bug class).

Verify-then-fix again: the Dirichlet rsample grad came back 0 — but that's **correct** (a
Dirichlet sample sums to 1, so a plain `sum()` objective is constant 1 → zero gradient, not
a detach bug); fixed the test to use a weighted objective. No jittor bug. Session breadth
total: **op_db 190 → 227 (+37 core ops)** across 7 definition modules + 4 standalone modules
(`test_special_grad` / `test_dtype_coverage` / `test_rnn_recurrence` / `test_distributions_grad`).

## Full-suite regression (all 227 ops) + 3 findings it surfaced

Ran the complete `test_ops` (forward + gradcheck + gradgradcheck, CPU) over all 227 ops:
**681 tests, 539 pass, 142 skip, 0 fail, 0 error** (green). Getting there surfaced three
real, distinct issues — all in PRE-EXISTING ops (not the new breadth work), exactly the kind
a comprehensive suite exists to catch:

1. **`cumsum` 2nd-order autodiff SEGFAULTS** (real jittor crash). `jt.cumsum` is a
   `numpy_code` op; grad-of-grad calls `NumpyCodeOp::grad` on a backward that carries no
   registered grad → null deref → process abort (worse than silent-wrong — it kills the
   run). Fixed by `supports_gradgrad=False` on cumsum (documented; same class as cumprod,
   which was already off). 1st-order gradcheck is fine.
2. **`conv2d` forward float32 tolerance** — conv sums `Ci·Kh·Kw` products, so a single
   element can miss the default `atol=1e-5` by ~2× (still ~1e-6 relative). Added an optional
   `reference_tol` to `OpInfo` (threaded into `test_reference`) and set conv2d/conv_transpose2d
   to `(1e-4, 1e-4)`. gradcheck (float64) is unaffected.
3. **`pad_constant` fractional fill fails to COMPILE on CPU** (real CPU-codegen finding).
   `nn.pad(mode='constant', value=0.7)` emits a reindex kernel whose overflow constant is the
   hex-float `itof(0x3fe6666666666666)`; jittor's CPU `asm_tuner` rewrites it into a malformed
   assembly literal → g++ `error: exponent has no digits`. INTEGER fills and CUDA are fine.
   Kept op_db's pad_constant samples on integer fills (semantics stay covered CPU+CUDA) and
   pinned the fractional-fill bug LOUDLY as an `expectedFailure` in `test_kernel_traps`
   (`test_constant_pad_fractional_fill_cpu_asmtuner`) — turns XPASS when the asm_tuner is fixed.

Device-parity (CPU-vs-CUDA fwd+bwd, all 227 ops): **215 OK + 4 xfail + 8 skip + 0 unexpected
divergence**. The 4 xfail are the documented sub-32-bit int-reduce CUDA-atomic gap
(sum/prod/max/min_int_reduce, finding #10). The 8 skip = 7 cupy-backed linalg ops
(det/inv/slogdet/solve/cholesky/qr/svd — cupy CUDA unavailable in this venv, env not kernel)
+ `median` (finding #7: jittor `misc.py::median` is broken under torch-compat —
`_, x = jt.argsort(...)` expects the native (idx,val) tuple but the override returns indices
only → "too many values to unpack"; already skipped in test_ops, now also pinned in
device-parity's known-issues). No silent-wrong kernel divergence anywhere.

Operational note: running `test_ops` and `test_device_parity` concurrently OOM-kills the
shared GPU; run them sequentially (the device-parity verifier is chunked one fresh process
per slice to bound GPU memory). The CPU `test_ops` is the correctness gate; device-parity is
the CUDA-kernel gate (every new op already passed it per-batch).

## Remaining work

- **NPU/ACL device parity** — the device-parity test runs on whatever accelerator the
  build has; on this box that's CUDA. The Ascend leg still needs a real-NPU run.
- **Per-op dtype sweep** — ✅ a dedicated value-level lattice now exists
  (`test_dtype_coverage.py`: integer widths + fp16/bf16 forward). Still open: weaving
  fp16/bf16/int *per op* into `test_ops` via `@dtypes`, plus more edge samples
  (empty/0-d/non-contiguous), negative/should-raise tests, out=/method variants.
- **Stateful subsystems** — ✅ RNN/LSTM/GRU (`test_rnn_recurrence.py`) and distributions
  `rsample` grads (`test_distributions_grad.py`) now covered. Still open as OpInfos:
  optimizers (have `test_optim_core`), native complex64, einsum, serialization.
- **Migrate/retire the legacy `test_torch_compat_*` + op files** onto the harness
  (audit B1–B3), preserving the must-keep assets (embedded numpy refs, @skip-locked
  semantic-diffs, gauge-invariant linalg, negative tests, backend-selection log checks,
  memory-lifecycle assertions).
