# Active Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-09-09
- Baseline: `7419412e2` plus KI-EXEC-001
- Owner: Jittor core maintainers
- Review cadence: on every strict XPASS, related fix, or quarterly maintenance

This ledger contains reproduced, currently relevant defects and explicit
limitations. Historical fixes remain in Git and dated `docs/results/` reports;
they are not copied here. Every entry needs executable evidence and an exit
condition. Environment outages are recorded in result reports rather than as
framework defects.

## Severity guide

- **Critical:** silent wrong result, gradient, state, or device placement.
- **High:** supported operation crashes or fails to compile without a practical
  equivalent path.
- **Medium:** compatibility divergence with a documented workaround or narrower
  supported domain.
- **Research:** an intentionally unsupported capability requiring architectural
  work.

## KI-TEST-001: formerly silent test cases expose unresolved contracts

- Severity: Medium
- Status: Strict expected failures
- Owner: test infrastructure and affected backend maintainers
- Evidence: `TestBF16.test_reduce_dtype_infer`,
  `TestCudnnConvOp.test_backward_nhwc`, `TestCore.test_swap`,
  `TestCore.test_swap_cuda`, `TestRingBuffer.test_dataset`,
  `TestOptStateDict.test_opt_state_dict`, and
  `TestArgPoolOp.test_cuda_old_pool`
- Symptom: these tests were permanently disabled with an initial `return` or
  `skipIf(True)`, so pytest reported success without executing their contracts.
  They now run as strict expected failures when their declared hardware/network
  prerequisites exist; optimizer state-dict coverage fails explicitly until it
  has an implementation.
- Workaround: do not cite these nodeids as passing evidence for reduction dtype
  inference, NHWC cuDNN backward, tensor swapping, repeated dataset RingBuffer
  use, optimizer state restoration, or legacy CUDA pooling.
- Review/expiry condition: fix and independently verify each named contract,
  then remove its expected-failure marker and this entry when the list is empty.

## KI-COMPILER-001: parallel compiler can corrupt process state

- Severity: High
- Status: Open for non-Jupyter workloads; Jupyter SIGCHLD path fixed
- Owner: compiler/executor maintainers
- Evidence: [investigation and reproduction](../../docs/development/known-issues/parallel-compiler-segfault.md)
- Workaround: set `jt.flags.use_parallel_op_compiler = 0` for deterministic
  validation workloads
- Resolved subcase: Jittor's process-wide `SIGCHLD` handler quick-exited a
  Jupyter kernel when any child was killed. Jupyter now retains SIGCHLD
  ownership. A later complete notebook smoke still reproduced a separate death
  with eight compile workers, including with Jittor's signal handler disabled,
  so the maintained notebook gate remains serial. See the
  [SIGCHLD verification and addendum](../../refactor-wip/results/2026-08-21-jupyter-sigchld.md).
- Review/expiry condition: remove only after sanitizer-backed root cause and
  repeated cold/warm stress, deadlock, multiprocess-cache, and performance gates

## KI-BACKEND-001: narrow integer sum/max/min lack NPU atomics

- Severity: High
- Status: NPU skips
- Owner: reduce and backend maintainers
- Evidence: [`reduce_dtypes.py`](../../tests/opinfo/definitions/reduce_dtypes.py)
  and [device parity](../../tests/backends/parity/test_device_parity.py)
- Symptom: `sum`, `max`, and `min` for sub-32-bit integer samples abort
  because their required ACL atomic overloads are not implemented. Jittor core
  bool `all_` and `any_` also lack a maintained generic ACL reduction kernel.
  The public `jt.all`/`Tensor.all` and `jt.any` paths needed by Transformers now
  route bool inputs to CANN 9 `aclnnAll`/`aclnnAny`; numeric inputs first compare
  nonzero. This does not establish support for the skipped core reduction ops.
- Workaround: promote inputs to a supported width before reduction on NPU; use
  the verified public CANN 9 truth reductions where their semantics apply
- Resolved subcase: full, single-axis, and multi-axis integer `prod` now route
  through CANN 9 `aclnnProd`/`aclnnProdDim`; uint8, int8, int16, int32, and
  int64 match NumPy without CPU compilation or fallback
- Review/expiry condition: every affected dtype executes and matches the CPU
  reference on a real NPU, turning the skips into passes

## KI-BACKEND-002: composed atan2 can crash on NPU

- Severity: High
- Status: NPU skip
- Owner: binary operator and ACL backend maintainers
- Evidence: [`pointwise_binary.py`](../../tests/opinfo/definitions/pointwise_binary.py)
  and [Ascend 910B validation](../../refactor-wip/results/2026-08-28-ascend-910b-validation.md)
- Symptom: the maintained float32 `atan2` composition can terminate the process
  with an ACL vector-core exception on a real 910B3
- Workaround: run this operation on a backend with a maintained `atan2` kernel;
  do not mask the process failure with a broad CPU fallback
- Review/expiry condition: the float32 OpInfo reference and focused crash
  reproducer pass repeatedly on a real NPU without an expected skip

## KI-BACKEND-003: complex irfft can stall on NPU

- Severity: High
- Status: NPU skip
- Owner: FFT and ACL backend maintainers
- Evidence: [`fft.py`](../../tests/opinfo/definitions/fft.py) and
  [Ascend 910B validation](../../refactor-wip/results/2026-08-28-ascend-910b-validation.md)
- Symptom: the complex-to-real inverse FFT does not complete within 600 seconds
  on a real 910B3, and the stalled native call is not interrupted reliably by
  pytest's signal timeout
- Workaround: execute `irfft` on a backend with a maintained complex FFT path
- Review/expiry condition: forward values match NumPy and the operation exits
  within the maintained timeout on repeated real-NPU runs

## KI-OPS-002: integer floor-division backend verification incomplete

- Severity: Critical
- Status: Core fix verified on CPU/CUDA/NPU; ROCm verification pending
- Owner: binary operator maintainers
- Evidence: [`test_floor_divide.py`](../../tests/ops/test_floor_divide.py),
  [`sample_floor_divide`](../../tests/opinfo/definitions/pointwise_binary.py), and
  [2026-08-21 verification](../../refactor-wip/results/2026-08-21-floor-divide.md)
- Previous symptom: C++ integer division made negative quotients truncate toward
  zero instead of flooring toward negative infinity
- Current implementation: shared CPU/CUDA codegen subtracts one exactly when a
  nonzero remainder has the opposite sign from the divisor; fixed vectors pass
  for uint8/int8/int16/int32/int64, and the selected int64 OpInfo samples cover
  negative operands on CPU, CUDA, and a real Ascend 910B3
- Workaround on the unverified backend: compare representative negative operands
  against `numpy.floor_divide` before relying on the backend
- Review/expiry condition: pass the same fixed-vector and OpInfo coverage on a
  real ROCm device, then remove this entry

## KI-OPS-003: floor division truncates float operands to integers

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: binary operator maintainers
- Evidence: `compat/tests/torch/test_division_remainder_family.py::
  test_float_floor_divide_matches_numpy` (strict expected failure on CPU and CUDA)
- Symptom: `floor_divide` casts float operands to integers before dividing, so
  the fractional part is discarded and the result comes back as `int32` where
  PyTorch returns a float. For `[-5.0, -2.7, -0.5, 2.7] // 2.0` the operator
  returns `[-3, -1, 0, 1]` where `numpy.floor_divide` gives `[-3, -2, -1, 1]`.
  The values match neither flooring nor truncation of the true quotient because
  the truncation happens to the *operands*: `int(-2.7) // 2 == -1`, and
  `int(-0.5) // 2 == 0`. Negative dividends whose magnitude is already an exact
  multiple happen to come out right, which is why a positives-only or
  whole-number check passes.
- Distinct from [KI-OPS-002]: that entry covers the *integer* path, whose
  flooring fix is verified on CPU, CUDA and a real 910B3. The integer path is
  confirmed correct here; only float operands are affected.
- Workaround: `(a / b).floor()` for float operands, which computes the quotient
  first and keeps the floating result type
- Review/expiry condition: float operands divide at full precision and return a
  floating dtype, the strict expected failure above turns red, and this entry is
  removed

## KI-OPS-004: reducing a rank-0 tensor fails an internal invariant

- Severity: High
- Status: Reproduced on CPU and CUDA, unfixed
- Owner: reduction operator maintainers
- Evidence: `compat/tests/torch/test_division_remainder_family.py::
  test_reducing_a_scalar_tensor` (strict expected failure)
- Symptom: `sum`, `mean`, `max` and `min` on a rank-0 tensor abort in
  `expr.cc:304` with `Check failed: nodes.size() == 1  Something wrong... Could
  you please report this issue?`, reported through
  `fused_op:( reduce.add,)` with `[Input]: float32[]`. PyTorch returns the value
  unchanged. Generic code that reduces without checking rank -- `loss.sum()`
  where the loss is already scalar -- hits this, and the message surfaces an
  internal invariant rather than naming the unsupported shape.
- Workaround: skip the reduction when `tensor.ndim == 0`, or `reshape(1)` first
- Review/expiry condition: rank-0 reductions return the input value on CPU and
  every advertised accelerator, the strict expected failure above turns red, and
  this entry is removed

## KI-OPS-006: max/min drop NaN where every other reduction propagates it

- Severity: Critical
- Status: Reproduced on CPU and CUDA, unfixed; a working implementation was
  measured and rejected on cost
- Owner: reduction and binary operator maintainers
- Evidence:
  [`test_minmax_nan_propagation.py`](../../tests/ops/test_minmax_nan_propagation.py)
  `::TestMinMaxNanPropagationCpu::test_max_and_min_reductions_propagate_nan` and
  `::test_elementwise_maximum_and_minimum_propagate_nan`, strict expected
  failures, with the same pair on the CUDA class
- Symptom: `jt.max([nan, 1.0, 2.0])` returns 2.0 and `jt.min` returns 1.0 where
  NumPy and Torch both return nan. `sum`, `mean` and `prod` propagate correctly,
  so one reduction family answers a NaN input two different ways. The
  elementwise operators also disagree with themselves across backends:
  `jt.maximum(1.0, nan)` is 1.0 on CPU and `jt.maximum(nan, 1.0)` is 1.0 on
  CUDA, because the two lower the same ternary differently.
- Cause: `std::max(a, b)` is `a < b ? b : a`, and CUDA's `::max` on floats
  lowers to `fmaxf`. Every comparison against NaN is false, so the operand that
  is not NaN survives; the reduction starts at its identity (`lowest()` on CPU,
  `-inf` on CUDA -- see [KI-OPS-008]) and folds `max(acc, x)`, so a NaN can
  neither enter the accumulator nor stay in it. The two rows are
  `maximum`/`minimum` in both tables of
  [`common_op_type.cc`](../../src/type/common_op_type.cc).
- Why it is not simply fixed -- three costs, all measured on `1e25ff68a`:
  1. The NaN test cannot be a comparison. JIT kernels compile with `-Ofast`,
     which implies `-ffinite-math-only`; `x != x` and `std::isnan(x)` fold to
     false there, and a max written with either collapses back to a single
     `vmaxss` (confirmed from the emitted assembly). Only a bit test on the
     exponent and mantissa survives. [`numerical.py`](../../python/jittor/ops/numerical.py)
     hits the same wall for isnan/isinf and drops that one kernel to `-O2`
     through `_simple_for`, which a reduction kernel cannot afford.
  2. A bit test is not a GCC-recognised reduction, so max/min reductions lose
     auto-vectorisation. Timed at the kernel's own flags
     (`-Ofast -march=native`), the NaN-propagating reduce runs at 3.9-4.9 GB/s
     flat, whatever the working set, because it is scalar and latency-bound;
     `std::max` vectorises and reaches 28-31 GB/s at 64 MB and 40-109 GB/s while
     cache-resident. That is 7.4x-8.2x slower out of memory and 10x-28x slower in
     cache -- the spread is the denominator moving with machine load, not the
     numerator. In-tree, with the addition-reduce control at 1.00x in the same
     run, float32 `max()` and `min()` over 1M and 8M elements regressed 7.1x to
     7.5x. The elementwise operators cost only 1.2x to 1.6x.
  3. The spelling is load-bearing elsewhere.
     [`parallel_pass.cc`](../../src/codegen/opt/pass/parallel_pass.cc) and
     [`atomic_tuner_pass.cc`](../../src/codegen/opt/pass/atomic_tuner_pass.cc)
     match the literal `std::max(T(a),T(b))` / `::max(T(a),T(b))` to route a
     parallel reduction through `cpu_atomic_max`/`cuda_atomic_max`, so changing
     the expression makes CUDA reductions fail to compile outright
     (`Expr not match`). A complete fix therefore also has to make those atomics
     NaN-correct -- `cuda_atomic_max` is a CAS over a sign-magnitude integer
     key, where a negative NaN sorts below -inf -- plus the float16 table and
     `shared_reduce_max`/`shared_reduce_min`.
- Workaround: test for NaN separately -- `float('nan') if jt.isnan(x).any() else
  x.max()` -- rather than reading it out of the reduction. `jt.isnan` is correct
  on both backends; it is the kernel that already compiles at `-O2`.
- Review/expiry condition: a max/min reduce that propagates NaN without losing
  the vectorised reduction -- most plausibly a second, OR-folded NaN accumulator
  in the reduce codegen, which stays a recognised reduction and costs about two
  vector operations per eight elements -- lands together with NaN-correct
  atomics. The four strict expected failures above then turn red and this entry
  is removed.

## KI-OPS-007: the CUDA unary math table narrows float64 to float32

- Severity: Critical
- Status: Reproduced on CUDA, unfixed for every entry except `round`
- Owner: unary operator maintainers
- Evidence:
  [`test_float64_unary_precision.py`](../../tests/ops/test_float64_unary_precision.py)
  `::TestFloat64UnaryPrecisionCuda::test_unary_family_keeps_float64_precision`,
  a strict expected failure; the CPU class of the same file passes, which is
  what makes this a backend divergence rather than a shared limitation
- Symptom: nearly every row of `common_op_type_cuda_map` in
  [`common_op_type.cc`](../../src/type/common_op_type.cc) is the `f` -- that is,
  single-precision -- spelling of its libm function: `::floorf`, `::ceilf`,
  `::sqrtf`, `::expf`, `::logf`, `::sinf` and the rest. A float64 operand is
  converted to float on the way in, so the result carries 24 mantissa bits
  instead of 53. Above 2**24 the answer is not merely imprecise:
  `jt.ceil(12345678901234.5)` is 12345678901235.0 on CPU and 12345679020032.0
  on CUDA, and `jt.log(1.0000000000000002)` is 2.22e-16 on CPU and exactly 0.0
  on CUDA.
- Cause: the table was written for float32 and the width was never dispatched.
  `round` now is -- `@if(@strcmp($1,float32)==0, ::rintf, ::rint)` -- and is the
  shape the remaining rows need.
- Workaround: run float64 unary math on CPU, or accept float32 accuracy and say
  so. A float64 tensor whose values stay inside 2**24 is unaffected.
- Review/expiry condition: the remaining rows dispatch on width the way `round`
  does, keeping the `f` spelling for float32 so consumer GPUs -- where float64
  throughput is a fraction of float32 -- do not pay for the fix; the strict
  expected failure above turns red and this entry is removed.

## KI-OPS-008: the CPU max/min reduction starts from a finite identity

- Severity: Critical
- Status: Reproduced on CPU, unfixed; CUDA is already correct
- Owner: reduction operator maintainers
- Evidence:
  [`test_minmax_reduction_identity.py`](../../tests/ops/test_minmax_reduction_identity.py)
  `::TestMinMaxReductionIdentityCpu::test_infinite_reductions_use_the_right_identity`,
  a strict expected failure; the CUDA class runs the same body and passes
- Symptom: `jt.max` of a float32 tensor whose every element is -inf returns
  -3.4028235e38 on CPU and -inf on CUDA; `jt.min` of an all +inf tensor is the
  mirror image. NumPy and Torch return the infinity. A fully masked attention
  row is exactly this input -- it is all -inf, and `logits.max(-1)` is exactly
  this reduction -- so the wrong value is reachable without anyone writing an
  infinity by hand.
- Cause: `init_maximum` is `std::numeric_limits<$1>::lowest()` in the CPU table
  of [`common_op_type.cc`](../../src/type/common_op_type.cc) and
  `::numeric_min<$1>()` in the CUDA table, and the CUDA one resolves to
  `-CUDART_INF` for float and double. `max(lowest(), -inf)` keeps the identity
  instead of the element, so on CPU no reduction can ever report an infinity it
  was given. Integers are unaffected: `lowest()` *is* their identity and they
  have no infinity to lose, which is why the fix has to dispatch on the dtype
  rather than replace the row.
- Distinct from [KI-OPS-006]: that entry is the NaN behaviour of the `maximum`
  and `minimum` *operators*, which is expensive to fix. This one is the identity
  the reduction folds from -- a per-output-element constant, with no effect on
  the inner loop -- and the two are independent.
- Workaround: on CPU, treat a result equal to `numpy.finfo(dtype).min` (or
  `.max` for `min()`) as possibly an infinity, or run the reduction on CUDA.
- Review/expiry condition: the CPU identity resolves to `-inf`/`+inf` for
  float32 and float64 while integers keep `lowest()`/`max()`, the strict
  expected failure above turns red, and this entry is removed.

## KI-SEMANTICS-003: floating-comparison backend verification incomplete

- Severity: Critical
- Status: CPU/CUDA verified; NPU float32 verified; full NPU dtype/ROCm pending
- Owner: compiler and comparison-operator maintainers
- Evidence: [`test_nan_self_comparisons_across_dtypes`](../../tests/debug/test_kernel_traps.py),
  [`test_float_comparisons_with_nan`](../../tests/ops/test_fusion_correctness.py),
  [2026-08-21 verification](../../refactor-wip/results/2026-08-21-ieee-nan-comparisons.md), and
  [Ascend 910B validation](../../refactor-wip/results/2026-08-28-ascend-910b-validation.md)
- Previous symptom: CPU JIT kernels inherited `-Ofast`, allowing both same-object
  and distinct floating comparisons to violate IEEE NaN behavior; low-precision
  `!=`, `<=`, and `>=` could also fail to compile on CPU
- Current implementation: floating and complex comparisons retain optimized
  `-O3` kernels without finite-math assumptions, and fused compile options are
  taken from the complete aggregated graph choices
- Current NPU result: float32 `isnan`/`isinf`/`isfinite` and all six
  same/distinct fused/unfused comparison forms pass on a real 910B3 without CPU
  fallback; general ACL float64 operation support is unavailable
- Workaround for unverified dtype/backend combinations: compare representative
  NaN values against NumPy before relying on direct comparison masks
- Review/expiry condition: pass the remaining dtype matrix on real NPU and the
  complete matrix on real ROCm, then remove this entry

## KI-DTYPE-002: implicit array construction narrows 64-bit NumPy values

- Severity: High
- Status: Accepted current default with explicit escape hatch
- Owner: dtype and compatibility maintainers
- Evidence: [`test_jt_array_float64_narrowing`](../../tests/debug/test_kernel_traps.py)
- Symptom: `jt.array` can narrow NumPy float64 and int64 inputs to 32-bit defaults,
  invalidating high-precision references or numerical gradient checks
- Workaround: always pass `dtype="float64"` or `dtype="int64"` when width is part
  of the contract
- Review/expiry condition: retain both default and explicit-dtype assertions until
  a public dtype-default decision changes them together

## KI-MEM-002: reading a device tensor relocates it to the host

- Severity: High
- Status: Reproduced, unfixed
- Owner: memory and executor maintainers
- Evidence: real CUDA, 10M float32 (40 MiB) on device 7.
  `b.numpy()` moves the storage: `location()` goes `device` -> `cpu`, and it is
  telling the truth. The next device operation migrates it back, measured at
  **0.1214s against 0.0006s for the same operation on a tensor already
  resident -- 215x** -- and device memory goes from `+40 MiB` to `+80 MiB`
  because both copies are live. `repr()` takes the same path, so printing a
  tensor at a REPL relocates it.
  Reading a *single element* is worse: `d[0].item()` leaves `d` itself on the
  host, so a one-element read moves 40 MiB off the device.
  `tolist()` is a third spelling of the same relocation, found by
  `tools/side_effect_probe.py` rather than by hand; it was not in this entry
  when the entry was written, which is the argument for the probe.
- Not this issue: a reduction result. `u.sum().item()` leaves `u` on the
  device, because the scalar is a new Var rather than a view of `u`. The common
  training-loop spelling `loss.item()` is therefore unaffected, and the defect
  should not be described as "reading anything moves it".
- Symptom: a read is a query, and this one mutates placement. Nothing in the
  API says so, no error is raised, and the cost lands on a later line -- the
  next device operation, which now pays a round trip. A `print` added while
  debugging permanently changes where the tensor lives.
- Divergence: PyTorch refuses `.numpy()` on a CUDA tensor and requires an
  explicit `.cpu()`; the tensor never moves as a side effect of being read.
- Workaround: `jt.array(x.numpy())` when a host copy is wanted, keeping `x`
  where it is; avoid `print(x)` and element indexing on large device tensors in
  hot paths.
- Review/expiry condition: `numpy()`, `repr()` and element/slice reads leave the
  source Var's `location()` unchanged on real CUDA, a device operation
  immediately after such a read costs the same as one without it, and
  `tests/core/test_var_residency_contract.py` covers all three spellings
  including the reduction case that must stay unaffected.

## KI-AUTOGRAD-003: register_hook makes the receiver misreport its residency

- Severity: Low
- Status: Reproduced, unfixed; the original severity was measured down
- Owner: autograd maintainers
- Evidence: real CUDA. `a = jt.ones((2048,2048)).cuda(); a.sync()` gives
  `a.location() == "device"`; `a.register_hook(lambda g: g)` then gives
  `a.location() == "none"`, the state of a Var whose data has not been produced.
- Not what it looks like: the data is still there. Device memory does not drop
  (`+32 MiB` before and after), the value reads back correct, and the next
  device operation costs the same as without the hook -- 0.9x, measured with an
  expensive forward (`A@B@A` at 2048 square) precisely so a cheap one could not
  hide a recompute. **Nothing is discarded and nothing is recomputed.**
- Symptom: `location()` and `device` answer `"none"` for a Var that is resident.
  Every other reading of them is trustworthy, so a caller who checks residency
  around a hook gets a wrong answer with no way to tell it apart from a genuine
  unmaterialized Var.
- Found by: `tools/side_effect_probe.py`, which compares a Var against a
  snapshot of itself across every public operation. It was not looking for this;
  the operation appears because it changes an input it was not asked to change.
  The probe reports *that* something changed, not what it costs -- the severity
  came from measurement afterwards, and the first reading of this entry claimed
  a recompute that measurement did not support.
- Workaround: none needed for correctness or speed; do not read `location()` or
  `device` immediately after registering a hook.
- Review/expiry condition: `register_hook` leaves `location()` unchanged on CPU
  and real CUDA, and the side-effect probe reports no mutation for it.

## KI-TEST-004: 28 ACL structure tests stopped executing and nobody read the report

- Severity: High (a whole backend's structural gates proved nothing for two days)
- Status: Partly fixed 2026-09-10 -- they execute again; the 28 mismatches they
  now report are open
- Owner: ACL backend maintainers
- Evidence: `tests/structure/backends/acl/test_acl_python_registration.py`
  collected 40 cases and executed **0** of them. Every one failed at fixture
  setup with:

  ```
  ModuleNotFoundError: No module named 'jittor._core'
  ```

- Cause: the fixture loads `python/jittor/_runtime/dispatch.py` by path --
  deliberately, so a static structure test does not pull the runtime in -- and
  builds a stub package tree in `sys.modules` for it to import from. On
  2026-09-08 (`7e83d6da4`) `dispatch.py` gained
  `from jittor._core.dtypes import ...` at module scope, and `jittor._core` was
  not one of the stubs. The module became unloadable in isolation, which is a
  property nothing had written down.
- Why it went unnoticed: the session report **said so** -- "files this session
  proved nothing about: test_acl_python_registration.py 0 skipped, 0 executed"
  -- and that report is marked "Reported only. Set
  `JITTOR_TEST_REQUIRE_EXECUTION=1` (the gates do) to make an unexplained entry
  fail the run." So the machinery for catching exactly this exists and works;
  what was missing is that a red suite makes one more red line invisible.
- Fix applied: the fixture stubs `jittor._core` and loads the real
  `python/jittor/_core/dtypes.py` by path alongside `dispatch.py`. The real
  module rather than a stub, because `dtypes.py` imports only `typing` at
  module scope -- its `jittor_core` references are lazy, inside functions -- so
  it costs nothing and the fixture stays honest about what it exercises.
  40 errors / 0 executed became **28 failed / 12 passed**.
- What the 28 are, and why they are left open: the ACL kernels moved to a
  structured attribute dictionary (`code_with_attributes(..., attributes=...)`,
  `backends/acl/kernels/ops/pool_op.py`) while the fixture's recorders still
  model the older positional `attr_code`, so they fail with
  `record_pool() got an unexpected keyword argument 'attributes'` and similar.
  The signature moved during the window in which these tests were not running.
  Reconciling them means deciding, per case, whether the test's expectation or
  the kernel is the stale one; that needs the ACL maintainers, and this machine
  has no NPU to settle a runtime question with.
- Same shape as KI-EXEC-002 and the roundtrip sweep's device defect: a refactor
  changed something no contract had written down, and a check went from
  asserting to asserting nothing. The three were found on the same day by
  asking one question of each gate -- *would this fail if the thing it checks
  were broken?*
- Review/expiry condition: the file reports 40 passed, or every remaining
  failure has an entry saying which side is wrong and why.

## KI-EXEC-002: the profiler cannot see work that `auto_flush_ops` already launched

- Severity: High (measurements are silently partial; two gates are permanently red)
- Status: Reproduced, unfixed
- Owner: executor and profiling maintainers
- Evidence: CUDA, a 64x64x64x64 float32 tensor sliced and concatenated, then
  differentiated. Same expression at each row; only the number of slices moves.

  | slices | rows `jt.profile_scope` reported | wall clock inside the scope |
  | --- | --- | --- |
  | 1, 2, 8 | 6-7 | 0.09-0.12s |
  | **16, 32, 64** | **0** | **0.0003-0.025s** |

  The result is correct at every row -- `b.numpy().sum()` is right -- so the
  work happened. It happened *before the scope opened*.
- Cause, established without a rebuild by moving one flag:

  ```
  auto_flush_ops=128  slices=64  rows=0  total=0
  auto_flush_ops=0    slices=64  rows=9  total=37740746
  auto_flush_ops=128  slices=32  rows=0  total=0
  auto_flush_ops=0    slices=32  rows=9  total=17530765
  ```

  `auto_flush_ops` (`src/core/executor.cc`, default 128, CUDA only) launches
  everything pending once that many operators have been created since the
  executor last ran, so the device computes while Python keeps building. It is
  a deliberate pipelining feature and it does what it says. What it also does
  is end the guarantee that a lazily built graph is still pending when the
  caller comes to run it: build more than ~128 operators' worth of graph and
  part of it has already executed, outside whatever scope the caller is about
  to open.
- Symptom: `jt.profile_scope` returns a report with no rows and no warning.
  Anything dividing by the total gets a zero -- which is how this was found:
  `tests/ops/test_concat_op.py::test_concat2_perf` and `::test_concat_perf`
  fail with `ZeroDivisionError` at every run, and have been doing so long
  enough that the failure reads as background noise.
- Why it matters beyond those two tests: the graphs worth profiling are the
  large ones, and those are exactly the ones that under-report. A profile that
  came back empty is indistinguishable from one that came back fast, and the
  report says nothing about the ops that were flushed before it started.
- Introduced 2026-09-02 (`c9176652f`), so this is a refactor-era regression
  rather than an old defect: the tests were written against fully lazy
  execution and the flag changed what "pending" means underneath them.
- Not the same as KI-EXEC-001, but the same shape and worth reading together:
  behaviour that changes once a graph passes a size threshold, where nothing in
  the API says a threshold exists.
- Workaround: `jt.flag_scope(auto_flush_ops=0)` around graph construction *and*
  execution. Setting it inside the profile scope alone does not help -- by then
  the flush has already happened.
- Review/expiry condition: a profile taken over a graph of any size either
  accounts for every operator that ran, or says out loud that it did not; and
  the two concat perf cases measure something again rather than dividing by
  zero.

## KI-EXEC-001: CUDA segfaults past a graph-size threshold

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: executor and CUDA backend maintainers
- Evidence: pure Jittor, real CUDA, no compatibility layer. A chain of
  bottleneck blocks (1x1 -> 3x3 -> 1x1 with a downsample, 512 -> 1024 channels,
  8x8 input) segfaults at **five blocks and crashes for six and seven; four is
  fine**. CPU is fine. `use_parallel_op_compiler=0` still crashes, so this is
  not KI-COMPILER-001. Symbolised backtrace:
  `run_exec_plan` <- `Executor::run_sync` <- `Executor::submit_pending`.
- Blast radius: ResNet50-class backbones do not run on CUDA. Two independent
  downstream projects hit it separately -- JSeg and JDet both crash with a
  ResNet50 backbone while JSeg's ResNet18 passes -- and per-stage bisection
  points at layer3 (six blocks, 1024 channels). A single bottleneck and stacks
  of four are fine, so the trigger is graph size, not the block itself.
- Not caused by the same-day `device_copy` fix (`715009c02`), which touches
  `run_exec_plan`: reverting its three hunks and rebuilding still segfaults at
  five blocks. The defect predates it.
- Reproduction: `$JITTOR_LAB_ROOT/_state/segv/repro.py <n>` (unversioned);
  `n=4` prints the output shape, `n=5` dies. Build the core with
  `addr2line_path=$(which addr2line)` to get the frames above.
- Workaround: none for CUDA. Shorter backbones (ResNet18) work; CPU works.
- Review/expiry condition: the repro passes for n in 4..8 on real CUDA, both
  downstream ResNet50 backbones run a forward and backward, and a regression
  covers a chain long enough to have crashed.

## KI-OPS-010: fixed -- an index arriving in a Var is now checked against the dimension

- Severity: was Critical (memory safety and silent wrong answers)
- Status: Fixed 2026-09-10
- Symptom it had: a length-5 float32 source, one index, CPU:

  | index | `take` | `gather` | `index_select` |
  | --- | --- | --- | --- |
  | 99 | `0.0` | `0.0` | `0.0` |
  | 100,000 | `0.0` | `0.0` | `0.0` |
  | 100,000,000 | **segfault** | **segfault** | **segfault** |
  | 2,000,000,000 | **segfault** | — | — |

  NumPy and Torch both raise `IndexError` for every row above. `setitem` shared
  the hole and was worse: it *wrote* past the buffer, so an out-of-range index
  corrupted the heap and surfaced somewhere else entirely.
- Cause: the check was in the wrong place, not missing everywhere. A Python
  `int` index is normalised and range-checked while the op is being built
  (`getitem_op.cc`, `User check failed: v>=0`), and a slice clamps to the tensor
  the way NumPy does. An index arriving in a **Var** took neither path: the
  kernel wrapped negatives (`if (iid@d < 0) iid@d += ishape@d;`) and then read,
  with nothing between. `take`, `gather`, `index_select` and every embedding
  lookup funnel into exactly that expression, and their indices come from *data*
  -- token ids, class ids, offsets -- so an out-of-range value is a malformed
  dataset or an off-by-one, not a programming exotic.
- Fix: `src/ops/composite/index_bounds.h` normalises and validates one index,
  and both `getitem_op.cc` and `setitem_op.cc` route the Var-index expression
  through it. The loop cannot raise from inside itself -- OpenMP region on CPU,
  kernel on CUDA -- so the two devices report differently: CPU clamps the
  offending index, records it, and the host raises after the loop (the clamp is
  what keeps the read inside the buffer until then); CUDA prints the index and
  traps, which is the bargain PyTorch makes for its own device-side asserts.
- What the first attempt got wrong: the post-loop check was emitted for both
  devices, and `cuda_indexing_optimize` takes `func->children.back()` to be the
  loop nest. It moved the check into the kernel in the loop's place and
  rewrote it as a loop, so six of eleven CUDA cases "raised" -- a **false
  green**, since they were failing in codegen rather than on the index. The
  check is now guarded with `@if(@is_def(JIT_cpu), ...)`, and
  `backends/cuda/kernels/core/indexing_codegen.cc` asserts its own structural
  assumption with a message that names it instead of `l->inner.size() == 3`.
- Cost: an in-kernel compare per indexed element. A separate measurement of the
  weaker alternative -- one extra reduction over the index tensor in the graph
  -- came to 13.3% on an 8192-index lookup into a 50000x256 table. An earlier
  note in this file claimed 276,183% for a host-side pre-check; that figure was
  wrong and has been withdrawn. It timed `.item()`, which is a synchronisation
  wait, not the check.
- Regression: `tests/ops/test_index_bounds.py`, 10 cases over both devices --
  `getitem`, `gather`, `index_select`, 2-D row indexing, `setitem`, and the
  backward pass (which scatters through `setitem` and so needs its own case).
  It asserts the message names the offending index, not merely that something
  failed, and it pins the two paths that were already right: a Python `int`
  index still raises and `x[2:99]` still clamps. The CUDA out-of-range case
  runs in a subprocess because a device trap takes the context with it.

## KI-BACKEND-007: CUDA `std`/`norm` return a small finite number instead of NaN

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: CUDA backend and reduction maintainers
- Evidence: no exotic input needed -- one NaN among ordinary numbers:

  ```
  jt.std([nan, 1.0, 2.0])      CPU nan      CUDA 0.0009999999310821295
  ```

  With `[nan, inf, -inf, 0.0, -0.0, 1.0, -1.0, 1e-45, 3.0, 3.0]`: `std` gives
  `nan` on CPU and `0.001` on CUDA; `norm` gives `nan` on CPU and `1e-15` on
  CUDA. NumPy agrees with CPU in both cases.
- Symptom: a NaN anywhere in the tensor is absorbed and the result is a small
  finite number. `0.001` and `1e-15` look like an epsilon the implementation
  adds for numerical safety, which the NaN path collapses onto.
- Why this is the dangerous shape: `std` is what normalisation layers compute.
  When a NaN appears in activations, CPU propagates it and the run stops with an
  obvious symptom; CUDA returns ~1e-3, the normalisation divides by it, and the
  model produces enormous finite values instead. The training diverges for a
  reason that no longer points at the NaN, on the device people actually train
  on.
- Same family as KI-BACKEND-004 (CUDA `maximum`/`minimum` swallow NaN). That
  entry is about a binary op; this is a composed reduction, so the suppression
  is not confined to one expression-table row and a fix has to be checked
  against both.
- Found by: `tools/adversarial_device_sweep.py`, comparing every OpInfo operator
  between CPU and CUDA on inputs built from NaN, both infinities, both signed
  zeros and a subnormal. Seven operators disagreed; `std`, `norm` and
  `lgamma` (which returns `inf` on CUDA for a subnormal where CPU gives the
  correct 103.28) are the ones triaged so far.
- Workaround: check for NaN explicitly before normalising on CUDA.
- Review/expiry condition: `std` and `norm` return NaN on both devices whenever
  the input contains one, a parity case covers a NaN-bearing reduction, and the
  remaining four operators from that sweep are triaged.

## KI-BACKEND-008: fixed -- the flush-to-zero decision is written down and asserted

- Severity: was Medium (undocumented device divergence)
- Status: Fixed 2026-09-10 -- what was missing was the statement, not a change
- Evidence, measured on both devices and both policies:

  | value | CPU | CUDA default | CUDA `strict` | NumPy |
  | --- | --- | --- | --- | --- |
  | `1e-45` (subnormal) | kept | **0.0** | kept | kept |
  | `1e-40` (subnormal) | kept | **0.0** | kept | kept |
  | `1e-30` (normal) | kept | kept | kept | kept |

  It reaches past the value: the input is already zero when the function sees
  it, so `log(1e-45)` is `-103.28` on CPU and `-inf` on CUDA default, and
  `count_nonzero([1e-45])` is 1 against 0. `lgamma` was the original symptom;
  `log` reproduces it and is in the regression.
- Cause: nvcc's `--use_fast_math` implies `-ftz=true`.
- What was done: `jt.flags.cuda_kernel_math = "strict"` already existed
  (`src/runtime/jit_policy.cc`) and **fully restores subnormals** -- measured,
  not assumed, and the `strict` column above is the measurement. The default
  is unchanged. What is new is that the behaviour is now stated in
  `docs/notes/float32-precision-policy.md` and asserted by
  `tests/backends/parity/test_subnormal_contract.py`, so a change to it is
  visible instead of surfacing as a parity mismatch someone has to diagnose.
- Cost of turning it off, RTX 4090, two interleaved rounds, minimum of five:

  | 16M float32 | default | strict |
  | --- | --- | --- |
  | `divide` | 218.6 us | 218.8 us |
  | `sqrt` | 148.1 | 148.8 |
  | `exp` | 150.0 | 150.2 |
  | `log` | 148.1 | 148.2 |
  | `mul-add` (control, unaffected by fast-math) | 218.6 | 218.6 |

  **Not measurable at this scale**: 0.1-0.8%, the same spread as the control.
  These kernels are memory-bound, so the ALU cycles fast-math saves are already
  hidden behind the loads.

  The compute-bound case is **unresolved, and is recorded that way**. Chaining
  `exp/log/sqrt` eight and thirty-two deep, the spread between two runs of the
  *same* policy (1.01e-3 against 4.16e-4, a factor of 2.4) was larger than the
  spread between the two policies -- fusion variance, not the flag. So the
  honest statement is that no cost was measured, not that there is none. A
  usable number needs a benchmark whose fusion shape is pinned.
- Why the default was left alone: nothing here says flush-to-zero is the wrong
  trade, and changing a process-wide numeric default on the strength of a
  measurement that could not resolve the compute-bound case would be the same
  mistake in the other direction.
- Regression teeth: the two policies must actually differ for the file to pass.
  Pointing the `strict` case at `default` turns it red --
  `AssertionError: 0.0 == 0.0 : cuda_kernel_math='strict' did not stop the
  flush for 1e-45`. A build where the policy switch did nothing cannot satisfy
  it, which is the failure mode a one-sided "CUDA flushes" assertion would miss.

## KI-OPS-011: CPU `digamma` returns -inf where NaN and +inf are correct

- Severity: Medium
- Status: Reproduced, unfixed
- Owner: operator maintainers
- Evidence: against `scipy.special.digamma` as the reference:

  | input | CPU | CUDA | scipy |
  | --- | --- | --- | --- |
  | `nan` | **-inf** | `nan` | `nan` |
  | `-0.0` | **-inf** | `inf` | `inf` |

  CUDA is right in both rows and CPU is wrong; the other eight inputs agree.
- Symptom: `digamma` of a NaN produces a finite-signed infinity rather than
  propagating the NaN, so a NaN entering here is converted into a value that
  looks like a legitimate pole. At `-0.0` the sign of the pole is inverted.
- Notable for the direction: every other divergence this sweep found had CUDA
  as the wrong side. Recorded because "CPU is the reference" is an assumption
  the parity suite makes, and this is a counter-example to it.
- Review/expiry condition: CPU `digamma` matches scipy for NaN and both signed
  zeros, and the probe's device-agreement case covers it.

## Where the device divergences are, and where they are not

The adversarial sweep was run twice over the same 231 operators with two
orthogonal input vectors:

| vector | probes | operators compared | disagreements |
| --- | --- | --- | --- |
| non-finite (NaN, both infinities, both signed zeros, a subnormal) | what the format treats specially | 225 | **7** |
| magnitude (1e30 against 1e-30, 2^24 and 2^24+1, both integer extremes, exact ties) | what the arithmetic loses | 44 | **0** |
| half (float16 either side of its overflow at 65504 and its subnormal edge at 6.1e-5) | what a narrower format reaches sooner | 7 | **0** |

Every CPU/CUDA divergence found so far lives in the first row. Precision,
cancellation, integer-boundary, tie handling and float16's own limits agreed on
both devices everywhere they were compared.

The two zeros are weaker evidence than the seven, and the reason is worth
writing down rather than glossing: the magnitude and half sweeps compared far
fewer operators (44 and 7 against 225), because most entries reject those
inputs outright and are recorded as unprobed. They are a real signal about
where the divergences are *not*, but they are not a clean bill of health for
those dimensions.

That is worth stating because it narrows the problem: this is not a general
numerical-quality gap between the backends. It is specifically that **neither
backend has a written contract for the values IEEE-754 treats as special**, and
each kernel author picked whatever the obvious spelling did with them. The
entries below -- 004, 007, 008, and OPS-011 -- are all instances, and so is
KI-BACKEND-005 on the CPU side.

The second vector's zero is a result, not an absence of testing: it says the
next defect of this kind is more likely to be found by adding another
special-value case than by adding another magnitude case.

## Three CPU float defects share one surface; one is still open

`KI-BACKEND-004`, `KI-BACKEND-005` and `KI-BACKEND-006` were found separately
and read as three bugs. They are three symptoms of one thing: **the CPU kernel
build never decided what its floating-point contract is.**

- 005 was the compile flag. `-Ofast` promised the compiler that infinities and
  NaN do not occur, and it optimised on that promise. Fixed 2026-09-10: kernels
  build at `-O3`, at no measured cost, and the fused-versus-unfused divergence
  went with it.
- 004 is the expression table. `std::max` and `::max` were each chosen for
  being the obvious spelling, and their NaN behaviour -- accidental on CPU,
  deliberate IEEE `maxNum` on CUDA -- was never part of the choice.
- 006 was the reduction shape. A single serial accumulator is what you write
  when accuracy at scale is not a stated requirement. It is fixed and its entry
  is gone; see
  [the result report](../../refactor-wip/results/2026-09-10-cpu-reduction-blocked-pairwise.md).

None of the three is a coding mistake. Each is a reasonable local decision
taken without a written contract to check it against, which is why they
accumulated quietly and why fixing them one at a time will not stop the next
one: the same gap produces the same class of defect again.

What is missing is a statement of what CPU float32 promises -- IEEE semantics
for infinities and NaN, and an accuracy bound for reductions that does not grow
with size. 006 supplies the second half of that as something executable rather
than prose: `tests/ops/test_reduce_accuracy.py` asserts the *shape of the
growth* -- the relative error at 16M within a small factor of the error at
65,536, and within a small factor of NumPy's on the same input -- so it says
what "accurate enough" means without a threshold that turns red on a different
machine. The probe categories added alongside these entries (`device-agree`,
`stability`, `float-edge` in `tools/semantic_divergence_probe.py`) are the rest
of that gate in draft; they are what found all three, and the CPU `stability`
mismatch they reported for 006 is now clear.

One more thing they have in common, and on the one that has been done it is now
measured rather than predicted: **006 was free.** Blocked accumulation with a
pairwise fold is **3.7x-4.9x faster** than the serial loop it replaced (4.9 ->
18.4 GB/s at 64M float32 on the reduction benchmark) *and* leaves the
worst-case relative error at 16.7M elements at 6.0e-7 instead of 1.5e-1 --
below NumPy's own 1.6e-5. The one cost found was JIT compile time on large
fused reduction kernels, +18% after the emitted code was scaled to the body.
The assumption that correctness here costs speed is what made all three easy to
defer, and where it has been tested it was not true.

## KI-BACKEND-004: CUDA `maximum`/`minimum` swallow NaN while CPU propagates it

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: CUDA backend and operator maintainers
- Evidence: `f = [nan, -inf, -0.0, 0.0, inf]` against zeros, float32:
  CPU gives `[nan, 0.0, -0.0, 0.0, inf]`, CUDA gives `[0.0, 0.0, 0.0, 0.0, inf]`,
  NumPy gives `[nan, 0.0, 0.0, 0.0, inf]`. The same expression on the same input
  disagrees between the two devices.
- Symptom: a NaN entering `maximum`/`minimum` disappears on CUDA. A model that
  starts producing NaN shows it on CPU and not on the GPU, which is the wrong
  way round for where people train. This is worse than either convention alone:
  a device-parity check comparing CPU against CUDA would flag it, and none does
  because no parity case feeds NaN.
- Cause: `src/type/common_op_type.cc` maps `maximum` to `::max(...)` for CUDA
  and `std::max(...)` for CPU. CUDA's overload resolves to `fmaxf`, whose IEEE
  `maxNum` semantics deliberately return the non-NaN operand; `std::max` is
  `a<b ? b : a`, and comparison against NaN is false, so the first operand --
  the NaN -- comes back by accident. Neither was chosen for its NaN behaviour.
- Also visible there: `maximum(-0.0, 0.0)` gives `-0.0` on CPU and `0.0` on
  CUDA; NumPy gives `0.0`. Same root, smaller consequence.
- **The reduction is worse than the elementwise case, and this entry had it
  wrong.** Measured 2026-09-10 on both devices, `n` = 5, 4096 and 1,048,576,
  one NaN among ones:

  ```
  jt.max(x)   CPU 1.0   CUDA 1.0   NumPy nan
  jt.min(x)   CPU 1.0   CUDA 1.0   NumPy nan
  ```

  So CPU does *not* propagate NaN in general -- it propagated in the evidence
  above only because `std::max(a, b)` is `a<b ? b : a` and the NaN happened to
  be the **first** argument. A reduction accumulates `tmp = std::max(tmp, b)`,
  where an incoming NaN is always the *second* argument, so it is discarded on
  every device at every size. `x.max()` is a common way to ask whether a tensor
  has gone bad; it cannot see a NaN at all.
- Entangled with KI-OPS-006: both entries want a NaN-aware max, and the same
  table row feeds the reduction, whose parallel passes match the literal
  `std::max(T(a),T(b))` / `::max(...)` spelling to route to atomics. A fix has
  to satisfy the elementwise case and the reduction together; the measurement in
  KI-OPS-006 (elementwise 1.2-1.6x, reduction 7.1-7.5x) says they cannot be
  treated as one change.
- Workaround: test for NaN explicitly before a max/min on CUDA where its
  presence matters.
- Was blocked on KI-BACKEND-005, now unblocked. A NaN-propagating max is
  written `a != a ? a : ...`, and `-Ofast` implied `-ffinite-math-only`, under
  which the compiler is free to fold `a != a` to false -- the fix would have
  read correct and compiled to the old behaviour, which is worse than not
  writing it. Kernels now build at `-O3`, and `x != x` on a NaN comes back 1.0,
  pinned by `tests/ops/test_ieee_arithmetic.py`.
- Review/expiry condition: CPU and CUDA agree with NumPy on NaN and on the sign
  of zero for `maximum` and `minimum`, **and for `jt.max`/`jt.min` over an
  array containing one**, and a device-parity case feeds NaN so the
  disagreement cannot return unnoticed.

## KI-BACKEND-005: fixed -- CPU kernels build at `-O3`, not `-Ofast`

- Severity: was Critical (silently wrong arithmetic, irreproducible results)
- Status: Fixed 2026-09-10
- Symptom it had: CPU, float32, vectors of length >= 4 (the vectorised path):

  | expression | before | IEEE / NumPy / Jittor CUDA |
  | --- | --- | --- |
  | `1 / 0` | `nan` | `inf` |
  | `-inf / 0` | `nan` | `-inf` |

  A single element computed correctly; the wrong answers began at length 4, so
  a scalar spot-check saw nothing.
- Cause: `python/jittor/build/compiler.py` appended `-Ofast` to
  `kernel_opt_flags` unconditionally. `-Ofast` implies `-ffast-math`, which
  implies `-ffinite-math-only` -- a promise that no operand is ever infinite or
  NaN. Operands that *were* infinite then took whatever path the transformed
  code happened to produce. The project already knew in one place: `nan_checker`
  had `-Ofast` stripped and `-O2` substituted, and `jt.misc._simple_for` exists
  to compile the `isnan`/`isinf` kernel at `-O2`. The workaround was applied
  where the problem was noticed rather than where it applied.
- Fix: `-O3`. One line, and the reason it is not a trade is that the
  reassociation `-ffast-math` also granted was not being used: g++ 12.3 does
  not vectorise the real reduction kernels, because the runtime
  `storage_stride(0)` blocks it (measured while fixing KI-BACKEND-006).
  Accuracy at scale is now `BlockedReductionPass`'s job, stated in the code
  rather than bought from a flag that also breaks arithmetic.
- Measured cost: none. Same machine, same warm cache, four kernels:

  | | `-Ofast` | `-O3` |
  | --- | --- | --- |
  | elementwise chain (4M) | 0.000320s | 0.000324s |
  | exp/sqrt chain (4M) | 0.000747s | 0.000679s |
  | `sum` (4M) | 0.000737s | 0.000719s |
  | matmul 512 | 0.659172s | 0.660539s |
  | IEEE table | **7/9** | **9/9** |

- What else stopped being wrong: `tools/fusion_consistency_sweep.py` on CPU
  went from one DIFFERENT to 12/12 IDENTICAL. That case was `(a + b) - a` with
  `a = -1e8, b = 2.0`, which gave `0.0` unfused and `2.0` fused because the
  larger expression was reassociated to `b + (a - a)`. Fusion depends on what
  else is in the graph, so the same code was giving different answers in
  different surroundings -- the property that made an earlier probe check
  unstable and forced it to be withdrawn. `tools/semantic_divergence_probe.py`
  on CPU went from 4 MISMATCH to 3, and the three that remain are all
  KI-BACKEND-004.
- Regression: `tests/ops/test_ieee_arithmetic.py` (ten expressions IEEE-754
  defines exactly, both devices, length 8 because length 1 passed even when the
  flag was wrong; plus `x != x` as a predicate, since `-ffinite-math-only` may
  fold a NaN test to false and then every hand-written check stops checking).
  Reverting the flag turns it red: `1 / 0 gave [nan ...], IEEE says inf`.
  `tests/structure/codegen/test_kernel_math_flags.py` names the flag, so a
  reintroduction says what was changed rather than only that arithmetic broke;
  it also asserts an optimisation level is still being asked for, since
  deleting the flag and putting nothing back would satisfy the first check by
  making things worse.
- Not covered by this fix: CUDA still compiles with `--use_fast_math`. That
  flag is about division, square root and transcendental accuracy rather than
  finite-math, and `jt.flags.cuda_kernel_math = "strict"` already exists to
  turn it off per process. Whether it should be the default is a separate
  question with its own measurement, and it is not answered here.

## KI-FFT-001: withdrawn -- current CUDA sequence regression is clean

- Severity: n/a
- Status: Withdrawn 2026-08-21; the old aggregate outcome is not reproducible on
  the current implementation
- Owner: FFT and Torch-compat maintainers
- Evidence: [CUDA rFFT sequence review](../../refactor-wip/results/2026-08-21-rfft-sequence-review.md)
  and
  [`test_rfft_after_complex_forward_backward_sequence`](../../compat/tests/torch/test_torch_compat_fft_einsum.py)
- What it claimed: complex forward/gradient work earlier in one CUDA process could
  remove the `rfft` imaginary half-spectrum and break `irfft(rfft(x))`
- Current finding: four fresh-process repetitions of the old aggregate sequence,
  the new deterministic CPU/CUDA regression, the full FFT compatibility module,
  and an aggregate comparison against independent binary PyTorch all produce the
  correct half-spectrum and round trip. `rfft` and `irfft_rfft` are hard failures
  again in the aggregate comparator rather than sequence-sensitive exemptions.
- Reopen condition: retain the deterministic regression and reopen only with a
  reproducible failing sequence, exact revision, device, cache isolation, and
  copied host results from both the spectrum and round trip

## KI-COMPAT-001: Torch namespaces publish native-only helpers and imported symbols

- Severity: Medium
- Status: Open, recorded in the Torch API manifest
- Owner: Torch compatibility frontend maintainers
- Evidence:
  [`tests/structure/torch_api_manifest.json`](../../tests/structure/torch_api_manifest.json)
  records all 291 native `Var` methods on `torch.Tensor`, 37 of which PyTorch's
  `Tensor` has no equivalent for -- `assign`, `start_grad`, `stop_grad`,
  `stop_fuse`, `reindex`, `reindex_reduce`, `reindex_var`, `migrate_to_cpu`,
  `migrate_to_gpu`, `fetch_sync`, `cast`, `float_auto`, `ceil_int`, `floor_int`,
  `round_int`, `safe_clip`, `debug_msg`, `peek`, `tape` and `candidate` among
  them -- plus `torch.nn.OrderedDict`, `torch.nn.deepcopy` and
  `torch.nn.partial`, which are imports leaking into a published namespace.
  `torch.random` is a third shape: a module subclass with `__call__`, so
  `torch.random(3)` returns a tensor while `torch.random` is also the published
  `torch.random` namespace; PyTorch's is a module only.
- Symptom: [`compat/torch/api_manifest.py`](../../compat/torch/api_manifest.py)
  states that native-only helpers copied into compatibility namespaces are not
  counted as PyTorch APIs, but they are public and reachable on the frontend, so
  downstream code can bind to them and `dir(torch.nn)` advertises them. They also
  sit in the Torch coverage denominator, where no Torch-facing test will call them.
- Workaround: do not treat a name's presence on `torch.*` as evidence that the
  Torch API has it; `compat/torch/api_manifest.py` holds the declared set.
- Review/expiry condition: keep the native-only names and the imported symbols
  out of the published namespaces, regenerate
  `tests/structure/torch_api_manifest.json` in the same commit, and remove this
  entry when the manifest no longer records them.

## KI-OPS-009: a broadcast index Var reads garbage on a CPU-only build

- Severity: Critical (out-of-bounds read *and write*)
- Status: Reproduced and narrowed 2026-09-10; root cause open. The segfault is
  gone -- it is now a catchable error that names the bad index.
- Owner: indexing and build-configuration maintainers
- **The title was too narrow.** This is not about `scatter_add`, and not about
  `setitem`. It is any indexing operation whose index Var came from a broadcast.
  Measured on a CPU-only build (`nvcc_path=""`), each row a fresh process:

  | index expression | result |
  | --- | --- |
  | `jt.zeros((4,5), 'int64')` | index `2697334449954054313`, out of bounds |
  | `jt.zeros((4,5), 'int32')` | index `-1887156860`, out of bounds |
  | `jt.zeros(...)` after `idx.sync()` | index `-356873746589917012`, out of bounds |
  | `jt.array(np.zeros((4,5), 'int64'))` | **20.0, correct** |
  | `jt.ones((4,5), 'int64') - 1` | **20.0, correct** |
  | `jt.array(...) + 0` | **20.0, correct** |
  | `jt.zeros(...)` through plain `setitem` | out of bounds |
  | `jt.zeros(...)` through `gather` (the read side) | out of bounds |

  The offending value differs every run, which is what reading unallocated
  memory looks like.
- What the split is: `jt.zeros(shape, dtype)` is
  `unary(0, dtype).broadcast(shape)` (`python/jittor/_core/var.py`). Every index
  that goes through *any* real computation is materialised as a side effect and
  works. So the index Var being a pure broadcast is the discriminator, not the
  dtype, not laziness -- **`idx.sync()` before the call does not help**, which
  rules out "the producer had not run yet".
- Build-specific, not device-specific: on a CUDA build with
  `jt.flags.use_cuda = 0` -- same device, same kernels' CPU path -- the same
  four lines give the right answer. Only `nvcc_path=""` fails. Whatever
  `HAS_ACCELERATOR`/`HAS_CUDA` changes about the indexing path or the registered
  optimisation passes is where the cause lives.
- Hypothesis tested and **rejected**: that the broadcast was being fused away
  and the kernel read an unallocated pointer. Setting `VarFlags::_stop_fuse` on
  every index Var in both `GetitemOp` constructors and `SetitemOp` -- the idiom
  `reindex_reduce_op.cc` uses for exactly this requirement -- changed nothing;
  all three cases still read garbage. The change was reverted rather than
  shipped with a confident comment. The next hypothesis worth testing is the
  *stride* the kernel derives for the index Var: a broadcast's storage is not
  laid out like a dense Var of the same logical shape, and the kernel computes
  `vp@d[... * vs@d@@s@j ...]` from the output shape.
- What improved: the index bounds check added the same day (KI-OPS-010) turns
  this from a segfault that takes the interpreter down into a
  `UserError` naming the offending index. That is how the values in the table
  above were obtained -- before it, the process died with no diagnosis. It does
  not fix the defect: the index is still wrong, the answer would still be wrong
  if it happened to land in range, and out-of-range values are merely no longer
  *written*.
- Why it stayed hidden: `tests/ops/test_ops.py` belongs to the Torch process
  mode, so a native `pytest tests/ops` never reaches it, and a CUDA-configured
  run never sees the failure. It needs a CPU-only build to appear, and the
  crash used to end the session, so every later test in it silently never ran.
- Review/expiry condition: the table above is all-correct on a CPU-only build,
  a regression covers a broadcast-produced index on both build configurations,
  and the reason a CUDA-less build differed is written down.

## KI-TEST-002: a dead session is indistinguishable from a short one

- Severity: High
- Status: Open
- Owner: test infrastructure maintainers
- Evidence: on a CPU-only build (`nvcc_path=""`, the configuration
  `tools/run_test_suite.py` sets), the maintained Torch selection printed
  progress to 48% and then ended with exit 1 -- no result line for the case that
  died, no traceback, no summary, and no mention of the ~2200 tests that never
  ran. Four cases do this on that build:
  `tests/ops/test_ops.py::TestCommonCPU::test_reference_scatter_add_float32`
  (KI-OPS-009),
  `tests/ops/test_ops.py::TestGradientsCPU::test_gradcheck_interpolate_bilinear`,
  `compat/tests/torch/test_torch_hf_alias.py::TestTorchHFAlias::test_small_transformers_forward_direct_alias`,
  and
  `compat/tests/torch/test_torch_hf_models.py::TestTorchHFModels::test_generate_greedy_kv_cache_and_beam`.
  On a CUDA build the last two report `1 failed` and the session continues, so
  the individual crashes are configuration-specific -- the reporting hole is not.
- Symptom: a session that dies mid-run reads like a session that ran fewer
  tests. This is the same disease as an entry that only ever skips, one degree
  worse: a skip at least leaves a line in the summary. Any exclusion taken to
  work around it -- this is how the first Torch coverage baseline had to be
  gathered -- then silently narrows what the gate covers.
- Workaround: compare the executed count against the collected count, or run the
  selection in parts and check that each part produced a summary line.
- Review/expiry condition: a session whose process ends before pytest writes a
  summary is reported as a failure naming the case that was running and the
  number of tests that never executed.

## KI-TEST-003: the coverage wrapper is visible to the Torch identity contracts

- Severity: Medium
- Status: Open, measured and excluded rather than hidden
- Owner: test infrastructure maintainers
- Evidence: with `JITTOR_API_COVERAGE=1` on the Torch surface, 86 cases in the
  14 files listed in `tests/_helpers/api_coverage.py::IDENTITY_CONTRACT_FILES`
  fail that pass with it off -- `assertIs(torch.addmm,
  installers.numerical.addmm)` and the object-keyed fidelity registry lookups.
- Symptom: the wrapper records a call by replacing the published object, and the
  Torch frontend contracts that the published object *is* the one its owner
  module holds. Two ways of hiding were measured and both made it worse:
  rebinding the defining module took the count from 86 to 101, and rebinding
  every alias took it to 97 with the failures moving from `assertIs` to the
  registry. The native surface states no such contract and is unaffected
  (`tests/ops/test_where_op.py` is 18 passed with the diagnostic on and off).
- Workaround: a Torch coverage run excludes those 14 files, and the exclusion is
  written into `tests/structure/torch_api_coverage_baseline.json` so the looser
  set is not read as a complete result.
- Review/expiry condition: record calls without replacing anything -- a
  `sys.setprofile` hook keyed by code object observes the same calls and mutates
  nothing -- then delete the exclusion list and re-take the Torch baseline. The
  characterisation case
  `tests/structure/test_api_coverage_helper.py::test_the_wrapper_is_visible_to_an_identity_contract`
  is where that change reports success.

## KI-COMPLEX-001: native complex capability gaps

- Severity: Research/High by operation
- Status: Explicit unsupported contracts
- Owner: dtype, autograd, and linear-algebra maintainers
- Evidence: [native complex dtype decision](../../docs/notes/complex-dtype.md)
- Gaps: CUDA complex `prod`, second-order complex autograd/JVP, complex128,
  native complex linear-algebra kernels, and some CUDA eig environments
- Review/expiry condition: remove each sub-item only with focused CPU and
  accelerator tests for its operation and derivative order

## KI-LOG-001: withdrawn -- log capture is not broken in CUDA builds

- Severity: n/a
- Status: Withdrawn 2026-08-20; the original diagnosis was an artefact of the
  probe, not of the build
- What it claimed: that a CUDA-enabled core captured almost no operator logging
  compared with a CPU-only one (12 lines against 328 for the same expression)
- Why that was wrong: the probe evaluated the graph with `.data`, and
  `VarHolder::data()` only syncs when the Var does not already hold host
  memory, so in one build the work happened before the capture window opened
  and in the other inside it. Measuring the same expression with an explicit
  `jt.sync_all()` inside the window gives 259 captured lines in the CUDA build
  against 260 in the CPU-only one, with `fused_op.cc` at 4 in both and
  `executor.cc` at 16 against 15. An unconditional `LOGi` compiled into
  `Executor::run_sync` is captured in both.
- What the affected tests really showed: the numerical parallel-pass defect was
  fixed for CPU in `137f9dd1`. A follow-up found that applying that CPU source
  rewrite to CUDA double-accumulated launch bit boundaries and left large fused
  outputs partially unwritten. `0a3458b3` limits the rewrite to `JIT_cpu`; the
  complete CPU and CUDA gates and compact network parity now pass. See the
  [parallel-range follow-up](../../refactor-wip/results/2026-08-22-cuda-parallel-range-network-oracle.md).
- Lesson for the next probe: never use `.data` to force evaluation inside a
  `log_capture_scope`; call `jt.sync_all()` and keep a reference to the Var.

## KI-TEST-001: fixed -- device tests now restore `use_cuda` instead of zeroing it

- Severity: was Medium (test isolation)
- Status: Fixed 2026-08-20
- Symptom it had: `tests/ops` reported 127 failed / 105 passed / 26 errors as a
  single process against 51 failed / 200 passed one file at a time. Later files
  failed with `Op array doesn't have cuda version`, the signature of a Var built
  for one device being evaluated on the other.
- Cause: every CUDA test class ended with `jt.flags.use_cuda = 0` rather than
  restoring the previous value. On a machine with a GPU the default is 1, so the
  first such class switched the accelerator off for the rest of the process --
  across files, the flag being process-global. The most-used copy was in
  `tests/_helpers/devices.py::cuda_test_case`, shared by many classes.
- Fix: remember `use_cuda` in setUp and put it back in tearDown, after a
  `jt.sync_all()` so the pending graph drains under the device it was built for.
- Effect: `tests/ops` as one process went to 49 failed / 202 passed, matching
  the per-file result; `tests/backends` from 28 failed to 25. Runs take longer
  now because tests that had been silently running on CPU do use the GPU and
  compile its kernels once.

## KI-COMPILER-004: fixed -- a CPU-only core no longer shadows the CUDA build

- Severity: was High (silently disabled the accelerator for a whole run)
- Status: Fixed 2026-08-20 in `python/jittor/compiler.py`
- Symptom it had: the cache holds `<cache>/2.0/jittor_core...so` built without
  CUDA and `<cache>/2.0/<cuda key>/jittor_core...so` built with it. Both were
  added to `sys.path` with `append`, and the parent went on first, so the
  CPU-only build won every import. Any run without nvcc creates that file, and
  from then on every process sharing the cache ran on CPU, with each CUDA
  operator failing "Op ... doesn't have cuda version". A three-hour Torch-mode
  suite ran that way before this was found.
- Fix: insert the CUDA cache directory ahead of the plain one instead of
  appending after it. Verified by planting a CPU-only core and confirming the
  CUDA build is still the one imported.
- Effect: `tests/structure` went from 3 failed / 209 passed to 212 passed.
- Guard: [cache path precedence](../../tests/build/test_cache_path_precedence.py)
