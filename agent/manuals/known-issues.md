# Active Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-09-09
- Baseline: `28e61e669`
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

## KI-OPS-009: scatter_add segfaults on a CPU-only build

- Severity: Critical
- Status: Reproduced, unfixed; CPU-only builds only
- Owner: indexing/scatter and build-configuration maintainers
- Evidence: at `65a220d53`, two freshly built cores (215 and 216 objects
  compiled from scratch, so not a stale cache), same four lines, same machine:

  ```
  # nvcc_path="" -- the CPU-only build tools/run_test_suite.py configures
  nvcc_path="" PYTHONPATH=<repo>/python python -c "
  import jittor as jt
  x = jt.zeros((4,5)); idx = jt.zeros((4,5), dtype='int64'); src = jt.ones((4,5))
  print(x.scatter_add(0, idx, src).numpy().sum())"
  # Caught segfault at address 0x... / Segfault, exit   (exit 1)

  # the same command with nvcc on PATH
  # 20.0                                                (exit 0)
  ```

  `tests/ops/test_ops.py::TestCommonCPU::test_reference_scatter_add_float32`
  follows the same split: the process dies on the CPU-only build and the case is
  `1 passed` on the CUDA build. The Torch spelling
  (`torch.zeros(4,5).scatter_add(...)`) behaves identically, so it is a core
  defect and not a frontend one.
- Symptom: the crash takes the interpreter with it, so on a CPU-only build the
  OpInfo case ends the pytest process after printing its nodeid -- no result, no
  traceback, no summary -- and every later test in that session silently never
  runs. `tests/ops/test_ops.py` belongs to the Torch process mode, so a native
  `pytest tests/ops` never reaches it, and a CUDA-configured run never sees it.
- Workaround: on a CPU-only build do not call `scatter_add`; `-k "not
  scatter_add"` to complete a Torch session. `tests/ops/test_ops.py -k "not
  interpolate"` is needed for the same reason and is probably the same defect.
- Review/expiry condition: the four-line reproducer returns the summed tensor on
  a CPU-only build, the OpInfo case passes there, and a build-configuration
  difference of this size is either explained or gated.

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
