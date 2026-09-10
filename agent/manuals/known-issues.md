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

## KI-OPS-010: the indexing family does not bounds-check, and reads out of memory

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: operator and memory-safety maintainers
- Evidence: a length-5 float32 source, one index, CPU:

  | index | `take` | `gather` | `index_select` |
  | --- | --- | --- | --- |
  | 99 | `0.0` | `0.0` | `0.0` |
  | 100,000 | `0.0` | `0.0` | `0.0` |
  | 100,000,000 | **segfault** | **segfault** | **segfault** |
  | 2,000,000,000 | **segfault** | — | — |

  NumPy and Torch both raise `IndexError` for every row above.
- Symptom: no bounds check at all. A modestly out-of-range index reads whatever
  is mapped after the tensor and returns it as a value -- `0.0` here, which is
  the most plausible wrong answer there is. A large one reads unmapped memory
  and takes the process down.
- Why it matters more than the numbers suggest: `gather` and `index_select` are
  how embeddings are looked up, how attention gathers, how labels are indexed.
  Their indices come from *data* -- token ids, class ids, offsets -- so an
  out-of-range index is a malformed dataset or an off-by-one, not a programming
  exotic. The two outcomes are a silently wrong training signal, or a crash with
  no Python traceback.
- Found by: `tools/adversarial_device_sweep.py`, which ran every OpInfo operator
  on inputs containing NaN and both infinities. Cast to integers those become
  huge indices, so the sweep segfaulted -- and the first version of the sweep
  could not say which operator did it, because it did not record progress per
  operator. That is the same lesson `tools/side_effect_probe.py` records.
- Workaround: validate indices before a gather. `jt.clamp(idx, 0, n-1)` makes
  the read safe but silently changes the result, so it is a stopgap, not a fix.
- Review/expiry condition: all three raise for an out-of-range index on CPU and
  CUDA, none can be made to read unmapped memory from Python, and a regression
  covers a modest and an extreme index for each.

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

## KI-BACKEND-008: CUDA flushes subnormals to zero, CPU keeps them

- Severity: Medium
- Status: Reproduced; a documented decision is what is missing, not a fix
- Owner: CUDA backend maintainers
- Evidence: float32, smallest normal is `1.18e-38`.

  | value | CPU | CUDA | NumPy |
  | --- | --- | --- | --- |
  | `1e-45` (subnormal) | kept | **0.0** | kept |
  | `1e-40` (subnormal) | kept | **0.0** | kept |
  | `1e-30` (normal) | kept | kept | kept |

  Two of the seven operator disagreements found by the adversarial sweep have
  this single cause: `count_nonzero` counts `1e-45` on CPU and not on CUDA, and
  `lgamma(1e-45)` gives the correct `103.28` on CPU and `inf` on CUDA -- the
  input reached the function already flushed to zero, and `lgamma(0)` is `inf`.
- Symptom: a gradient that underflows into the subnormal range is exactly zero
  on CUDA and a tiny non-zero on CPU. The two devices then take different
  update paths for the same model, which is invisible until someone compares
  them.
- This one is a decision, not a mistake: flush-to-zero is the normal CUDA
  trade -- subnormal arithmetic is slow, and most training does not care. What
  is missing is that the decision is nowhere written down, so it reads as a
  defect when a parity comparison hits it, and the two devices are documented
  as equivalent when they are not.
- Fix direction: state it. Either document flush-to-zero as CUDA's contract and
  make the parity suite tolerate it explicitly, or disable it (`-ftz=false`) and
  measure the cost. Silently differing is the only option that should be off
  the table.
- Review/expiry condition: the subnormal behaviour of each backend is stated in
  the backend documentation, and the parity suite either asserts agreement or
  names this as a known and accepted difference.

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

## Three CPU float defects share one surface

`KI-BACKEND-004`, `KI-BACKEND-005` and `KI-BACKEND-006` were found separately
and read as three bugs. They are three symptoms of one thing: **the CPU kernel
build never decided what its floating-point contract is.**

- 005 is the compile flag. `-Ofast` promises the compiler that infinities and
  NaN do not occur, and it optimises on that promise.
- 004 is the expression table. `std::max` and `::max` were each chosen for
  being the obvious spelling, and their NaN behaviour -- accidental on CPU,
  deliberate IEEE `maxNum` on CUDA -- was never part of the choice.
- 006 is the reduction shape. A single serial accumulator is what you write
  when accuracy at scale is not a stated requirement.

None of the three is a coding mistake. Each is a reasonable local decision
taken without a written contract to check it against, which is why they
accumulated quietly and why fixing them one at a time will not stop the next
one: the same gap produces the same class of defect again.

What is missing is a statement of what CPU float32 promises -- IEEE semantics
for infinities and NaN, and an accuracy bound for reductions that does not grow
with size -- and a gate that holds the build to it. The probe categories added
alongside these entries (`device-agree`, `stability`, `float-edge` in
`tools/semantic_divergence_probe.py`) are that gate in draft; they are what
found all three.

One more thing they have in common, and it is the practical argument for doing
this as one piece of work: **006 measured out as free** -- NumPy's pairwise sum
is 4.4x faster *and* 108x more accurate than the current serial one. The
assumption that correctness here costs speed is what made all three easy to
defer, and it is not true for at least one of them.

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
- Entangled with KI-OPS-006: both entries want a NaN-aware max, and the same
  table row feeds the reduction, whose parallel passes match the literal
  `std::max(T(a),T(b))` / `::max(...)` spelling to route to atomics. A fix has
  to satisfy the elementwise case and the reduction together; the measurement in
  KI-OPS-006 (elementwise 1.2-1.6x, reduction 7.1-7.5x) says they cannot be
  treated as one change.
- Workaround: test for NaN explicitly before a max/min on CUDA where its
  presence matters.
- Review/expiry condition: CPU and CUDA agree with NumPy on NaN and on the sign
  of zero for `maximum` and `minimum`, and a device-parity case feeds NaN so the
  disagreement cannot return unnoticed.

## KI-BACKEND-005: CPU kernels are built with `-Ofast`, so infinities compute wrong

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: compiler and CPU backend maintainers
- Evidence: CPU, float32, vectors of length >= 4 (the vectorised path):

  | expression | Jittor CPU | IEEE / NumPy / Jittor CUDA |
  | --- | --- | --- |
  | `inf - inf` | `0.0` | `nan` |
  | `1 / 0` | `nan` | `inf` |
  | `-inf / 0` | `nan` | `-inf` |
  | `inf * 1` | `inf` | `inf` |

  A single element computes correctly; the wrong answers begin at length 4,
  which is where the kernel vectorises. CUDA is correct for all of them.
- Cause: `python/jittor/build/compiler.py:834` appends `-Ofast` to
  `kernel_opt_flags` unconditionally. `-Ofast` implies `-ffast-math`, which
  implies `-ffinite-math-only` -- a promise to the compiler that no operand is
  ever infinite or NaN. It then optimises on that promise, and operands that
  *are* infinite take whatever path the transformed code happens to produce.
- The project already knows: `compiler.py:89` strips `--use_fast_math` and
  `-Ofast` and substitutes `-O2` for one file, `nan_checker`. The workaround was
  applied where it was noticed rather than where it applies.
- Symptom: silently wrong arithmetic, and the shapes it takes are plausible
  rather than obviously broken. `inf - inf` returning `0.0` is the dangerous
  one: a fully masked attention row subtracts its own `-inf` maximum, and a `0`
  there produces a well-formed but wrong softmax instead of an obvious `nan`.
  KI-OPS-008 reaches the same input from the other side.
- Related: KI-OPS-006 measured that `-Ofast` also folds comparison-based NaN
  tests to false in the shipping build, which is the same flag defeating a
  different piece of correctness.
- Fix direction: `-O3` rather than `-Ofast`, or `-Ofast -fno-finite-math-only`.
  Both cost throughput and the amount is unmeasured -- vectorisation of
  reductions is the exposed part -- so this needs the same measure-then-decide
  the KI-OPS-006 entry records, not a straight substitution.
- Workaround: none within a kernel. Values that may be infinite have to be
  masked before they reach a CPU kernel.
- Review/expiry condition: the four expressions above agree with NumPy on CPU
  at every length, a probe case covers infinities on both devices, and the
  throughput change from the flag is measured and recorded.

## KI-BACKEND-006: CPU float32 sum/mean accumulate serially, so error grows with size

- Severity: Critical
- Status: Reproduced, unfixed
- Owner: CPU backend and reduction maintainers
- Evidence: summing `n` copies of `0.1` as float32, relative error against the
  exact value:

  | n | Jittor CPU | Jittor CUDA | NumPy |
  | --- | --- | --- | --- |
  | 65,536 | 6.2e-4 | 1.6e-7 | 1.6e-7 |
  | 1,048,576 | 9.9e-3 | 1.6e-7 | 9.8e-7 |
  | 16,777,216 | **1.5e-1** | 4.3e-7 | 1.6e-5 |

  `mean` inherits it: 1M copies of `0.1` average to `0.09975` on CPU
  (2.5e-3 relative) against `0.10000002` on CUDA.
- Symptom: at sixteen million elements the CPU sum is **15% wrong**. The growth
  is linear in `n`, the signature of a single serial accumulator; CUDA's tree
  reduction and NumPy's pairwise summation both stay near machine epsilon and
  do not grow. Nothing warns, and the result is a plausible number rather than
  an obviously broken one.
- Blast radius: any large reduction on CPU -- a loss averaged over a big batch,
  BatchNorm statistics, a norm, an accumulated metric. It also silently widens
  every CPU-versus-CUDA comparison, so a real backend divergence investigated
  at that size would be measured against a CPU reference that is itself wrong.
- Why the parity suite misses it: `tests/backends/parity` compares CPU against
  the accelerator, which is exactly the comparison that would show this, but its
  cases are small enough that the serial error is still near epsilon.
- Fix direction, and it is not a trade-off. Measured on 16.7M random float32
  (67 MB), same machine, same run:

  | | throughput | relative error |
  | --- | --- | --- |
  | Jittor CPU `sum` | 4.9 GB/s | 9.2e-5 |
  | NumPy `sum` (pairwise) | **21.3 GB/s** | **8.5e-7** |

  NumPy is **4.4x faster and 108x more accurate at the same time**. Blocked
  accumulation keeps several partial sums, which is what makes it accurate and
  also what lets the loop use more than one execution port -- the accuracy is a
  consequence of the faster shape, not a payment for it. The earlier reading of
  this entry said the fix would cost throughput and should be measured first;
  the measurement says the current reduction is leaving both on the table.
  (The 1.5e-1 in the table above is the worst case, all elements equal; random
  inputs cancel and land at 9.2e-5. Both are far above NumPy.)
- Workaround: reduce in float64 (`x.float64().sum()`), or reduce in chunks.
- Review/expiry condition: CPU relative error stays within an order of magnitude
  of NumPy's for n up to 16M in float32, a parity case covers a reduction large
  enough to have failed, and the throughput change is measured and recorded.

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
