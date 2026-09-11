# Active Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-09-11 -- a documentation pass over id collisions and
  statements the `-Ofast` removal made stale, not a re-verification of every
  entry
- Baseline: `2d716db31`
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

## KI-OPS-004: fixed -- reducing a rank-0 tensor returns its value

- Severity: was High
- Status: Fixed 2026-09-10, both devices
- Symptom it had: `sum`, `mean`, `max`, `min` and `prod` on a rank-0 tensor
  failed in one of two ways, depending on which pass reached it first -- a
  compiler diagnostic (`expected initializer before '-' token`) or
  `expr.cc:304 Check failed: nodes.size() == 1  Something wrong... Could you
  please report this issue?`. Neither named the shape or the operator. Generic
  code that reduces without checking rank -- `loss.sum()` where the loss is
  already scalar -- hit it on both devices.
- Cause: the reduce kernel opens with `index_t ystride@{DIM-1} = 1;`, and `DIM`
  is zero for a rank-0 input, so the generated source read
  `index_t ystride-1 = 1;`. A second line, `(void)yshape0, (void)ystride0;`,
  referred to names that likewise do not exist at rank 0.
- Fix: both guarded with `@if(DIM>0, ...)`. That is correct rather than merely
  compilable: with the guard every `@for` in the kernel expands to an empty
  nest, the body runs once with `yid == xid == 0`, and the result is the single
  input element -- which is what the reduction of one element is, and what
  NumPy and PyTorch return.
- Regression: `tests/ops/test_rank0_reduction.py`, both devices. It asserts the
  **shape** alongside the value: returning `3.5` with shape `(1,)` satisfies a
  value-only check and still breaks every caller that feeds the result
  somewhere expecting a scalar, which is the code this exists for. It also pins
  the two neighbours -- a one-element rank-1 tensor keeps its own shape, and
  rank-3 whole-tensor and per-axis reductions are compared elementwise against
  NumPy, because a `@if(DIM>0)` guard is exactly the kind of edit that can drop
  a stride declaration for every rank while a whole-tensor sum still looks
  right. Removing the guard turns it red with the `expr.cc:304` text above.
- `compat/tests/torch/test_division_remainder_family.py::
  test_reducing_a_scalar_tensor` was converted from a strict expected failure
  to an ordinary assertion. **That file could not be run on the machine where
  this was fixed**: the deployed Torch shim in site-packages is from an older
  release, so `import torch` raises `TorchActivationError` and the module does
  not collect. That failure predates and is unrelated to this change.

## KI-OPS-006: the NaN-correct CPU max/min reduction runs at half the speed

- Severity: Medium (throughput; the answers are correct)
- Status: Reproduced and measured on CPU, unfixed; CUDA is unaffected
- Owner: reduction operator and CPU codegen maintainers
- What this entry used to be: `maximum`/`minimum` and the `max()`/`min()`
  reductions dropped NaN. That is fixed -- see the KI-BACKEND-004 record below
  and [`test_minmax_nan_propagation.py`](../../tests/ops/test_minmax_nan_propagation.py),
  whose four strict expected failures are now ordinary passing cases. What is
  left is the second of the three costs that entry listed, and it is the only
  one that survived measurement.
- Evidence: `benchmarks/reductions.py` driven directly, jittor backend, three
  interleaved before/after repetitions on the same machine, float32:

  | | before | after | |
  | --- | --- | --- | --- |
  | `max` 1M | 14.29 GB/s | 7.24 GB/s | **1.97x slower** |
  | `max` 16M | 13.99 GB/s | 7.25 GB/s | **1.93x slower** |
  | `min` 1M | 14.30 GB/s | 7.26 GB/s | **1.97x slower** |
  | `min` 16M | 13.89 GB/s | 7.23 GB/s | **1.92x slower** |
  | `sum` 1M (control) | 24.53 GB/s | 24.21 GB/s | 1.01x |
  | `sum` 16M (control) | 18.67 GB/s | 18.41 GB/s | 1.01x |

  Elementwise `maximum`/`minimum` are unaffected (float32 and int32 at 16M,
  1056-1585 GB/s across the six runs, with before and after interleaved
  through that whole range -- the run-to-run spread covers the difference
  several times over). So is the closest real case: `softmax` over
  (16, 128, 1024), whose last-dim `max` is exactly this reduction, moves 1.175
  -> 1.207 ms while the `layernorm` control moves 1.690 -> 1.757 ms in the same
  run. The control moved more, so there is no measurable cost there.
- Cause, measured rather than assumed. `std::max(a, b)` is one `maxss`;
  NumPy's `maximum` is a compare, an or, and a select, and g++ does not
  recognise the result as a reduction. Timed in isolation on 16.7M float32 at
  the kernel's own flags (`-O3 -march=native`, one thread):

  | form | unit stride | runtime stride |
  | --- | --- | --- |
  | `std::max(a,b)` | 13.9 GB/s | 13.7 GB/s |
  | `((a>b) \| (a!=a)) ? a : b` -- shipped | 9.8 GB/s | 4.9 GB/s |
  | `(a>b \|\| a!=a) ? a : b` | 6.9 GB/s | -- |
  | `if (a!=a) ...; if (b!=b) ...;` | 2.4 GB/s | -- |
  | eight partials of the shipped form | **13.7 GB/s** | 4.3 GB/s |

  Two things follow. The spelling is worth 4x on its own -- `|` instead of
  `||` removes a branch the vectoriser will not cross, and the three-way `if`
  chain is the worst of the three -- which is why the shipped form is the one
  in the table. And the remaining 2x is **not** the comparison: eight
  independent partials of the same expression are level with `std::max`. They
  only are at unit stride.
- Which is the actual blocker. The reduce kernel indexes with
  `op0_xid = id0 * op0_xstride0` where `op0_xstride0` is `storage_stride(0)`, a
  runtime value. The measured fact is the right-hand column above: an opaque
  stride costs `std::max` nothing (13.9 -> 13.7) and costs the NaN-correct form
  half (9.8 -> 4.9). The mechanism is presumably loop versioning on the stride,
  which g++ does for a reduction it recognises and not for one it does not, but
  the column is the evidence and the mechanism is the reading of it. **Measured, not predicted:** extending
  `BlockedReductionPass` to `maximum`/`minimum` was implemented and it made
  things *worse*, 7.24 -> 3.58 GB/s, because there is no vectorisation for the
  partials to unlock and the block bookkeeping is pure cost. That change is not
  in the tree.
- Not a CUDA problem: the CUDA reduction folds in registers and through
  `cuda_atomic_max`, neither of which depends on this.
- Why the numbers here are not the 7.1-7.5x the old entry recorded, which was
  measured on `1e25ff68a`. Both ends moved. The baseline was `-Ofast` then, so
  `std::max` was reassociated and vectorised at 28-31 GB/s; KI-BACKEND-005
  removed that flag, and the same expression at `-O3` is 14 GB/s. And the old
  implementation was a bit test on the exponent inside an `if` chain, timed at
  3.9-4.9 GB/s; the shipped comparison is 7.2. A smaller numerator over a
  smaller denominator: 2.0x, not 7.4x. Neither figure was wrong for the tree it
  was taken on.
- Workaround: none needed for correctness. Where the throughput matters and the
  input cannot contain NaN, reduce on CUDA, or reduce in a dtype whose max is
  already exact (integer max/min never lost their vectorised form -- `a != a` is
  constant-false there and the compiler deletes it).
- Review/expiry condition: the reduce kernel's innermost stride is a
  compile-time constant when it is one -- the loop versioned, or `@if` on a
  unit-stride specialisation -- and the float32 `max`/`min` reduction is within
  10% of the `std::max` figures above on the same benchmark. Then this entry
  goes.

## KI-OPS-007: fixed -- the CUDA unary math table dispatches on dtype

- Severity: was High (silent precision loss on CUDA for every float64 transcendental)
- Status: Fixed 2026-09-11
- Symptom it had: nineteen entries in the CUDA expression table spelled their
  function with the float-only C variant -- `::logf`, `::expf`, `::sinf`,
  `::tanhf`, `::erff` and so on -- whatever the operand's dtype. A float64
  operand was narrowed to float32, evaluated at single precision and widened
  back. Most inputs hide it; it shows where the answer lives below float32's
  resolution:

  ```
  log(1 + 2**-51)   CPU 4.4408920985006252e-16   CUDA 0.0   NumPy 4.4408920985006252e-16
  ```

  `1 + 2**-51` is exactly `1.0` in float32, and `log(1.0)` is zero. Not
  slightly off -- gone.
- Cause: `round` had been given a dtype dispatch
  (`@if(@strcmp($1,float32)==0, ::rintf(...), ::rint(...))`) at some point;
  the other nineteen had not. `mod` had one too. The idiom existed in the file
  and was applied to one row.
- Fix: the same dispatch on the nineteen. For a float32 operand the template
  emits exactly the `::xxxf` text it emitted before, so the single-precision
  path is unchanged by construction; only float64 operands now reach the
  double-precision function. `src/type/common_op_type.cc`.
- Verified: `log`, `sqrt`, `sin`, `tanh`, `erf`, `exp` in float64 agree
  **bit for bit** across CPU, CUDA and NumPy at inputs chosen to collapse in
  float32. float32 `log`/`exp`/`sqrt`/`sin`/`tanh` over 1M elements unchanged
  against NumPy (the ~1e-7 spread is `--use_fast_math`, present before and
  after) and 23-25 us per call before and after.
- Regression: `tests/ops/test_float64_unary_math.py`, both devices. Inputs
  are chosen so the float32 answer is *qualitatively* wrong (zero, or equal to
  the input) rather than merely less precise -- a tolerance test passes on the
  old build for several of them. It also pins that the inputs really do
  collapse in float32, and that float32 is unchanged. Reverting the table turns
  it red: `log(1.0000000000000004) in float64 gave 0.0, NumPy gives
  4.440892098500625e-16`.

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
  was given. It is the `init_maximum`/`init_minimum` rows that are wrong, not
  the `maximum`/`minimum` rows beside them, which is why KI-BACKEND-004
  changing the latter did nothing here. Integers are unaffected: `lowest()`
  *is* their identity and they have no infinity to lose, so the fix has to
  dispatch on the dtype rather than replace the row.
- Distinct from KI-BACKEND-004, and untouched by it. That was the NaN
  behaviour of the `maximum`/`minimum` *operators*, now fixed; this is the
  identity the reduction folds *from* -- a per-output-element constant, with no
  effect on the inner loop. Re-measured after that fix, 2026-09-10: CPU
  `jt.max` of an all `-inf` float32 tensor still returns `-3.4028235e38` and
  CUDA still returns `-inf`, and the strict expected failure above still
  fails. `jittor::_max(lowest(), -inf)` is `lowest()`, exactly as
  `std::max(lowest(), -inf)` was, so nothing about this entry moved.
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

## KI-MEM-003: fixed -- a numpy_code backward read a pointer the op did not own

- Severity: was Critical (silent wrong gradient on CPU; illegal device access
  and a dead CUDA context on GPU)
- Status: Fixed 2026-09-10
- Symptom it had: every `jt.numpy_code` backward whose incoming gradient was
  non-contiguous got a `data["dout"]` pointing at host memory nobody had
  written. That is the *ordinary* case, not an exotic one: `jt.grad(loss, x)`
  seeds the backward with a stride-0 broadcast of `1`.

  On CPU it did not raise. The op's own docstring example,
  `np.copyto(out, dout*2.0)` over a length-5 float32:

  | | value |
  | --- | --- |
  | `dout` handed to the callback | `[1.5324866e+07, 3.3814733e-41, 0.0, 0.0, 4.4841551e-44]` |
  | gradient produced | `[3.0649732e+07, 6.7629466e-41, 0.0, 0.0, 8.9683102e-44]` |
  | correct gradient | `[2.0, 2.0, 2.0, 2.0, 2.0]` |

  On CUDA the same host address was wrapped as device memory by
  `jittor.numpy2cupy` and handed to a CuPy kernel. `compute-sanitizer`:

      Invalid __global__ read of size 4 bytes
        at 0x160 in cupy_multiply__float32_float_float32
        Address 0x60cd138fa1e0 is out of bounds
        Host Frame: ... PyNumber_Multiply          <- the `dout * 2.0`

  `cudaErrorIllegalAddress` is sticky, so it took the context with it and every
  later test in the process failed on a poisoned context -- CPU-only ones
  included -- until the process died in jittor's fatal-log `_exit(1)`.
- Cause: `NumpyCodeOp::grad` recorded the gradient operands as raw `Var*` in
  `NumpyResult` and `run()` dereferenced them later. Between the two,
  `make_numpy_code` (generated by `python/jittor/build/codegen.py`) runs
  `adapt_storage_input<NumpyCodeOp>` over the constructor's input vector:
  `accepts_storage_strides` is false for this op, so a non-contiguous operand
  is replaced in its slot by `contiguous_storage(value)` *before* the op is
  built. The recorded Var was therefore never an input of the op that ran, was
  not kept alive by it, and had `mem_ptr == 0` by run time -- measured, with the
  op's own inputs beside it:

      NPDBG doutVar= 0x...113f90 mem_ptr= 0        alloc_is_cuda= -1
      NPDBG   input 1 0x...cfee10 mem_ptr= 0x7c0888e46c00 alloc_is_cuda= 1

  The null pointer was then silently upgraded into a real buffer:
  `to_py_object(DataView)` passes it to `PyArray_New`, which allocates its own
  **host** buffer when `data` is NULL. The callback received a well-formed
  array of the right shape and dtype over uninitialized host memory, and
  `numpy2cupy`'s `cp.cuda.UnownedMemory` wrapped that host address without
  complaint.
- Not the cause: the unused `get_jittor_cuda_malloc`/`get_jittor_cuda_free`
  handles in `python/jittor/build/init_cupy.py`, the absent
  `cp.cuda.set_allocator`, the hard-coded `device_num = 0`, or multiple GPUs.
  The defect reproduces with `CUDA_VISIBLE_DEVICES=0`, and `cvt()` computes the
  right byte count for every key -- the address it was given was already wrong.
  Reading `data["inputs"][1]` instead of `data["dout"]` was always correct, on
  both devices, which is what pinned the cause.
- Fix: `src/ops/composite/numpy_code_op.cc`. `grad()` records *where* it put
  dout and the forward outputs in the backward op's input list -- two private
  `NumpyResult::ints` keys, stripped in `run()` before the callback sees the
  dict -- instead of recording the Vars, and `run()` resolves them from
  `_inputs`, which is post-replacement. Slot positions survive
  `adapt_storage_input` because it substitutes in place. Every `DataView` handed
  to the callback now also asserts a non-null buffer for a non-empty var, so the
  `PyArray_New` host-allocation fallback can no longer stand in for a missing
  pointer.
- Blast radius it had: `data["dout"]` and `data["f_outputs"]` are the entire
  gradient interface of `jt.numpy_code`, so this covered `jt.linalg`
  (`solve`, `svd`, `cholesky`, `inv`, `det`, `eigh`, ...) through
  `python/jittor/linalg/*.py` and `_standard_gamma_grad` through
  `python/jittor/contrib/math_util/gamma.py` -- on both devices.
- Measured, `tests/ops` + `tests/opinfo` on CUDA, 690 tests. One process,
  before: the session **died** at test 389 of 690 (`test_matmul_dispatch.py::
  TestDispatchCUDA::test_every_relay_is_optional`, reproduced 2/2), no pytest
  summary and no junit XML, having recorded 308 passed / 30 failed / 50 skipped
  and leaving **301 tests unrun**. After: the session completes --
  `49 failed, 554 passed, 81 skipped, 6 xfailed, 2 errors in 271.41s`. Over the
  389 nodeids both runs reached: 12 newly passing, **0 newly failing**. Per-file
  isolation (each file its own process, so no cross-file cascade) puts the
  directly-attributable count at 51 -> 44 failures, 6 newly passing, 0 newly
  failing; 16 distinct nodeids turn green across the two views. One further
  per-file flip, `test_reindex_op.py::TestReindexOpCuda::test_conv_transpose_group`,
  is *not* claimed: it passes 3/3 on the unfixed source in isolation and was
  baseline noise.
- Regression: `tests/ops/test_numpy_code_op.py::TestCodeOp::test_backward_dout_is_device_resident`
  asserts the gradient value and, on CUDA, that
  `cudaPointerGetAttributes(dout).type != cudaMemoryTypeUnregistered`. The value
  assertion alone is not sufficient: `cupy.copyto` off a host pointer returns a
  wrong gradient without raising and without a sanitizer error, and only an
  arithmetic kernel trips the illegal access.

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

## KI-EXEC-003: cuDNN autotuning is not isolated from execution scheduling

- Severity: High (silent, deterministic change to training numerics)
- Status: Cause identified 2026-09-10; a small residue unexplained. Unfixed.
- Owner: CUDA backend maintainers
- Evidence: the same model, the same input, the same build -- only
  `auto_flush_ops` moves. Forward loss **bit-identical** at every setting; the
  gradients are not. Four blocks, a size that never crashes, so this is not the
  segfault of [KI-EXEC-001] wearing another hat:

  | `auto_flush_ops` | 0 | 32 | 64 | 128 | 256 |
  | --- | --- | --- | --- | --- | --- |
  | loss | 20322.414062 | (same) | (same) | (same) | (same) |
  | gradient norm | 33076.398 | **33063.719** | 33076.398 | 33076.398 | 33076.398 |

  Five blocks, `auto_flush_ops=64` is the odd one: 45822.207 against 45839.379
  everywhere else. Both outliers are about `3.8e-4` relative. **Deterministic**
  -- the same setting gives the same number three runs out of three.
- **Cause, measured.** `cudnn_conv_op.cc` chooses its convolution algorithm by
  *measuring* the candidates rather than asking cuDNN's heuristic, and caches
  the winner per shape; `max_workspace_ratio` is part of the cache key. What is
  resident when that measurement runs decides which algorithm wins, and
  `auto_flush_ops` changes what is resident. Turning the measurement off
  (`set_benchmark(0)`, which forces the heuristic) collapses the spread:

  | | flush 0 | 64 | 256 |
  | --- | --- | --- | --- |
  | benchmark on (default) | 45839.379 | **45822.207** | 45839.379 |
  | benchmark off | 45822.207 | 45822.207 | 45822.207 |

  The five-block divergence disappears entirely.
- **Residue, unexplained.** With the benchmark off, four blocks at
  `auto_flush_ops=32` still gives 33063.723 against 33063.688 elsewhere --
  `1.1e-6` relative, three hundred times smaller than the algorithm effect and
  in the range a changed accumulation order would produce. Not chased further;
  recorded so it is not mistaken for zero.
- Not the reduction order alone: a tape-free elementwise model
  (`for _ in range(300): y = y * w + 0.001`) has **identical** gradients at
  every setting, and only its `sum()` loss moves between two values.
- Why it matters: `3.8e-4` is far above float32 rounding for this quantity, it
  is deterministic rather than jittery, and it is invisible -- the loss agrees
  to the last bit, so any check watching the loss reports nothing. Two runs of
  the same script with different `auto_flush_ops` train to different weights.
  The comparison to draw is PyTorch's `torch.backends.cudnn.benchmark`, which
  has the same property and **says so in its documentation**; here the coupling
  is undocumented and reached through a flag that reads as a scheduling knob.
- Related, same flag: [KI-EXEC-001] and [KI-EXEC-002].
- The "state it where users will read it" half of the exit condition is done
  as of 2026-09-11: `docs/notes/numerics-contract.md` carries the measured
  table, the `set_benchmark(0)` workaround and the discipline that a numeric
  comparison across a residency-changing flag must hold the autotuner still.
  What keeps this entry open is the coupling itself and the unexplained
  residue above.
- Review/expiry condition: the algorithm chosen for a given shape does not
  depend on what else is resident -- measure into a scratch buffer of a fixed
  size, or key the cache on something stable. A regression sweeps
  `auto_flush_ops` and compares **gradients**, not the loss.

## KI-EXEC-002: fixed -- the profiler no longer loses flushed work, and says when it measured nothing

- Severity: was High (measurements silently partial; two gates permanently red)
- Status: Fixed 2026-09-10 for the half that is fixable; the other half is now
  stated rather than silent.
- Symptom it had: CUDA, a 64x64x64x64 tensor sliced and concatenated, then
  differentiated. Same expression at each row; only the number of slices moves.

  | slices | rows `jt.profile_scope` reported |
  | --- | --- |
  | 1, 2, 8 | 6-7 |
  | **16, 32, 64** | **0** |

  The result was correct at every row, so the work happened -- outside the
  scope. `auto_flush_ops` (`src/core/executor.cc`, default 128, CUDA only)
  launches everything pending once that many operators have been built, so a
  graph constructed before `with jt.profile_scope()` may already have run by
  the time the scope opens.
- Fix, two parts, because the problem has two:
  * **Work built inside the scope is no longer launched behind the profiler's
    back.** `profile_scope` now sets `auto_flush_ops=0` for its duration unless
    the caller overrides it. Profiling is a measurement; the pipelining it
    would otherwise measure is not what is being asked about. With the graph
    built inside the scope, 64 and 32 slices went from **0 rows to 9**.
  * **A report with no operators is no longer silent.** Nothing can recover
    work that ran before the scope opened, but the scope can say so: it now
    raises a `RuntimeWarning` naming the cause and what to do about it. "It ran
    fast" and "nothing was measured" used to be the same output.
- The two gates it kept red: `tests/ops/test_concat_op.py::test_concat_perf`
  and `::test_concat2_perf` divided the transferred bytes by the profiler's
  total and failed with `ZeroDivisionError` on every run -- a message naming
  neither the profiler nor the cause. Both now build their graph **inside** the
  scope, which is the right window to measure anyway, and the file passes
  3/3.
- Residual, stated: a graph built before the scope still cannot be measured.
  That is inherent -- the work is gone -- and the warning is the honest
  answer rather than a fix.
- Same flag, still open: [KI-EXEC-003] (cuDNN autotuning is not isolated from
  scheduling). [KI-EXEC-001] is fixed.
- Review/expiry condition: met -- a profile over a graph built inside the scope
  accounts for its operators at any size, and one that measures nothing says so.

## KI-EXEC-001: fixed -- a control-only op is no longer held to a compute op's rules

- Severity: was Critical (segfault; ResNet50-class backbones did not run on CUDA)
- Status: Fixed 2026-09-10
- Symptom it had: five bottleneck blocks segfaulted at `auto_flush_ops` 1, 16,
  32 and **128, the shipping default**, while 64 and 256 happened not to.
  Deterministic per setting, five runs each. JSeg and JDet both crashed with a
  ResNet50 backbone while JSeg's ResNet18 passed. That is what made it read as
  "past a graph-size threshold": the threshold was not a size, it was the first
  place the cut landed on a tape.

  ```
  Allocator::is_cuda()          src/mem/allocator.h:35     <- segfault
  run_exec_plan                 src/core/exec_runner.cc:331
  Executor::run_sync            src/core/executor.cc
  Executor::submit_pending      src/core/executor.cc
  schedule_pending_from_python  src/core/var_holder.cc:62
  to_py_object<VarHolder*>      src/bindings/pyjt/py_converter.h:626
  ```

  Not a GPU fault: `compute-sanitizer --tool memcheck` reported
  `ERROR SUMMARY: 0 errors` on a run that segfaulted.
- Cause: `Tapes` is a **control-only op**. It has no `run` and no `jit_run`; its
  single output is a zero-sized Var wired as an edge into the producer of each
  taped output, and it names the pre-tape Vars so the backward can reach them.
  It reads none of their bytes, sets `_manual_set_vnbb`, and marks none of them
  needed -- so the executor frees them once their real consumers finish, which
  is correct. `run_exec_plan` then applied the rule for a *compute* op to it:
  migrate every input to the device, and assert every input is backed. On a
  freed Var the first of those is a null dereference. With the whole graph in
  one batch the two never met; `auto_flush_ops` puts a `Tapes` in a batch whose
  producers have already finished.
- Fix: `OpFlags::_no_input_storage`, set by `Tapes` alone, and honoured by the
  two places in `run_exec_plan` that walk `op->inputs()`. Three files, two
  lines of behaviour.
- Verified: no crash at `auto_flush_ops` 0, 1, 16, 32, 64, 128, 256 or 512 at
  five blocks, nor at six, seven and eight blocks on the default. The loss is
  **bit-identical** at every setting. With cuDNN autotuning disabled -- so the
  comparison isolates this defect from [KI-EXEC-003] -- the gradient residue
  across settings is at most `2e-6`, which is reassociation from different
  batch boundaries.

### The first rejection of this fix was wrong, and how

This exact fix was tried earlier the same day and rejected on the grounds that
it "turns the crash into gradients wrong by 40-60% on individual elements".
Both halves of that were the reviewer's error:

* the comparison ran with cuDNN autotuning **on**, and autotuning depends on
  what is resident, which is what the flag under test changes -- worth `3.8e-4`
  on the gradient norm all by itself ([KI-EXEC-003]);
* "40-60%" came from dividing a maximum absolute difference by the gradient's
  **RMS** rather than by the magnitude of the element it belonged to.

Repeated with autotuning off, the residue is `2e-6`. The lesson is not
"measure more" -- it is that a comparison across a flag that changes memory
residency must first hold the autotuner still, and that a relative error needs
the element it is relative to.

- Regression: `tests/backends/cuda/test_auto_flush_graph_split.py`. Each
  setting runs in **its own process**, because the failure is a segfault: in
  process it ends the suite rather than failing a case, and every later test
  silently does not run. It asserts the loss is bit-identical **and** compares
  gradients -- a fix that stops the crash while leaving the backward reading
  the wrong bytes passes a "did it run" check and fails here, which is the
  point. It disables cuDNN autotuning so a `3.8e-4` band does not hide
  anything smaller. Reverting the fix turns it red with
  `auto_flush_ops=1 produced no result`.
- Still open on the same flag: [KI-EXEC-002] (the profiler cannot see flushed
  work) and [KI-EXEC-003] (cuDNN autotuning is not isolated from scheduling).

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

## KI-BACKEND-007: fixed -- CUDA `std`/`norm` propagate NaN

- Severity: was Critical
- Status: Fixed 2026-09-10, as a consequence of [KI-BACKEND-004]
- Symptom it had: `jt.std([nan, 1.0, 2.0])` gave `nan` on CPU and
  `0.0009999999310821295` on CUDA; `norm` gave `nan` against `1e-15`. NumPy
  agrees with CPU.
- Cause: both are composed reductions that pass through the `maximum`/
  `minimum` rows of the expression table, and CUDA's `::max` resolves to
  `fmaxf`, whose IEEE `maxNum` semantics deliberately return the non-NaN
  operand. The NaN was dropped inside the composition and the small finite
  number is the epsilon the composition adds.
- Fix: none of its own. With `maximum`/`minimum` propagating NaN the way NumPy
  does, re-measured 2026-09-11: `std` and `norm` of `[nan, 1.0, 2.0]` are `nan`
  on both devices.
- Regression: covered by `tests/ops/test_minmax_nan_propagation.py` (the row
  it passes through) and the semantic divergence probe's `std`/`norm` cases.

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
  `docs/notes/numerics-contract.md` (it was written into
  `docs/notes/float32-precision-policy.md` first, and moved 2026-09-11 when the
  numeric contracts were collected onto one page) and asserted by
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
- 004 was the expression table. `std::max` and `::max` were each chosen for
  being the obvious spelling, and their NaN behaviour -- accidental on CPU,
  deliberate IEEE `maxNum` on CUDA -- was never part of the choice. It is fixed;
  the record is below, and the numbers are in
  [the result report](../../refactor-wip/results/2026-09-10-minmax-nan-numpy-parity.md).
- 006 was the reduction shape. A single serial accumulator is what you write
  when accuracy at scale is not a stated requirement. It is fixed and its entry
  is gone; see
  [the result report](../../refactor-wip/results/2026-09-10-cpu-reduction-blocked-pairwise.md).
  (`KI-OPS-006`, a different number, is what is left of 004's throughput cost.)

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

One more thing they have in common, and on the two that have been done it is
now measured rather than predicted. **006 was free**: blocked accumulation with
a pairwise fold is **3.7x-4.9x faster** than the serial loop it replaced (4.9 ->
18.4 GB/s at 64M float32 on the reduction benchmark) *and* leaves the
worst-case relative error at 16.7M elements at 6.0e-7 instead of 1.5e-1 --
below NumPy's own 1.6e-5. The one cost found was JIT compile time on large
fused reduction kernels, +18% after the emitted code was scaled to the body.

**004 was not free, and it was not the 7.1-7.5x that had been quoted for it
either.** Elementwise `maximum`/`minimum` and the softmax case cost nothing
measurable; the CPU `max`/`min` *reduction* costs 1.9-2.0x, and the remaining
factor is a runtime stride that stops the loop vectorising rather than the
comparison itself -- eight partials of the shipped expression are level with
`std::max` at unit stride. The details are in KI-OPS-006. So the shared lesson
is not "correctness is free" -- it is that the estimates written down to justify
deferring were wrong in the same direction both times, and by a lot: 006 was
predicted to cost speed and gained 3.7-4.9x, 004 was quoted at 7.1-7.5x and
costs 2.0x. In each case taking the measurement was less work than the argument
about whether to take it.

## KI-BACKEND-004: fixed -- `maximum`/`minimum` and `max`/`min` now answer NaN the way NumPy does

- Severity: was Critical
- Status: Fixed 2026-09-10, both devices
- Symptom it had: `f = [nan, -inf, -0.0, 0.0, inf]` against zeros, float32.
  CPU gave `[nan, 0.0, -0.0, 0.0, inf]`, CUDA gave `[0.0, 0.0, 0.0, 0.0, inf]`,
  NumPy gives `[nan, 0.0, 0.0, 0.0, inf]` -- the same expression on the same
  input disagreeing between the two devices, and neither agreeing with NumPy.
  The reduction was worse and agreed across devices only by being uniformly
  wrong: `jt.max`/`jt.min` over an array holding one NaN returned `1.0` on both
  devices at `n` = 5, 4096 and 1,048,576, where NumPy returns `nan`. `x.max()`
  is a common way to ask whether a tensor has gone bad and it could not see a
  NaN at all.
- Cause: two spellings, each obvious, neither chosen for its NaN behaviour.
  `std::max(a, b)` is `a < b ? b : a`; every comparison against NaN is false,
  so it returns whichever operand was written *first*. That looked like
  propagation in the elementwise evidence above only because the NaN happened
  to be written first; a reduction folds `acc = max(acc, x)`, where an arriving
  NaN is always *second*, so it was discarded. CUDA's `::max` lowers to
  `fmaxf` -- IEEE `maxNum`, which deliberately returns the operand that is not
  NaN -- so it discarded a NaN in either position.
- Fix: `src/type/minmax_compute.h` defines `jittor::_max` / `jittor::_min` as
  NumPy defines them, `((a > b) | (a != a)) ? a : b`, and both tables in
  [`common_op_type.cc`](../../src/type/common_op_type.cc) now emit those for
  every dtype. One template covers integers: `a != a` is constant-false there
  and the compiler deletes it, so the integer lowering is unchanged.
- The sign of zero came with it, and it is order dependent on purpose. NumPy
  decides `maximum` with `>` alone, so `maximum(-0.0, 0.0)` is `+0.0` and
  `maximum(0.0, -0.0)` is `-0.0`; `minimum` mirrors it. That was measured
  against NumPy rather than assumed, and `a > b` reproduces it exactly, so no
  signbit special case was needed. CPU used to return its *first* operand for
  both -- `-0.0` then `+0.0`, wrong in both directions -- and CUDA `+0.0` for
  both, right by accident in one of them.
- Two things the fix needed beyond the table, and the second was found by
  measuring rather than by reading:
  1. `parallel_pass.cc` and `atomic_tuner_pass.cc` match the *literal*
     `std::max(T(a),T(b))` / `::max(...)` to route a reduction through
     `cpu_atomic_max` / `cuda_atomic_max`, so changing the table without them
     loses the atomic path. `expr::match` does accept a qualified call --
     `jittor::_max(T(a),T(b))` matches, verified by the emitted kernel still
     containing `cuda_atomic_max` -- and a spelling that did *not* match would
     not degrade silently, it reaches a fatal `Expr not match`. The `std::max`
     and `::max` patterns are still there because the float16 table still emits
     them.
  2. With the kernel body correct, CUDA still returned `1.0` from `jt.min` over
     an array holding `+nan` and from `jt.max` over one holding `-nan`, at every
     size. `cuda_atomic_max/min(float*)` are an `atomicMax` over an ordered-int
     encoding in which a positive NaN sorts above `+inf` and a negative NaN
     below `-inf`, so each operation carried one sign of NaN and dropped the
     other. Any NaN is now encoded as the extreme key of its operation
     (`0x7FFFFFFF` for max, `0x80000000` for min), both of which decode back to
     a NaN, so it outranks every real number in either direction. `fix_float`
     and `float_atomic_fix_pass` are untouched. `shared_reduce_max/min` and the
     raw-IEEE `cuda_atomic_max_rmw/min_rmw` used by scatter got the same
     treatment -- the last one so that a CPU scatter-maximum, which lowers
     through the table, does not start disagreeing with its CUDA counterpart.
- Verified against NumPy on both devices: elementwise over every float class in
  both operand orders, float32 and float64; both signs of NaN through `max` and
  `min` at `n` = 5, 4096 and 1,048,576; a 64x4096 reduction along a dim, which
  is the parallel/atomic path rather than the scalar one; and int8/int16/int32/
  int64/uint8 asserted unchanged.
- Regression: [`test_minmax_nan_propagation.py`](../../tests/ops/test_minmax_nan_propagation.py),
  22 cases, including the device-parity class this entry asked for -- the same
  operands run on both devices and compared to each other as well as to NumPy,
  so a future divergence cannot pass unnoticed even if each device looks
  individually plausible. On the unfixed tree 16 of the 22 fail.
- Cost, measured: the CPU `max`/`min` *reduction* runs at half speed. See
  [KI-OPS-006], which is what remains of that entry. Elementwise and the
  softmax case are unaffected, and CUDA is unaffected.
- Removal condition: delete this record once the result report
  [2026-09-10-minmax-nan-numpy-parity.md](../../refactor-wip/results/2026-09-10-minmax-nan-numpy-parity.md)
  has been read by a maintainer.

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
- Related: the same flag also folds comparison-based NaN tests to false in the
  shipping build, which is one flag defeating two separate pieces of
  correctness. KI-BACKEND-004 could not be fixed until it was removed -- a
  NaN-propagating `max` is written `a != a`, which `-ffinite-math-only` deletes,
  so the fix would have read correct and compiled to the old behaviour.
- And a third consequence, which is the one that makes CPU results
  irreproducible rather than merely wrong: **whether an expression was fused
  changes its answer.** `(a + b) - a` with `a = -1e8`, `b = 2.0` in float32
  gives `0.0` unfused -- `2.0` is below the ULP of `1e8`, so the addition
  discards it, which is what the written expression says -- and `2.0` fused,
  because the larger expression handed to the compiler is reassociated to
  `b + (a - a)`. Measured with `tools/fusion_consistency_sweep.py`: 12 cases,
  CPU has one differing, CUDA has none.

  Fusion depends on what else is in the graph, so the same code gives different
  answers in different surroundings. This is what made an earlier probe check
  unstable -- it returned `1.0` inside the probe and `0.0` standalone and was
  withdrawn for having no stable expectation. Stated as "fused and unfused must
  agree" it needs no expectation at all, which is why that invariant is the one
  worth gating on.
- Fix applied 2026-09-10 (`1e50d76c5`): `python/jittor/build/compiler.py`
  appends `-O3` to `kernel_opt_flags`. The alternative considered was
  `-Ofast -fno-finite-math-only`; `-O3` was taken because the reassociation
  `-ffast-math` also grants was not being used -- g++ 12.3 does not vectorise
  the real reduction kernels, the runtime `storage_stride(0)` blocks it -- so
  the throughput it was thought to buy was not there to lose.
- Cost, measured, in two parts. On the shapes measured with the change, none:
  same machine, same warm cache, elementwise chain 4M 0.000320 -> 0.000324 s;
  `exp`/`sqrt` chain 4M 0.000747 -> 0.000679; `sum` 4M 0.000737 -> 0.000719;
  matmul 512 0.659172 -> 0.660539. Full `tests/ops` + `tests/opinfo` compared
  nodeid by nodeid: 260 failures at `-O3` against 261 at `-Ofast`, and **the
  set that fails only at `-O3` is empty**.

  One shape did pay, and KI-OPS-006 carries the number: the float32 `std::max`
  reduction ran at 28-31 GB/s under `-Ofast`, where the reassociation did
  vectorise it, and runs at 14 GB/s at `-O3`. That is a 2x on that one kernel,
  paid for IEEE arithmetic everywhere. It is recorded rather than netted out,
  and it is the reason KI-OPS-006's ratio changed without either measurement
  being wrong.
- The fused-versus-unfused divergence went with it:
  `tools/fusion_consistency_sweep.py` on CPU went from 1 differing case to
  12/12 identical.
- No workaround is needed any more. Before the fix there was none inside a
  kernel: values that might be infinite had to be masked before they reached
  one.
- Regression: `tests/ops/test_ieee_arithmetic.py` (ten IEEE-defined
  expressions, both devices, length 8 -- at length 1 they all pass even with
  the flag wrong) and `tests/structure/codegen/test_kernel_math_flags.py`,
  which names the flag so a reintroduction says what changed, and separately
  asserts an optimisation level is still requested so "delete the flag and put
  nothing back" cannot satisfy it.
- Review/expiry condition: met. Delete this record once a maintainer has read
  it.

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

## KI-OPS-009: fixed -- an index Var that is a strided view is now read through its strides

- Severity: was Critical (out-of-bounds read *and write*, silent wrong answers)
- Status: Fixed 2026-09-10
- Symptom it had: any indexing operation whose index Var came from a broadcast.
  Not `scatter_add`, where it was first seen, and not `setitem`: the read side
  (`gather`) failed identically. On a CPU-only build (`nvcc_path=""`), each row
  a fresh process:

  | index expression | result |
  | --- | --- |
  | `jt.zeros((4,5), 'int64')` | index `2697334449954054313`, out of bounds |
  | `jt.zeros((4,5), 'int32')` | index `-1887156860`, out of bounds |
  | `jt.zeros(...)` after `idx.sync()` | index `-356873746589917012`, out of bounds |
  | `jt.array(np.zeros((4,5), 'int64'))` | correct |
  | `jt.ones((4,5), 'int64') - 1` | correct |
  | `jt.zeros(...)` through plain `setitem` | out of bounds |
  | `jt.zeros(...)` through `gather` | out of bounds |

- Cause: **a broadcast is a storage descriptor, and the index kernels read
  index Vars as if they were dense.** `jt.zeros(shape, dtype)` is
  `unary(0, dtype).broadcast(shape)`, and `BroadcastToOp::infer_shape`
  ([`broadcast_to_op.cc`](../../src/ops/broadcast_to_op.cc)) gives its output
  zero strides and `share_with(x)` -- no elementwise kernel, no allocation of
  the logical footprint. So a `(4,5)` int64 index Var is backed by **eight
  bytes**. The `getitem`/`setitem` kernels compute the index Var's strides from
  the *output* shape --
  `vp@d[0 @for(j,0,VD, @if((VS@d>>j)&1, + i@{j+FOV} * vs@d@@s@j,))]`, with
  `vs@d@@s@j` folded out of `oshape@{j+FOV}`
  ([`getitem_op.cc`](../../src/ops/composite/getitem_op.cc),
  [`setitem_op.cc`](../../src/ops/composite/setitem_op.cc)) -- and so walked 20
  elements off the end of that one-element buffer and used what it found as an
  index.
- Proof that it is the neighbours in memory and not "unallocated pointer", with
  no heap forensics: broadcast a one-element *view* of a buffer whose next 19
  elements are a known pattern. `base = jt.array(np.arange(20) % 4)`,
  `idx = base[0:1].broadcast((4,5))` -- every logical element of `idx` is
  `base[0] == 0`. Gathering rows out of a source whose values name their row
  returned `[[0,1,2,3,0],[1,2,3,0,1],[2,3,0,1,2],[3,0,1,2,3]]`: exactly
  `base[0..19]`, read densely. It is `tests/ops/test_broadcast_index.py::
  test_a_broadcast_index_reads_its_own_element`, and it fails deterministically
  on both builds before the fix.
- **The build was never the discriminator, and the previous reading of this
  entry had that wrong.** The overshoot for a `(4,5)` int64 index is 152 bytes;
  whether those bytes are zeroes or garbage is a property of the heap, not of
  the code. A CUDA build fails too. With `jt.flags.use_cuda = 0` -- same
  kernel, same device as the build that was called broken -- a `(4,64)` index
  overshoots 2040 bytes and gives
  `index 1115160576 is out of bounds for dimension 0 with size 4`; with
  `use_cuda = 1` the device-side check prints
  `[jittor] index 4870502260641759232 is out of bounds for dimension 0 with
  size 4` and traps. That is also why the generated kernel was byte-identical
  between the two builds: there was nothing build-specific to see.
  `HAS_CUDA`/`HAS_ACCELERATOR` are not involved.
- Why the two hypotheses that were tried did nothing: the broadcast Var **is**
  allocated and its single element **is** correct, so `idx.sync()` cannot help
  and neither can `VarFlags::_stop_fuse` on the index Vars -- nothing was being
  fused away. The dtype is irrelevant for the same reason. What was wrong was
  the arithmetic the kernel used to reach the second element of a Var that has
  only one.
- The guard existed and had never once been emitted: `adapt_index_storage`
  ([`var_slices.h`](../../src/core/var_slices.h)) routes a non-contiguous index
  Var through `contiguous_storage` before the op is built, and
  [`codegen.py`](../../python/jittor/build/codegen.py) is supposed to insert a
  call to it into every generated `make_*` that takes a `VarSlices`. It never
  did. The argument list is split off the C++ declaration, so every argument
  after the first still carries the space that followed the comma, and
  `" VarSlices&& slices".startswith("VarSlices")` is false. `Var*` arguments are
  rebuilt from the parsed type and happen to arrive clean, which is why
  `adapt_storage_input` worked and its neighbour did not. `grep
  adapt_index_storage` over the generated sources returned nothing, for every op
  and every build.
- Fix: one `strip()` before the match. `make_getitem` and `make_setitem` now
  emit `adapt_index_storage(slices, _storage_owners);`, so a non-contiguous
  index Var is materialised into a dense one before either kernel sees it.
- Cost, CPU-only build, 8192x256 int64 index into an 8192x256 float32 table,
  minimum of 20:

  | | before | after |
  | --- | --- | --- |
  | dense index (the common case) | 0.000334s | 0.000324-0.000344s |
  | broadcast index | wrong answer | 0.000473s |

  The dense path is unchanged, and not only by measurement: `is_contiguous()` is
  true, `contiguous_storage` hands the same Var back, and no op is created. A
  broadcast index now pays one materialisation -- +42% on this shape, which is
  what a 16 MB copy costs next to this gather.
- The cheaper fix that was **not** taken, and why it is the follow-up rather
  than the fix: `reindex_op.cc` and `reindex_reduce_op.cc` already read their
  `extras` through `extras[@i]->storage_stride(@j)` and need no copy. The two
  index kernels could do the same and skip the materialisation entirely. That
  is one kernel template against every current and future `VarSlices` op, and it
  would leave `adapt_index_storage` dead -- which is the condition that produced
  this defect. Worth doing on top, with the copy kept as the fallback for any op
  that does not opt in.
- Regression: [`test_broadcast_index.py`](../../tests/ops/test_broadcast_index.py),
  12 cases over both devices -- `gather` at three index widths and two dtypes
  (the widths matter: a narrow index can overshoot into zeroed memory and look
  healthy, which is how this was first mis-recorded as build-specific), the
  heap-independent neighbour-read case above, `setitem`, `scatter_add`, a dense
  index that must not move, and the premise itself (`broadcast` still produces a
  strided view, so the file cannot quietly stop testing anything). Reverting the
  codegen change turns it red on both builds: **4 failed / 2 passed**
  on CPU-only, **7 failed / 5 passed** on CUDA, with
  `index 2796023709697 is out of bounds for dimension 0 with size 4` from
  `getitem_op.cc:461` and `index 1082130432 ...` from `setitem_op.cc:385`.
  [`test_index_storage_adaptation.py`](../../tests/structure/codegen/test_index_storage_adaptation.py)
  is the label: it reads the generated `jit_op_maker.h` and fails naming the
  maker that lost the call, which is the half a behavioural test cannot cover
  for an op that does not exist yet.
- Nothing else moved: `tests/structure` on a CPU-only build reports the same 69
  failures before and after, name for name (35 ACL, 15 cuDNN/cuSPARSE error
  boundaries, the rest refactor-era -- see KI-TEST-004), and
  `tests/ops/test_slice.py` + `tests/ops/test_reindex_op.py` on a CUDA build go
  from 9 pre-existing failures to 8. Both selections were re-run against the
  unpatched tree in a separate cache to get those baselines rather than assumed.
- Also cleared by this: `tests/ops/test_ops.py::TestCommonCPU::
  test_reference_scatter_add_float32`, which KI-TEST-002 names as one of the
  four cases that end a CPU-only session. It went from
  `index 7887331678563036767 is out of bounds for dimension 1 with size 4` to
  `1 passed`. The reporting hole KI-TEST-002 is about is unaffected.
- What KI-OPS-010 did and did not do: the index bounds check added the same day
  turned this from a segfault that took the interpreter down into a `UserError`
  naming the offending index, which is how the table above was obtained at all.
  It did not fix this defect, and a reading of it as "no longer crashes" would
  have been wrong -- the index was still whatever happened to be next to the
  broadcast's single element.

## KI-TEST-002: a dead session is indistinguishable from a short one

- Severity: High
- Status: Open
- Owner: test infrastructure maintainers
- Evidence: on a CPU-only build (`nvcc_path=""`, the configuration
  `tools/run_test_suite.py` sets), the maintained Torch selection printed
  progress to 48% and then ended with exit 1 -- no result line for the case that
  died, no traceback, no summary, and no mention of the ~2200 tests that never
  ran. Four cases did this on that build:
  `tests/ops/test_ops.py::TestCommonCPU::test_reference_scatter_add_float32`
  (KI-OPS-009, fixed 2026-09-10 -- that case now passes there, and the
  reporting hole this entry is about is untouched by it),
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

## KI-TEST-005: fixed -- device tests now restore `use_cuda` instead of zeroing it

- Severity: was Medium (test isolation)
- Status: Fixed 2026-08-20
- Renumbered 2026-09-11: this entry was filed as `KI-TEST-001`, an id already
  held by the open "formerly silent test cases" entry at the top of this
  ledger. Every `KI-TEST-001` citation under `tests/` means that one; a
  citation about cross-file device-state leakage means this one.
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
