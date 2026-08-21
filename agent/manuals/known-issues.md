# Active Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-08-21
- Baseline: `85aad9f3`
- Owner: Jittor core maintainers
- Review cadence: on every strict XPASS, related fix, or quarterly maintenance

This ledger contains reproduced, currently relevant defects and explicit
limitations. Historical fixes remain in Git and dated `agent/results/` reports;
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

## KI-COMPILER-001: parallel compiler can corrupt process state

- Severity: High
- Status: Open; root cause unproven
- Owner: compiler/executor maintainers
- Evidence: [investigation and reproduction](../../docs/development/known-issues/parallel-compiler-segfault.md)
- Workaround: set `jt.flags.use_parallel_op_compiler = 0` for deterministic
  validation workloads
- Additional reproduction: forking the parallel compiler from a Jupyter kernel
  kills the kernel during the first cold compile. `use_parallel_op_compiler=0`
  in [`test_notebooks.py`](../../tests/integration/test_notebooks.py) is the
  current mitigation for the offline CPU tutorial gate.
- Review/expiry condition: remove only after sanitizer-backed root cause and
  repeated cold/warm stress, deadlock, multiprocess-cache, and performance gates

## KI-COMPILER-002: fractional constant padding can break CPU asm tuning

- Severity: High
- Status: Strict expected failure
- Owner: CPU compiler/asm-tuner maintainers
- Evidence: [`test_constant_pad_fractional_fill_cpu_asmtuner`](../../tests/compiler/test_kernel_traps.py)
- Symptom: a fractional constant pad value can be rewritten into malformed C++;
  integer fill values and CUDA are unaffected
- Workaround: use an integral fill when semantically acceptable or disable the
  affected asm-tuner path for the reproduction
- Review/expiry condition: the strict expected failure XPASSes with fractional,
  negative, and representative hexadecimal constants covered

## KI-BACKEND-001: narrow integer reductions lack accelerator atomics

- Severity: High
- Status: CUDA/NPU expected failures
- Owner: reduce and backend maintainers
- Evidence: [`reduce_dtypes.py`](../../tests/opinfo/definitions/reduce_dtypes.py)
  and [device parity](../../tests/backends/parity/test_device_parity.py)
- Symptom: `sum`, `prod`, `max`, and `min` for sub-32-bit integer samples may fail
  to compile because required atomic overloads are absent
- Workaround: promote inputs to a supported width before reduction
- Review/expiry condition: every affected dtype executes and matches the CPU
  reference on each backend, turning all strict expected failures into passes

## KI-BACKEND-002: logical reductions lack CUDA/NPU atomic-bool paths

- Severity: Medium
- Status: Explicit skips
- Owner: reduce and backend maintainers
- Evidence: [`reduce_dtypes.py`](../../tests/opinfo/definitions/reduce_dtypes.py)
- Symptom: native `all`/`any` cannot execute their logical reduction path on the
  listed accelerators
- Workaround: no implicit device fallback is accepted; callers must choose a
  supported representation/path explicitly
- Review/expiry condition: remove skips only after real-device forward and parity
  tests pass

## KI-COMPAT-001: median conflicts with Torch-style `argsort`

- Severity: Critical
- Status: OpInfo and parity skip
- Owner: misc and Torch-compat maintainers
- Evidence: [`ordering_ops.py`](../../tests/opinfo/definitions/ordering_ops.py)
- Symptom: median assumes a native `(index, value)` return while the installed
  Torch-style `argsort` contract returns indices; higher-axis indexing also needs
  correction
- Workaround: use an independently validated sort/select implementation
- Review/expiry condition: enable OpInfo forward, gradient, and device-parity
  coverage for scalar and nonzero dimensions

## KI-NN-001: CPU `DepthwiseConv` backward raises `save_vars`

- Severity: High
- Status: Reproduced failure contract
- Owner: neural-network/autograd maintainers
- Evidence: [`test_cpu_backward_keeps_pre_migration_failure_contract`](../../tests/nn/test_depthwise_conv.py)
- Symptom: CPU forward matches grouped-convolution reference, but differentiating
  through the dedicated operation raises `AttributeError`
- Workaround: use the maintained grouped-convolution path when CPU backward is
  required
- Review/expiry condition: replace the failure assertion with independent input
  and weight gradient checks on CPU, plus CPU/CUDA parity

## KI-DTYPE-001: low-precision elementwise gradients upcast

- Severity: Medium
- Status: Strict CUDA expected failure
- Owner: dtype/autograd maintainers
- Evidence: [`test_elementwise_grad_dtype_KNOWN_DIVERGENCE`](../../tests/backends/cuda/test_low_precision.py)
- Symptom: float16/bfloat16 elementwise backward can return float32 gradients,
  unlike the input-dtype gradient contract
- Workaround: consumers must not assume low-precision gradient dtype until the
  path is corrected
- Review/expiry condition: strict expected failure XPASSes for supported low
  precision dtypes without weakening numerical checks

## KI-SEMANTICS-001: empty-axis mean returns zero

- Severity: Medium
- Status: Strict expected failure
- Owner: reduction semantics maintainers
- Evidence: [`test_mean_over_empty_axis_is_nan_KNOWN_DIVERGENCE`](../../tests/core/test_edge_cases.py)
- Symptom: mean over an empty axis returns zero instead of the NumPy/Torch NaN
  convention
- Workaround: check the reduced extent before calling when NaN semantics matter
- Review/expiry condition: decide and document the public convention; if parity is
  chosen, convert the expected failure to a passing NaN regression

## KI-SEMANTICS-002: fused scalar arithmetic differs from strict float32

- Severity: Medium
- Status: Strict expected failure
- Owner: compiler numerical-semantics maintainers
- Evidence: [`test_large_magnitude_add_precision_KNOWN_DIVERGENCE`](../../tests/core/test_edge_cases.py)
- Symptom: fused/scalar evaluation may preserve a small value that strict float32
  evaluation loses, creating a Torch-parity divergence
- Workaround: materialize or cast at the required precision boundary when exact
  float32 step semantics matter
- Review/expiry condition: adopt and document one precision contract, then make
  the test a normal assertion for it

## KI-OPS-002: integer floor-division backend verification incomplete

- Severity: Critical
- Status: Core fix verified on CPU/CUDA; NPU/ROCm real-device verification pending
- Owner: binary operator maintainers
- Evidence: [`test_floor_divide.py`](../../tests/core/test_floor_divide.py),
  [`sample_floor_divide`](../../tests/opinfo/definitions/pointwise_binary.py), and
  [2026-08-21 verification](../results/2026-08-21-floor-divide.md)
- Previous symptom: C++ integer division made negative quotients truncate toward
  zero instead of flooring toward negative infinity
- Current implementation: shared CPU/CUDA codegen subtracts one exactly when a
  nonzero remainder has the opposite sign from the divisor; fixed vectors pass
  for uint8/int8/int16/int32/int64, and the selected int64 OpInfo samples cover
  negative operands on CPU and CUDA
- Workaround on unverified backends: compare representative negative operands
  against `numpy.floor_divide` before relying on the backend
- Review/expiry condition: pass the same fixed-vector and OpInfo coverage on real
  NPU and ROCm devices, then remove this entry

## KI-COMPAT-002: batched Normal sampling shape divergence

- Severity: Medium
- Status: Explicit skipped regression
- Owner: distribution maintainers
- Evidence: [`test_torch_compat_distributions.py`](../../tests/compat/torch/test_torch_compat_distributions.py)
- Symptom: `Normal.sample(sample_shape)` with batched parameters uses the sample
  shape as the full output shape instead of prepending it to the batch shape
- Workaround: construct the complete output shape explicitly where possible
- Review/expiry condition: enable the skipped test for multiple sample and batch
  ranks, including gradients where applicable

## KI-SEMANTICS-003: same-Var equality mishandles NaN

- Severity: Critical
- Status: Reproduced compiler/fusion behavior
- Owner: graph optimizer and comparison-operator maintainers
- Evidence: [`test_nan_not_equal_to_itself_via_isnan`](../../tests/compiler/test_kernel_traps.py)
- Symptom: comparing a `Var` with itself can be folded to all true, violating the
  IEEE rule that NaN is not equal to itself
- Workaround: use `jt.isnan`/`jt.isfinite` for masks; do not derive a non-NaN mask
  from `x == x`
- Review/expiry condition: add a direct same-object NaN equality regression and
  pass it through fused and unfused CPU/accelerator paths

## KI-SHAPE-001: reductions do not produce 0-D scalar tensors

- Severity: Medium
- Status: Accepted current representation divergence
- Owner: tensor-shape and compatibility maintainers
- Evidence: [`test_no_zero_d_scalar`](../../tests/compiler/test_kernel_traps.py)
- Symptom: a full reduction produces shape `(1,)`, which can introduce an extra
  dimension when stacked or composed with scalar-oriented code
- Workaround: use `.item()` for a Python scalar or reshape explicitly when the
  surrounding tensor contract requires a particular rank
- Review/expiry condition: keep the regression and compatibility adapters until
  a separately reviewed core scalar representation exists

## KI-DTYPE-002: implicit array construction narrows 64-bit NumPy values

- Severity: High
- Status: Accepted current default with explicit escape hatch
- Owner: dtype and compatibility maintainers
- Evidence: [`test_jt_array_float64_narrowing`](../../tests/compiler/test_kernel_traps.py)
- Symptom: `jt.array` can narrow NumPy float64 and int64 inputs to 32-bit defaults,
  invalidating high-precision references or numerical gradient checks
- Workaround: always pass `dtype="float64"` or `dtype="int64"` when width is part
  of the contract
- Review/expiry condition: retain both default and explicit-dtype assertions until
  a public dtype-default decision changes them together

## KI-FFT-001: CUDA rFFT has a sequence-sensitive correctness risk

- Severity: Critical
- Status: Under investigation; reproduced in an aggregate probe but not minimized
- Owner: FFT and Torch-compat maintainers
- Evidence: [2026-07-05 CUDA complex audit](../results/2026-07-05-complex-cuda-audit.md)
- Symptom: after earlier complex forward/gradient work in the same process,
  `rfft` may lose half-spectrum imaginary components and break `irfft(rfft(x))`,
  while an isolated test passes
- Workaround: validate the round trip in the actual process sequence; use process
  isolation for correctness-critical runs until the trigger is minimized
- Review/expiry condition: first land a deterministic sequence regression; close
  only after the current revision passes repeated clean and aggregate-process
  CUDA runs, or after a result report disproves the old reproduction

## KI-COMPLEX-001: native complex capability gaps

- Severity: Research/High by operation
- Status: Explicit unsupported contracts
- Owner: dtype, autograd, and linear-algebra maintainers
- Evidence: [native complex dtype decision](../../docs/architecture/complex-dtype.md)
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
- What the affected tests really show: `tests/compiler/test_parallel_pass.py`
  fails 3 cases in a CPU-only build and 8 in a CUDA one, and the CUDA set is
  the same cases plus their CUDA class variants. The shared failures are
  numerical -- `assert np.allclose(a.data*2, b)` on a reduce under the parallel
  pass -- so they are a real defect to chase, unrelated to logging.
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
- Guard: [cache path precedence](../../tests/compiler/test_cache_path_precedence.py)

## KI-COMPILER-005: the parallel pass mis-partitions work at some thread counts

- Severity: High (silently wrong results, or a hang, depending on thread count)
- Status: Reproduced and bounded; not fixable from this repository
- Owner: compiler maintainers
- Evidence: `tests/compiler/test_parallel_pass.py` -- `TestParallelPass3::test`
  fails on `assert np.allclose(a.data*2, b)`
- Symptom: with `compile_options={'parallel':1, 'merge_loop_var':0}` and a
  16-per-dimension tensor, `a+a` for `ndim>=3` leaves most of the output
  untouched -- 3072 of 4096 elements stay 0. The generated kernel derives its
  tiling from the runtime thread count:

      int tn1 = get_thread_range_log(thread_num_left, range1);
      int tn0 = get_thread_range_log(thread_num_left, range0);
      int tnum0 = 1<<(tn0-tn1);

  Each call consumes bits from the remaining thread budget, so with 64 threads
  and two dimensions of 16 it yields `tn1=3, tn0=2` and shifts by -1, which is
  undefined behaviour. Measured on this host: 128 threads hangs past a
  600s timeout, 64 and 32 both compute wrong values.
- Reach: not confined to the opt-in `parallel` compile option. 2182 of the 4958
  JIT kernels in a populated cache contain this `tnum` partitioning, so it is
  the ordinary CPU parallelisation path.
- Why it cannot be fixed here: `ParallelPass::run` has no source in the
  repository. `python/jittor/src/opt/pass/parallel_pass.h` declares it and
  `opt/pass_manager.cc` calls `run_pass<ParallelPass>()`, but the definition
  ships as `python/jittor/utils/data.gz`, a gzipped blob that `compiler.py`
  expands, compiles to `<cache>/data.o` with `-include src/utils/vdp`, and then
  deletes. `vdp` is `#define _P(...)`, which erases the decoy lines the blob is
  padded with. That object supplies 31 functions, including this pass.
- Consequence for this branch: the OpenMP thread-count default cannot be
  lowered to the physical core count, even though doing so measured 1841 GFLOPS
  against 406 on a 2048x768 by 768x768 matmul and 380us against 5955us on a
  batched oneDNN call. That change was reverted for this reason.
- Review/expiry condition: close when the tiling no longer depends on the
  thread count being large enough, which needs the pass's source
