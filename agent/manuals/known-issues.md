# Active Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-08-22
- Baseline: `866914d4`
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
  [SIGCHLD verification and addendum](../results/2026-08-21-jupyter-sigchld.md).
- Review/expiry condition: remove only after sanitizer-backed root cause and
  repeated cold/warm stress, deadlock, multiprocess-cache, and performance gates

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

## KI-SEMANTICS-003: floating-comparison backend verification incomplete

- Severity: Critical
- Status: Core fix verified on CPU/CUDA; NPU/ROCm real-device verification pending
- Owner: compiler and comparison-operator maintainers
- Evidence: [`test_nan_self_comparisons_across_dtypes`](../../tests/compiler/test_kernel_traps.py),
  [`test_float_comparisons_with_nan`](../../tests/ops/test_fusion_correctness.py),
  and [2026-08-21 verification](../results/2026-08-21-ieee-nan-comparisons.md)
- Previous symptom: CPU JIT kernels inherited `-Ofast`, allowing both same-object
  and distinct floating comparisons to violate IEEE NaN behavior; low-precision
  `!=`, `<=`, and `>=` could also fail to compile on CPU
- Current implementation: floating and complex comparisons retain optimized
  `-O3` kernels without finite-math assumptions, and fused compile options are
  taken from the complete aggregated graph choices
- Workaround on unverified backends: compare representative NaN values against
  NumPy before relying on direct comparison masks
- Review/expiry condition: pass the same dtype and fused/unfused matrices on real
  NPU and ROCm devices, then remove this entry

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

## KI-FFT-001: withdrawn -- current CUDA sequence regression is clean

- Severity: n/a
- Status: Withdrawn 2026-08-21; the old aggregate outcome is not reproducible on
  the current implementation
- Owner: FFT and Torch-compat maintainers
- Evidence: [CUDA rFFT sequence review](../results/2026-08-21-rfft-sequence-review.md)
  and
  [`test_rfft_after_complex_forward_backward_sequence`](../../tests/compat/torch/test_torch_compat_fft_einsum.py)
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
