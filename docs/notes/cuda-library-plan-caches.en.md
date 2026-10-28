# CUDA library plan caches

- Status: implemented
- Last reviewed: 2026-10-05 (baseline `e3c369acb`)
- Owner: CUDA library maintainers
- Review trigger: compute-stream ownership, backend teardown order, or changes to the
  cuFFT/cuTT/cuDNN plan APIs

cuFFT, cuTT and cuDNN all construct a plan (or algorithm selection, execution plan)
before executing, and a single construction costs several milliseconds, so Jittor
caches them. This page explains **who owns** these caches, **when** they are
released, and how to observe them from Python.

## One per device, evicted in creation order

The cuFFT and cuTT caches are **partitioned by CUDA device**: each device's cache
owns its own compute stream, creation-order queue and plan handles, and the
capacity limit is counted per device. Shapes changing endlessly on one device
**never evict another device's working set**. Cache keys are POD structs, and
once the limit is exceeded the earliest-created plan is evicted. The shared
implementation lives in `backends/cuda/include/device_plan_cache.h`
(`DevicePlanCache`).

- **The cache takes ownership as soon as the SDK construction succeeds.** cuFFT
  uses `cufftCreate` followed by `cufftMakePlanMany`, so configuring the
  workspace or binding a stream and failing releases the same handle; it never
  calls `cufftPlanMany` again after `cufftCreate`, which would create a new handle
  (that was the early pattern that leaked a second plan).
- **Both consumption and explicit cache operations are serialized through
  `ExecutorEntryScope`.** Before releasing a plan it waits for the compute stream
  that owns it; other copy/communication streams are not synchronized.
- **`CacheDeviceScope` selects the device that actually allocates and restores it
  on exit.** It does not change the Runtime's placement state and does not trigger
  library switch hooks.
- **Destructors report SDK failures instead of throwing from a finalizer.** cuTT's
  allocation callback keeps the original allocator and device, so cleanup never
  guesses the allocator from the executor's current device; the free callback
  intercepts before an exception crosses cuTT's own `noexcept` destructor, and
  the plan owner records the cleanup as a failure rather than a successful
  destruction.

cuDNN's handle, three legacy algorithm tables and the Backend API plan tables
live in **per-device state** in
`backends/cuda/libraries/cudnn/src/cudnn_wrapper.cc`. The six paths of 2-D and 3-D
convolution obtain their own POD-keyed algorithm tables through the same owner;
the Backend API plan storage is no longer an inline singleton in the header. Each
entry owns its plan and its descriptor dependencies, and is **destroyed before
that device's cuDNN handle**; entries that fail to build are deleted, and
lowering the algorithm cache limit immediately clears the excess tables.

## Convolution goes through the cuDNN Backend API

The legacy cuDNN API redoes the heuristics and internally builds an execution
plan on every call: on cuDNN 8.9, `cudnnGetConvolution*WorkspaceSize` plus
`cudnnConvolution*` costs about 110 µs of CPU time per call regardless of whether
the descriptors are reused. One diffusers UNet2D step makes about 150 such calls,
accounting for 20 of its 42 ms of CPU time.

`backends/cuda/libraries/cudnn/include/cudnn_conv_plan.h` routes the three 2-D
convolution operators (forward, backward to input, backward to weight) through
the Backend (graph) API first: keys are formed from kind, shape, stride, dtype,
parameters and numeric tier, and each key builds its plan only once (heuristic
mode B, then A, then the fallback list; engines that require runtime compilation
are skipped; for fp32 operands, when tensor ops are not allowed, engines that use
tensor cores or down-convert the inputs are also skipped), after which each
execution costs about 12 µs. When `cudnn_benchmark` is set, up to six candidates
are timed on real buffers like legacy `cudnnFind`. Requests the Backend rejects
are remembered and handed to the legacy path.

The payoff: a UNet2D step goes from 1.48× PyTorch down to 1.10×. The cost is the
few-millisecond build per new key, which **workloads whose shapes never repeat
pay every time** — hence the cache bound and the retained legacy path.

The legacy convolution API also requires the host-side scaling scalar to match the
tensor's scaling ABI: double tensors use double, everything else uses float. Six
call sites share `CudnnScalingType`; previously a float pointer passed for a double
convolution produced wrong values.

## Observation and cleanup

| Library | Query | Cleanup |
| --- | --- | --- |
| cuFFT | `cufft_plan_cache_size(device)`, `cufft_plan_build_count(device)`, `cufft_plan_destroy_count(device)`, `cufft_plan_destroy_failures(device)` | `cufft_clear_plan_cache(device)`; `cufft_set_plan_cache_size(n)` sets the per-device limit |
| cuTT | the four same-named `cutt_*` counters | `cutt_clear_plan_cache(device)`, `cutt_set_plan_cache_size(n)` |
| cuDNN | `cudnn_algorithm_cache_size(device)`, `cudnn_plan_cache_size(device)`, `cudnn_plan_destroy_count(device)` | `cudnn_clear_algorithm_cache(device)`, `cudnn_clear_plan_cache(device)` |

- Omitting `device` (default `-1`) makes queries aggregate across all devices and
  cleanup act on all devices.
- **Counters persist after explicit cleanup**, so callers can distinguish
  "reuse", "rebuild" and "destroy failure".
- `cudnn_plan_cache_size` counts only **valid SDK plans**, not remembered
  "Backend unsupported" requests; `cudnn_plan_destroy_count` counts only
  successful SDK destructions.
- Process exit destroys every library, and can be called repeatedly.

## Verification

- `tests/backends/cuda/test_plan_cache_lifetime.py` performs real cuFFT, cuDNN and
  cuTT operations on **two devices**: reuse, per-device independence, targeted and
  repeatable cleanup, restoration of the original CUDA device, rebuilds, and
  forward/gradient references against NumPy. The cuDNN Backend API case requires
  three live plans and exactly three successful destructions, so a **fallback to
  the legacy path cannot satisfy the plan-lifetime assertions**.
- `tests/backends/cuda/test_plan_cache_bounds.py` guards that the cuFFT/cuTT
  caches stay bounded; `tests/backends/cuda/test_cudnn_conv_plan.py` guards the
  Backend API convolution path.
- The cuTT cases call `cutt_transpose` explicitly: native `transpose` can now be a
  storage view and **must not be taken as evidence that a cuTT plan executed**.
  The plan-miss overlap case checks that an unrelated communication stream is
  still pending.
- When rerunning these cases, use matching core and library headers; cached
  custom libraries cannot be interchanged after the core ABI changes.
