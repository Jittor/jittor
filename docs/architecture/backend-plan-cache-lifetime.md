# Device-owned CUDA library caches

- Status: Implemented; reviewed 2026-09-08
- Owner: CUDA library maintainers
- Recheck when: compute-stream ownership, backend teardown or plan APIs change

cuFFT and cuTT keep one bank per CUDA device. A bank owns its compute stream,
creation-order queue and plan handles. Cache limits apply per device; one
device's changing shapes cannot evict another device's working set. POD keys
and creation-order eviction remain unchanged.

`DevicePlanCache` owns a plan as soon as SDK construction succeeds. cuFFT uses
`cufftCreate` followed by `cufftMakePlanMany`, so failure while configuring a
workspace or binding the stream releases that same handle. It does not call
the handle-creating `cufftPlanMany` after `cufftCreate`.

Plan consumers and explicit cache operations serialize through
`ExecutorEntryScope`. Before releasing a plan, cleanup waits for its owning
compute stream. `CacheDeviceScope` selects the actual allocating CUDA device
and restores the previous one. It does not mutate Runtime placement or invoke
library switch hooks. Other copy/communication streams are not synchronized.
Destruction reports SDK failures instead of throwing from a finalizer. cuTT
allocation callbacks retain the original allocator and device, so cleanup
does not guess an allocator from the executor's current device.
The free callback catches allocator failures before they can cross cuTT's own
noexcept destructors; the plan owner records such cleanup as failed rather than
counting it as a successful destruction.

cuDNN owns its handle, three legacy algorithm tables and Backend API plan
table in a device state in `cudnn_wrapper.cc`. The six 2-D/3-D convolution
paths obtain typed POD-keyed tables through that owner. Backend plan storage
is no longer an inline header singleton shared accidentally across devices.
Each entry owns its plan and descriptor dependencies; entries are destroyed
before their device's cuDNN handle. Failed builds are removed, and shrinking
the algorithm-cache limit clears oversized tables immediately. The existing
Backend API admission limit and legacy selection policy are retained.

## Inspection and cleanup

For cuFFT/cuTT, `*_plan_cache_size(device)`, `*_plan_build_count(device)`,
`*_plan_destroy_count(device)` and `*_plan_destroy_failures(device)` describe
one bank. Omitting the device aggregates banks. `*_clear_plan_cache(device)`
clears only that bank; omission clears all banks. Counters survive explicit
clear operations, allowing a caller to distinguish reuse, reconstruction and
failed destruction. Shutdown destroys all banks and is idempotent.

cuDNN provides `cudnn_algorithm_cache_size`/`cudnn_clear_algorithm_cache` and
`cudnn_plan_cache_size`/`cudnn_clear_plan_cache`, with the same device selection.
The plan-size accessor counts live valid SDK plans, not cached unsupported
requests. `cudnn_plan_destroy_count` counts successful SDK plan destruction.

The legacy convolution API also requires host scaling scalars to use the
tensor's supported scaling ABI: double for double tensors, float otherwise.
All six calls use the same `CudnnScalingType`; passing a float pointer to a
double convolution previously produced invalid values.

## Verification

`tests/backends/cuda/test_plan_cache_lifetime.py` exercises actual cuFFT,
cuDNN and cuTT operations on two devices. It verifies reuse, independent banks,
targeted/idempotent cleanup, raw CUDA device restoration, reconstruction and
NumPy forward/gradient references. The cuDNN Backend API test requires three
live plans and exactly three successful destructions, so a legacy fallback
cannot satisfy the plan-lifetime assertions.

cuTT tests call `cutt_transpose` explicitly. Native `transpose` can now be a
storage view and is not evidence that a cuTT plan executed. The plan-miss
overlap test checks that an unrelated communication stream remains pending.
Use the matching core and library headers when repeating these tests; cached
custom libraries are not interchangeable across core ABI changes.
