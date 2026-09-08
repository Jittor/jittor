# CUDA plan-cache ownership verification

- Status: Passed focused real-device verification
- Date: 2026-09-08
- Source: backend-cache-lifetime worktree on the post-10.23 layout
- Owner: CUDA library maintainers
- Recheck after: core ABI, CUDA/cuDNN/cuTT, stream or allocator changes

The run used two NVIDIA RTX 4090 devices, CUDA compiler 12.2.140 and cuDNN
8.9.7. An already-built matching core was borrowed from the placement
validation tree; the changed wrappers and all subsequent JIT products used
an independent task cache and lock. No core rebuild was performed.

Results:

- `test_plan_cache_lifetime.py`: five real CUDA cases passed, covering cuFFT,
  all six legacy cuDNN convolution paths, cuDNN Backend API plans and cuTT.
- The final combined run added the cuTT nonblocking-stream miss regression:
  **6 passed, 0 skipped, 39.84 seconds**. It requires actual valid cuDNN plans
  and successful destroy counts, not just entries in an unsupported-result cache.
- Existing cuFFT bounds/error and library-stream regressions passed. Running
  the formerly unreachable cuTT overlap case exposed an obsolete C++ stream
  accessor in the test; the test now uses the current raw stream-handle API
  and passed in the final run.
- POD-key collision tests: **3 passed**. Execution-key source contracts:
  **15 passed**, updated to verify the typed device-cache accessor boundary.
- Final cuTT callback hardening: **1 host negative test passed**. It extracts
  the actual free callback and invokes it inside a vendor-style noexcept
  destructor with a throwing allocator. The callback reports the failure,
  retains the allocator record, and does not terminate the process. This last
  failure-path change was host-checked after the CUDA runs; no further library
  build was started once the borrowed core's source was advanced elsewhere.

The legacy-path numerical test exposed a pre-existing scalar ABI bug:
float64 convolution descriptors received addresses of float alpha/beta values.
After all six call sites switched to the correct scaling type, double-precision
forward, input gradients and weight gradients matched NumPy at 1e-10 tolerance
on both devices. Test inputs explicitly request float64 rather than relying on
the native array constructor's default dtype.

cuTT was not skipped: existing local cuTT 1.2 source was compiled into the
private cache and the tests executed `cutt_transpose` directly. The prior
assumption that this environment could not exercise cuTT is not used as evidence.

Raw build/run logs and the one-off warm-core loader remain unversioned under
the task's `_state/backend-cache-lifetime-*` directory. The decisive log is
`final-lifetimes.log`; it records both the borrowed core and private library
origins. That borrowed validation source was later advanced for a separate
precision task. Repeat verification with matching current core/library builds,
not by combining those old libraries with the newer core headers.
