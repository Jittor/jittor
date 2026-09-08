# 10.17 bounded asynchronous launch origins

- Status: implementation and focused CUDA verification complete
- Integration baseline: `4a7aae78d` plus this change
- Owner: native Runtime / CUDA maintainers
- Review when: executor dispatch, Python binding source capture, stream/error
  handling or graph transformation ownership changes

## Behavior and interfaces

CUDA checked errors now append recent launch candidates with operator name/id,
fused member ids, source file/line, device/stream and a monotonic sequence.
The original CUDA error remains first. A device-wide failure cannot identify
the guilty stream; its report explicitly marks the stream unknown. Known
stream waits/readbacks filter that stream. Candidates are not proof of the
faulting launch, because errors can surface later or propagate through stream
dependencies. Missing, overwritten and truncated evidence is explicit.

Source capture stores only native filename/line values, never Python frames,
code objects, Ops or Vars. Each Op holds an integer source id; fusion,
compilation, execution and both gradient entrances preserve the source.
An explicit unavailable source remains unavailable and cannot be replaced by
the later synchronization/backward caller's line. Launch records copy source
values before graph release. Backend-managed copy/communication streams can
record the active operation on their actual stream. Raw third-party CUDA
launches outside these managed entry points are not independently intercepted.

Native Runtime owns the history. Per-thread rings hold 64 entries, with at
most 64 registered thread slots. Retired slots keep bounded evidence until
overwritten. Source interning admits 8192 distinct locations, never recycles
ids, and counts rejected source captures. Source keys over 4096 bytes are rejected as well,
preventing oversized dynamic filenames from bypassing the table byte bound.
A thread lease shares only native
implementation storage, allowing safe thread exit even if a standalone
history owner has already been destroyed. Recording allocates no memory after
that thread's ring initialization; formatting is confined to errors/queries.

`core.async_launch_history(backend, device=0, stream=-1)` returns a read-only
formatted snapshot. It accepts canonical `cpu/cuda/acl/rocm/corex` identifiers,
validates nonnegative device and stream values (`-1` is the device-wide
sentinel), and does not initialize or query hardware. Root integrated the
formal `jt.introspection.diagnostics.launch_history(backend="cuda", device=0,
stream=None)` facade. Ten host introspection tests pass, including readonly
forwarding, rejected arguments and absence of device queries. Root also reran
the final native history host executable successfully after integration.

## Validation

The real C++ history implementation was compiled in a standalone host harness:

`python -m pytest --noconftest -q
tests/structure/runtime/test_launch_history.py
tests/structure/runtime/test_async_error_diagnostics_contract.py`

**2 passed / 1.46 seconds.** The executable prohibits global allocation for
1000 post-initialization records and checks ring overwrite order, two
concurrent writers, stream/device filtering, caller-owned source-string
destruction, source-table overflow without id recycling, unknown-source
scope restoration, rejected query arguments, no backend callbacks during
queries, and thread-lease teardown after owner destruction. Seven changed
translation units also passed host syntax checks before real CUDA compilation.

After CUDA validation, final review switched its prerequisite to the shared
fail-closed `require_accelerator("cuda")` helper and added the oversized-source
key rejection above. The expanded host checks passed on those final sources;
the successful CUDA path was unchanged and the three GPU nodes were not
repeated solely for these boundary changes.

Actual CUDA command:

`python -m pytest --noconftest -q -s
tests/backends/cuda/test_async_error_location.py`

**3 passed / 0 skipped / 87.92 seconds**, including the final incremental
core/library build. Validation used the existing
`refactor-tensor-placement-integrated-20260908` source/cache path, CUDA
12.2.140, `sm_89`, physical devices 6/7 visible. These tests execute on visible
device 0; they do not claim a two-GPU fault test.

The normal child checks a correct computed result, releases the graph and
forces Python GC, then confirms its original creation file/line remains in
history. The two crash-isolated children first verify a real CUDA result,
launch a kernel that delays before an illegal write, and print
`ISSUED-WITHOUT-WAIT` after `Var.sync()` has returned. With `JT_SYNC=0` and
`trace_py_var=0`, later `sync_all(True)` and `numpy()` respectively must raise
`cudaErrorIllegalAddress` containing the exact `jt.code` creation file/line,
operator name, device and candidate disclaimer. Thus the reported location
cannot be obtained by turning on per-operator debug synchronization.

Unversioned raw logs under the validation state directory:

- `async-error-locations-final.log`: final three-node result above.
- `async-error-locations.log`: first bootstrap process, **1 failed / 2 passed**;
  the first import rebuilt `jit_utils` and deliberately exited with its
  reload-required code. Both asynchronous probes passed there, but a fresh
  process was used for the final result after lifetime/bounding review fixes.

No full-suite, model-performance, NPU/CANN or multi-machine validation is
claimed. The low-overhead contract is bounded storage and allocation-free
recording, not an unmeasured throughput claim. See the maintained
[diagnostic contract](../testing/async-error-diagnostics.md) for the API and
limitations.
