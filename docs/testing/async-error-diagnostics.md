# Async Error Diagnostics Contract

The existing `TraceData` stack is graph metadata: it is keyed by nodes and
may be disabled by `trace_py_var`. It is not a recent-launch history and must
not be read as one.

The native Runtime owns a bounded per-thread ring of launch records, each containing:

- operator id/name and fused member ids;
- Python file and line, copied at launch time;
- device/stream identity and a monotonic sequence number.

Recording is allocation-free on the launch path after initialization. Each
thread has 64 records, with at most 64 simultaneously registered thread slots;
retired slots retain their records until subsequent writers overwrite them.
Mutex-protected snapshots tolerate concurrent writers. A Runtime-wide sequence
orders records; reports show the most recent 16 matching entries and explicit
overwrite, omission and unavailable-thread counts. Thread leases retain native
implementation storage, not a raw owner pointer, so thread exit is safe even
when a standalone history object has already been destroyed.

Python file and line are captured at construction without retaining Python
frames, code objects, Vars or Ops. Source locations are interned native values;
the launch copies the location into the ring before graph nodes may be freed.
The source table admits at most 8192 locations; ids are never reused while Ops
may hold them. Overflow produces unavailable origins and an explicit
`unrecorded_origin_captures` count rather than unbounded dynamic-source growth.
Source keys longer than 4096 bytes are also rejected and counted, so dynamic
`exec` filenames cannot bypass the byte bound of the table.
Fused records include up to eight member ids and an explicit total when
truncated. Python paths and operator names have bounded storage. No source is
invented when no Python frame was available: that record prints `not-found`.

CUDA errors automatically include recent launch candidates. Stream waits and
readbacks use the known stream; device or event waits without a known stream
report device-wide candidates and label the stream unknown. A candidate is
not proof of which launch caused the fault: asynchronous errors can be
reported by later API calls or propagate through dependencies. The original
CUDA error is retained. `core.async_launch_history(backend, device, stream)`
is a read-only native snapshot; `stream=-1` requests all streams on a device.
The supported Python observation is
`jt.introspection.diagnostics.launch_history(backend="cuda", device=0, stream=None)`;
`None` selects the native all-streams sentinel without querying the device.
It validates backend spelling and nonnegative device/stream values without
initializing or querying a hardware backend (`-1` is the only stream sentinel).

Do not use `trace_py_var>=2` for this test: the legacy graph tracer deliberately
synchronizes after each Op. The new history is independent of that tracer and
does not add synchronization.

The implementation cannot be fully accepted on a CPU-only host: the static
contract may validate record layout, bounded capacity, overwrite ordering, and
report formatting, but the final gate requires a CUDA probe that injects an
asynchronous error and checks the reported Python line.

Suggested CUDA acceptance command:

```bash
CUDA_VISIBLE_DEVICES=0 JITTOR_TORCH_SHIM=1 JT_SYNC=0 trace_py_var=0 \
  python -m pytest -q -s tests/backends/cuda/test_async_error_location.py
```
