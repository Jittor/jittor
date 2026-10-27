# Asynchronous error diagnostics contract

The existing `TraceData` stack is **graph metadata**: it is keyed by node and can be turned off
by `trace_py_var`. **It is not recent launch history, and it must not be read as launch
history.**

The native Runtime owns a **per-thread, bounded launch record ring**; each record contains:

- the operator id / name, together with the ids of its fusion members;
- the Python file and line copied at launch time;
- a device / stream identifier and a monotonically increasing sequence number.

After initialization, **records on the launch path are allocation-free**. Each thread holds 64
records, and at most 64 thread slots are registered at the same time; a retired slot keeps its
records until a later writer overwrites them. A mutex-protected snapshot tolerates concurrent
writers. A Runtime-level sequence number orders the records; the report shows the most recent
16 matching entries and explicitly gives **the overwritten count, the omitted count and the
unavailable-thread count**. A thread lease holds the native implementation storage rather than
a bare owner pointer, so thread exit is safe even if a separate history object has already been
destroyed.

The Python file and line are captured at construction and **do not retain a Python frame, code
object, Var or Op**. The source location is an interned native value; a launch copies the
location into the ring before graph nodes may be freed. The source table holds at most 8192
locations, and it **never reuses an id** while an Op may still hold it. Overflow produces
"origin unavailable" and an explicit `unrecorded_origin_captures` count instead of letting
dynamic sources grow without bound. Source keys larger than 4096 bytes are likewise rejected
and counted, so the filename of a dynamic `exec` cannot bypass the table's byte limit.

A fusion record contains at most 8 member ids and gives an explicit total when truncated. Both
the Python path and the operator name have bounded storage. **No origin is invented when no
Python frame is available**: that record prints `not-found`.

CUDA errors automatically attach the most recent launch candidates. Stream waits and readbacks
use the known stream; a device wait or event wait without a known stream reports device-wide
candidates and marks the stream as unknown.

> **A candidate is not proof of "which launch caused the fault"**: an asynchronous error may be
> reported by a later API call, or may propagate along dependencies. The original CUDA error is
> preserved.

`core.async_launch_history(backend, device, stream)` is a read-only native snapshot;
`stream=-1` means all streams on that device. The supported Python observation entry point is

```python
jt.introspection.diagnostics.launch_history(backend="cuda", device=0, stream=None)
```

`None` selects the native "all streams" sentinel and **does not query the device**. It
validates the backend spelling and non-negative device/stream values, and it neither
initializes nor queries the hardware backend during the call (`-1` is the only stream
sentinel).

**Do not test this with `trace_py_var>=2`**: the old graph tracer deliberately synchronizes
after every Op. The new history is unrelated to that tracer and **introduces no additional
synchronization**.

This implementation **cannot be fully accepted on a pure CPU host**: the static contract can
check the record layout, the capacity bounds, the overwrite order and the report format, but
the final gate needs a CUDA probe -- inject an asynchronous error and check the Python line
number the report names.

Suggested CUDA acceptance command:

```bash
CUDA_VISIBLE_DEVICES=0 JITTOR_TORCH_SHIM=1 JT_SYNC=0 trace_py_var=0 \
  python -m pytest -q -s tests/backends/cuda/test_async_error_location.py
```
