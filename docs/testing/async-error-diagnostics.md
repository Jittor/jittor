# Async Error Diagnostics Contract

The existing `TraceData` stack is graph metadata: it is keyed by nodes and
may be disabled by `trace_py_var`. It is not a recent-launch history and must
not be read as one.

The 10.17 implementation should add a bounded per-thread ring of launch
records, each containing:

- operator id/name and fused member ids;
- Python file and line, copied at launch time;
- device/stream identity and a monotonic sequence number.

Recording must be allocation-free on the launch path after initialization and
must tolerate concurrent executor threads. On an asynchronous CUDA failure,
the reporter selects the most recent records for the failing stream and prints
the operator name plus Python location. Missing records are a visible
`not-found` diagnostic, never a silently empty success.

The implementation cannot be fully accepted on a CPU-only host: the static
contract may validate record layout, bounded capacity, overwrite ordering, and
report formatting, but the final gate requires a CUDA probe that injects an
asynchronous error and checks the reported Python line.

Suggested CUDA acceptance command:

```bash
CUDA_VISIBLE_DEVICES=0 JT_SYNC=0 trace_py_var=3 \
  python -m pytest -q -s tests/backends/cuda/test_async_error_location.py
```
