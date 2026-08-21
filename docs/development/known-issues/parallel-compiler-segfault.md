# Parallel Operator Compiler Segmentation Fault

- Status: Open for non-Jupyter workloads; Jupyter SIGCHLD path fixed
- Last reviewed: 2026-08-22
- Baseline: `137f9dd1`
- Owner: compiler/executor maintainers
- Workaround: `jt.flags.use_parallel_op_compiler = 0`
- Exit condition: a minimized stress test passes repeatedly with parallel
  compilation enabled and no deadlock, cache corruption, or material regression

## Symptom

Some large model and device-parity workloads can terminate with allocator or
segmentation faults while multiple fused operations compile concurrently. The
failure is timing-sensitive and may surface after the compilation that damaged
state, so the last printed operator is not reliable attribution.

Serial compilation has been used successfully by dataset, ACL, and parity test
paths. This is a containment measure, not evidence that runtime operator
execution is defective.

## Resolved Jupyter subcase

The former Jupyter reproduction had a separate, proven cause. The parallel
operator compiler uses `std::thread`; it does not fork compiler workers. Jittor
nevertheless installed a process-wide `SIGCHLD` handler inside an ipykernel and
quick-exited the host whenever any child was killed by a signal. A minimal
process containing only `import jittor` and an unrelated `SIGKILL`ed subprocess
reproduced the exit with `si_code=CLD_KILLED` and `si_status=SIGKILL`.

Jupyter kernels now keep their existing SIGCHLD disposition, just as they
already kept SIGINT, while Jittor retains SIGILL and SIGBUS fault diagnostics.
The deterministic host-ownership regression and focused CPU/CUDA cold-operator
probes pass. A later complete nbclient smoke nevertheless died twice during a
transformer attention workload with eight compile workers, including once with
`JT_NO_SIGNAL_HANDLER=1`; the same smoke passed with serial compilation. The
offline notebook gate therefore remains serial while this broader issue is
open. See the [2026-08-21 verification report and 2026-08-22 addendum](../../../agent/results/2026-08-21-jupyter-sigchld.md).

## Current hypothesis

The leading hypothesis is a missing synchronization boundary between parallel
compiler workers and process-level compilation/cache locking. A worker may
observe lock state established by the owning process and incorrectly treat
shared compiler or relay state as protected. Concurrent mutation could then
surface as heap corruption.

For non-Jupyter workloads this remains a hypothesis. The ownership of relay
groups, the exact shared mutable object, and the first invalid access have not
been demonstrated with a sanitizer trace. Adding one broad mutex is therefore
not an accepted fix: it may deadlock nested compilation, eliminate intended
parallelism, or fail to protect multi-process cache access.

## Reproduction protocol

Use a dedicated state directory and preserve the exact commit, compiler, device,
and environment:

```bash
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/parallel-compiler/repro/jittor-home"
export cache_name=parallel-compiler-repro
python -m pytest -v tests/backends/parity/test_device_parity.py
```

Run once with the default parallel compiler setting and once with:

```python
import jittor as jt

jt.flags.use_parallel_op_compiler = 0
```

Do not run the two variants concurrently or against the same cache. A useful
reproduction records whether the cache was cold, worker count, the last
completed compile, signal/backtrace, and whether repeated serial runs are clean.

## Investigation plan

1. Minimize the workload while retaining a cold-cache failure.
2. Instrument compiler task creation, relay-group ownership, cache lock
   acquisition, and worker completion with stable identifiers.
3. Run the minimized native compiler path under AddressSanitizer or ThreadSanitizer.
4. Identify the first invalid access or race before changing synchronization.
5. Apply the narrowest ownership or locking fix and add a deterministic stress
   regression.

## Acceptance gate

A fix must demonstrate all of the following:

- repeated cold- and warm-cache stress runs no longer crash;
- a timeout-backed test shows no deadlock;
- two processes using separate and shared cache configurations do not corrupt
  artifacts;
- parallel compile time does not regress materially against the recorded
  baseline;
- compiler, device-parity, and representative model tests pass with parallel
  compilation restored;
- the serial workaround and ledger entry are removed in the same change.

Until that evidence exists, callers that prioritize deterministic validation may
disable the parallel compiler explicitly and should report that choice with
their results.
