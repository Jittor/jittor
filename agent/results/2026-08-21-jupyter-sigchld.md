# Jupyter SIGCHLD 与并行编译复核

- Status: Jupyter subcase fixed; broader KI-COMPILER-001 remains open
- Date: 2026-08-21
- Baseline: `3fc3f6fd`
- Owner: compiler, runtime, and integration-test maintainers
- Review when: process-level signal registration, ipykernel launch, compiler
  subprocess management, or parallel operator compilation changes

## Finding

A real nbclient kernel with eight operator compile workers and 32 cold
floating-point graphs reproduced `DeadKernelError: Kernel died` in about seven
seconds. The same graphs passed when operator compilation was serial. They also
passed with eight workers in an ordinary Python process, so the compiler workload
alone was not sufficient to reproduce the failure.

Three independent controls isolated the process-level signal handler:

- `JT_NO_SIGNAL_HANDLER=1` kept the Jupyter kernel alive;
- suppressing Jittor's child-signal registrations, then restoring eight compile
  workers after import, also passed;
- restoring only the pre-Jittor `SIGCHLD` disposition kept the kernel alive.

The deterministic minimum did not need JIT compilation: after `import jittor`,
an unrelated subprocess killed itself with `SIGKILL`. Before the change, the
parent exited with status 1 and logged `si_code=2`, `si_status=9`. Jittor's
handler treated every signalled child as its own failed worker and called
`quick_exit`; it did not check child ownership. This explains the Jupyter death
without invoking the unproven shared-compiler-state hypothesis.

## Change

When `JPY_PARENT_PID` is present, Jittor no longer installs its `SIGCHLD`
handler. It already used the same host marker to preserve Jupyter's SIGINT
ownership. SIGILL and SIGBUS remain registered for faults in the Jittor process,
and the non-Jupyter dataset-worker failure contract is unchanged.

The offline notebook smoke test no longer disables the parallel operator
compiler. It explicitly selects eight workers and asserts that value in the
kernel's first cell. A subprocess regression kills an unrelated child under a
Jupyter host marker and requires both the `-SIGKILL` child status and a live
parent marker.

## Verification

CPU work used Python 3.11.15 for real nbclient/ipykernel execution and Python
3.12.13 for repository tests. CUDA work used Python 3.12.13, CUDA 12.2, and an
RTX 4090. Caches, homes, temporary files, and logs were isolated below
`$JITTOR_LAB_ROOT/_state/`. Core rebuilds were serialized before cold operator
compilation was enabled.

| Gate | Result |
| --- | --- |
| Pre-change real nbclient, 32 cold CPU graphs, 8 workers | kernel died |
| Post-change real nbclient, 32 new cold CPU graphs, 8 workers | passed; kernel survived next cell |
| Deterministic Jupyter SIGCHLD ownership regression | `1 passed` |
| Real CUDA host marker, unrelated killed child + 32 cold graphs, 8 workers | passed; all arrays matched NumPy |
| Non-Jupyter dataset worker-death contracts | `2 passed`, 11 deselected |
| Parallel fusion and inf/NaN ternary regression | `6 passed`, 18 subtests passed |
| Signal ownership plus compiler log tests | `2 passed` |
| Repository layout | passed |
| `tests/structure` | `213 passed`, 1511 subtests passed |

The change does not alter compiler scheduling, generated kernels, or the worker
count, so no compile-speed claim is inferred. It removes the forced-serial
notebook path; no standalone performance number is reported from the short
correctness probes.

## Maintained commands

With the environment and isolated cache configured according to
`agent/manuals/environment.md`:

```bash
python -m pytest -q tests/compiler/test_signal_handlers.py
python -m pytest -q tests/integration/test_notebooks.py::test_notebook_smokes_execute_offline_on_cpu
python -m pytest -q tests/data/test_dataset.py -k children_died
python -m pytest -q tests/ops/test_fusion_correctness.py
```

## Limits

This closes only the Jupyter reproduction. KI-COMPILER-001 remains open for the
older timing-sensitive large-model and device-parity reports because they still
lack a minimized sanitizer trace. The serial compiler workaround remains valid
for deterministic validation outside Jupyter. NPU and ROCm were not available;
the accelerator evidence here is CUDA only.
