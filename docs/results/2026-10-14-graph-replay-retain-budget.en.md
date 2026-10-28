# Automatic replay budgets and no-grad boundaries

Status: dynamic budget restoration verified on CPU and CUDA; default policy
unchanged. CUDA graph-related regressions pass. The generic CPU backend retains
one wide intermediate in an existing regression, identically before and after
this change; this report does not accept its eager-sized retention claim.

Baseline: `1e2ff1aa6b3a391c0c84747413c01bc2731e5626`, after fetching and integrating
`origin/2.0-refactor` at `7ed9b3b52b658d2a3c1ac390eba09087c5e80f95`.
Implementation: the commit containing this report. Maintainer: Jittor core
maintainers. Review when: automatic replay eligibility, working-set accounting,
recording lifetime, allocator policy or backend capture support changes.

## Existing policy and verified defect

The TODO's default/no-grad policy already exists: `61b4cb1db` raised the default
recording budget from 64 to 256 MiB; `29e5998b0` changed accounting from cumulative
allocations to peak working set. Automatic replay remains restricted to repeated,
outermost `no_grad` module calls with at most 4 MiB of Var inputs. Exceeding the
recording budget keeps executor replay available. Zero is unbounded, and explicit
`jt.graph_replay` has no automatic budget.

The minimal CPU and CUDA probes warmed a 1,024-byte graph under the default
budget, then entered `jt.flag_scope(auto_graph_replay_retain_bytes=1)`. The cached
wrapper retained its 256 MiB budget; CUDA kept launching the same recorded graph.
The stateful-object path also kept its old budget. New regression tests failed
before the fix for those two precise cases.

The automatic policy now detects budget changes on the next eligible call,
releases every existing plain/stateful capture through its current owner, and
captures according to the new budget. Restoring a scope's old budget also
recaptures. Tests compare changing-input outputs, assert old capture release and
recording refusal, and prove cache updates occur exactly once. Separate guards
prove a warmed inference capture does not replace a differentiable call and that
dropping its module releases the capture. Default values and explicit replay
semantics are unchanged.

## Measurements and limits

Native Jittor 2.0.0, Python 3.11.16, NumPy 1.26.4, GCC 11.5 and a GeForce RTX 4090
(compute capability 8.9, driver 580.178.04) were used. CUDA compilation used nvcc
12.2.140 and the existing pip-cu122 libraries. CPU explicitly disabled CUDA; this
environment lacks oneDNN, so CPU matmul uses generic kernels. Real CUDA parameter
placement and nonzero device-graph handles were checked. Numerical acceptance
runs forbade backend fallback and observed zero fallback attempts.

Tests and benchmarks used separate CPU/CUDA caches, with first compilation
serial. Timing used `jt.benchmark`, 12 materialized rotating inputs, eight warmups
and 25 synchronized samples. Each table value is the median of three independent
process medians, with baseline/current order alternated. The baseline policy was
a byte-identical copy from the baseline commit, SHA-256
`fcb754d1dfe5e39683a994e6707c766a251a3c487b688477451a92a7f8b147ca`;
only that Python policy owner differed, and artifacts record both source hashes.
The measured native math policy was `default`; input/output agreement was checked
outside timing against each backend's eager execution.

| Small 8x8 matmul/tanh call | Baseline, microseconds | Current, microseconds |
| --- | ---: | ---: |
| CPU, replay disabled control | 34.33 | 32.88 |
| CPU, default replay | 31.01 | 30.87 |
| CUDA, replay disabled control | 35.65 | 35.41 |
| CUDA, default replay | 45.41 | 43.84 |

These toy rows do not establish an overall 4% improvement. CUDA default replay
is slower than eager for the small row; the budget change fixes policy correctness
and does not select an optimal execution mode for every workload.

The wide CUDA call has a 256-byte input, two 32 MiB weights, four successive
32 MiB activations, storage views, and an 8x8 output. Every output was checked
against eager using changing fixed inputs:

| Recording policy | Microseconds | Extra live GPU pool bytes after calls | Extra GPU pool peak |
| --- | ---: | ---: | ---: |
| Replay disabled | 486.45 | 0 | 64 MiB |
| 16 MiB budget: executor replay | 482.57 | 1 KiB | 64 MiB + 2 KiB |
| 64 MiB budget: recorded replay | 451.72 | 64 MiB + 1.5 KiB | 64 MiB + 2 KiB |
| Default 256 MiB: recorded replay | 452.79 | 64 MiB + 1.5 KiB | 64 MiB + 2 KiB |
| Unbounded: recorded replay | 452.13 | 64 MiB + 1.5 KiB | 64 MiB + 2 KiB |

The estimated recording working set is exactly 64 MiB. Private input and output
buffers add fixed overhead beyond that estimate. All policies release extra live
capture storage when the module is dropped; recorded cases report 41 graph
launches, and the 16 MiB case explicitly refuses recording while replaying on
CUDA. These are active-pool measurements, not NVML process peaks or a promise
that total process VRAM is capped. This workload supplies no reason to raise or
lower the existing default.

The CPU wide regression retains 33,555,456 bytes beyond weights. The identical
pre-change policy fails the same less-than-1-MiB assertion with the same byte
count. A separate lifecycle probe confirms module deletion releases it and
returns extra live pool bytes to zero. The failure is preserved, not xfailed or
weakened; generic CPU intermediate reclamation needs a separate fix.

## Regression results

- New boundary cases: CPU four passed/four CUDA skips; CUDA eight passed, no
  skips, fallback count zero. Before the fix, both dynamic-budget CPU cases fail.
- Five neighboring graph replay/capture/step files: CUDA 103 passed with zero
  fallback; CPU 43 passed, 59 accelerator skips, and the unchanged wide retention
  failure described above.
- CPU `core` whitelist: native 98 passed, 41 skipped, one existing xfail; Torch
  224 passed, one accelerator skip. These ran the whitelist from
  `tests/_helpers/tiers.py` in the isolated warm test cache, rather than the
  standard runner's separately provisioned cold sessions.

Repository layout and `git diff --check` passed. Full `tests/structure`:
1,426 passed, four skipped and four failed in 263 seconds. Failures are unavailable
ASV in the isolated environment, the existing ACL gradient-manifest discrepancy,
unregistered result toctree pages and the generated packaging manifest. Shared
index/manifest files were outside this worktree's edit ownership and require the
integration owner to update them. This is not an all-green structure claim;
raw output is `logs/structure.log`.

## Reproduction and evidence

Use a working native environment, pin `PYTHONPATH` to the checkout's `python`,
and isolate `HOME`, `JITTOR_HOME` and `TMPDIR` per backend. For CPU use
`JT_USE_CUDA=0 JT_BUILD_NVCC_PATH=''`; for CUDA select an allocated GPU and set
`JT_USE_CUDA=1 JT_BUILD_NVCC_PATH=<nvcc>`. Run:

```bash
JITTOR_TORCH_SHIM=0 python -m pytest -q --rootdir=. \
  tests/core/test_graph_replay_retention.py \
  tests/core/test_graph_replay.py tests/core/test_graph_capture.py \
  tests/core/test_step_capture.py tests/nn/test_graph_replay_multi_output.py
```

Before/after probes, the exact baseline policy copy, measurement scripts, raw
samples and logs are unversioned under
`$JITTOR_LAB_ROOT/_state/r19-graph-replay/`. Key evidence:
`logs/baseline-{cpu,cuda}.log`, `logs/negative-budget-cpu.log`,
`logs/positive-boundaries-{cpu,cuda}.log`, `logs/graph-neighbors-{cpu,cuda}.log`,
`logs/baseline-wide-cpu.log`, `logs/wide-cpu-lifetime.log`, and `artifacts/*.json`.
Baseline and current outputs were independently checked in every timed process;
this is an internal policy comparison, not a PyTorch performance claim.
