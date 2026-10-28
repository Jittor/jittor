# Performance and memory profiling: `jt.profile`

- Status: maintained
- Corresponding commit: `wip/profiler-rework` branch (baseline `cb93ba04`)
- Owner: Jittor core maintainers
- Review trigger: executor batch/operator launch paths, allocator (SFRL/Temp)
  accounting, graph replay, or CUPTI version changes

When a training step is slow or uses a lot of memory, the questions are always the
same few: where does host time go (Python graph construction, executor planning,
per-operator launching, or waiting for the device)? Which operators, which kernels,
take the device time? Is this step host-bound or device-bound? What is the memory
peak, who was alive at that moment, and who created them?

`jt.profile` answers these questions through one entry point:

```python
import jittor as jt

with jt.profile() as prof:          # 默认：设备时间（有 CUPTI 时）+ 显存
    train_step()
print(prof.summary())               # 主机拆分、设备拆分、瓶颈判断、算子/kernel/调用点表、显存
prof.export_chrome_trace("step.json")   # chrome://tracing 或 Perfetto 打开
```

In Torch compatibility mode, `torch.profiler.profile` and
`torch.autograd.profiler.profile` are exactly this: `key_averages().table()`,
`export_chrome_trace`, `schedule`/`on_trace_ready` and `record_function` all work.

## What it measures, and how

**It does not change what it measures.** No synchronization is inserted inside the
region, no operator is rerun, no compile option is changed. The old
`jt.profile_scope` did the opposite — synchronize after every operator, optionally
rerun — so it answered "how fast does this kernel run on its own", not "where did
this step's time go" (see "Legacy tools" below). By default the region does one
`jt.sync_all(True)` at each end, so work queued outside does not leak in and work
built inside finishes before the report; `sync=False` turns that off.

### Host breakdown

The executor timestamps every batch, every device wait and every launched operator
in `src/runtime/profiler/step_trace.{h,cc}` (CLOCK_MONOTONIC, the same clock as
Python's `time.perf_counter_ns`). The region's wall-clock time is **partitioned**
into (the parts sum to the wall clock):

| Part | Meaning |
| --- | --- |
| Python + graph construction | wall clock minus executor and graph launch: Python code, frontend, pyjt entry, Op/Var construction |
| Executor planning | `build_exec_plan`: traversal, fusion, scheduling |
| JIT compilation / kernel lookup | `parallel_compile_all_ops`; on a cache hit this is the table-lookup cost |
| Operator launch | time in the execution loop other than device waits: allocation, preparation, launching |
| Waiting for device | `DeviceWaitScope`: end-of-batch synchronization, synchronization before CPU operators, blocking copies |
| Device graph launch | `graph_launch` (graphs recorded by `jt.graph_replay`) |

Each operator also records its "allocation finished" timestamp, so you can see how
much of launch time was allocation. Device waits that happen inside an operator are
listed separately as that operator's `wait us` — an operator that synchronizes when
it should not shows up directly in that column.

`python=True` starts a separate sampling thread that attributes host time to Python
lines using the same rules as `Op::launch_origin` (skipping frames from `jittor.*`
and `torch.*` modules). Sampling needs the GIL and temporarily sets the interpreter
switch interval to the sampling interval; it is statistical: lines with a handful of
samples are noise, so run several steps and look again.

### Device breakdown

With CUPTI, every kernel, memcpy and memset arrives with device timestamps. When the
executor launches an operator it pushes the operator's index as the CUPTI external
correlation id, so every kernel (including those inside cuBLAS/cuDNN and inside
device graphs) is attributed to the operator that launched it; the report's
"attributed to operators" is the attributed fraction, and unattributable work is
recorded as "(device work outside any operator)", for example input/output copies
that graph replay performs in Python. Device busy time is the union of intervals.

Bound verdict: device busy ≥ 80 % of wall clock is device-bound; < 50 % is
host-bound, with the split of host time between Python/executor/waiting; in between
is mixed. Without CUPTI it falls back to estimating from the "waiting for device"
fraction, marked *(estimated)*.

CUPTI is an optional dependency (extra `profile`); `jt.profile` works without it,
just without device time. Lookup order: `$JITTOR_CUPTI_LIB`, the
`nvidia-cuda-cupti-cu12` wheel (`site-packages/nvidia/cuda_cupti`), the CUDA
toolkit's `extras/CUPTI` used for compilation, then the loader path. When it cannot
be found, the report says "device timing unavailable" with the reason, and does not
invent numbers; `device=True` raises instead.

```bash
pip install "jittor[profile]"   # nvidia-cuda-cupti-cu12==12.2.142，对应 jittor[cuda12] 的 CUDA 12.2
```

With a different CUDA runtime, install the same version of `nvidia-cuda-cupti-cu12`
or point `JITTOR_CUPTI_LIB` at the corresponding `libcupti.so`.

### Memory

With `memory=True` (the default), every SFRL pool allocation/release, every
TempAllocator workspace allocation/release, and every pool request to/return to the
driver records a timestamped event; pool allocations also carry the Var's
dtype/shape, the operator that produced it (`name_ex`), and the Python line that
built that operator. At the start of the region, the graph is traversed from the
held Vars and already-live allocations are recorded as "allocated before profile"
(parameters, optimizer state).

Replaying this log yields:

- **the exact peak** and its moment. It must equal the pool's own
  `device_memory_peak` (reset at the start of the region); when they differ the
  report writes both;
- **the peak including workspaces**: cuDNN/cub workspaces go through TempAllocator
  and are not in `device_memory_peak`; this adds them back;
- **the allocations alive at the peak moment**, grouped by producing operator,
  Python call site, tensor (dtype+shape), and before/inside the region;
- the pool's retained peak (new counter `jt.core.device_memory_reserved_peak`), the
  number of cached blocks at the end of the region, the largest free block, and the
  fragmentation ratio (`1 - largest free block / total cached`);
- this process's usage as seen by NVML (start, end, and a peak sampled about every
  2 ms), and the "outside the pool" portion (CUDA context, modules, library handles).

## Result object

Beyond `prof.summary()`, the structured data is in:

| Attribute/method | Content |
| --- | --- |
| `prof.host` | the parts in the table above (ns), batch count, operator count |
| `prof.device` | `busy`, `kernel`, `copy`, `kernels`, `attributed`, `utilization`; `None` without CUPTI |
| `prof.bound` | `(verdict, basis)` |
| `prof.op_stats(group_by_shapes=False)` / `prof.table(sort_by=...)` | per operator: call count, host time, wait, device time, kernel count, input shape, call site |
| `prof.kernel_stats()` / `prof.kernel_table()` | per kernel: count, device time, which operators launched it |
| `prof.site_stats()` / `prof.site_table()` | per Python line: operators built, launch time, device time, sampled host time |
| `prof.memory` | `MemoryReport`: `peak_allocated`, `peak_with_workspace`, `pool_peak_reserved`, `live_by("op"/"site"/"tensor"/"origin")`, `fragmentation`, `timeline` |
| `prof.replay` | device graph launches inside the region; count deltas and refusal reasons for registered replay objects |
| `prof.op_records` / `kernel_records` / `copy_records` | raw records |

`prof.step()` marks a step boundary. `jt.profiling.record_function(name)` marks a
host interval in the trace.

## Graph replay

Once `jt.graph_replay` has recorded a device graph, a step is a single
`graph_launch` and the executor does not run — so the old per-operator profile is
empty. Now `graph_launch` itself is a record (`cuda_graph_launch`), and the kernels
inside the graph are attributed to it through correlation; `GraphReplay` instances
register automatically, and the report gives their in-region `stats` deltas and
refusal reasons. Other capture/replay mechanisms can enter the report by calling
`jittor.profiling.register_replay_source(obj)` (the object needs a `stats` dict and
optionally `refused`/`_graph_refused`).

## Overhead

Same process, same cache, best of three over 10–20 steps (RTX 4090):

| Workload | No profiling | Host only | Default (+CUPTI+memory+shape) | Default + Python sampling |
| --- | --- | --- | --- | --- |
| MLP training, device-bound (2.3 ms/step) | 2.287 | 2.294 (+0.3 %) | 2.344 (+2.5 %) | — |
| Torch-mode small GPT-2, host-bound (10.1 ms/step) | 10.10 | 10.31 (+2 %) | 11.86 (+17 %) | 12.16 (+20 %) |

On host-bound workloads the overhead comes mainly from CUPTI's runtime API
recording (correlation needs it). Because the host breakdown's "Python + graph
construction" is "wall clock minus executor", the profiling overhead itself lands
in that part; use `device=False, memory=False` for precise host-time comparisons.
The first step of a single-step profile includes CUPTI's one-time enablement cost;
multi-step profiles amortize it.

## Boundaries

- The call-site rule skips all `jittor.*`/`torch.*` frames: for models inside
  native `jittor.models`, the call site lands on the user's line that calls the
  model; Transformers/Diffusers models in Torch mode land on library source lines.
- Under lazy execution, operators launch at synchronization; `record_function` only
  brackets the host interval and immediate launches inside it (device graphs, for
  example), and lazily built operators are attributed by call site.
- `profile` does not nest. The host breakdown is computed from one timeline; the
  case of multiple Python threads driving the executor at once is not verified.
- Records are truncated beyond 2^20 per category, and the report says so.
- Only CUDA has been verified (RTX 4090, CUPTI 12.2); on ROCm/NPU the host breakdown
  and memory sections take the same executor path, but device timing is not
  implemented.

## Legacy tools

| Tool | Status |
| --- | --- |
| `jt.profile_scope` / `jt.profiler` | Retained, positioned as an **operator-level kernel microbenchmark**: synchronizes after each operator, optionally reruns, and reports "launch + kernel + sync" time, not a step. When only device graph launches happen inside the region, a warning points to `jt.profile` |
| `profile_mark` | Retained, only affects `profile_scope`; not rechecked this round |
| `profile_memory_enable` + `get_max_memory_info` / `get_max_memory_treemap` / `get_peak_*` | Retained; restarting now begins from zero (it used to inherit the previous session's peak and Var list). Call sites need `trace_py_var=3`, which is expensive; `jt.profile(memory=True)` is recommended |
| `jt.core.device_memory_used/reserved/peak/reset_device_memory_peak/allocated_total` | Correct, retained; adds `device_memory_reserved_peak`, and `reset_device_memory_peak` now resets both peaks |
| `display_memory_info`, `liveness_info`, `jt.introspection.counters` | Debugging, retained |
| `graph_build_profile_*` | Requires rebuilding core with `JT_GRAPH_BUILD_PROFILE=1`; gives self time of each C++ graph-construction stage. A developer tool, retained |
| `trace_py_var>=2` + `dump_trace_data` | Debugging trace; it synchronizes after every operator, so it cannot be used to measure time |
