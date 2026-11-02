# Performance and memory profiling: `jt.profile`

[Section contents](index.en.md) · [中文](profiling.md)

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
`torch.autograd.profiler.profile` are exactly this, and what they produce is what
PyTorch 2.11 produces (`compat/torch/profiler.py`):

- `events()`/`key_averages()` are PyTorch's `FunctionEvent`/`FunctionEventAvg`:
  each operator and each `record_function` range is a CPU event, each kernel/copy
  a CUDA event whose time is also its launching operator's self device time; a
  range's total is its own time plus its children's. Under lazy execution an
  operator **built** inside a range often runs after the range closes; it is still
  the range's child (the executor records the range open when the op was built,
  `op_records[i]["range"]`).
- `table()` has PyTorch's `_build_table` columns, widths, truncation, units, sort
  keys and footer; `profile_memory` gives each operator's allocated-minus-freed
  CPU/CUDA Mem (self and total), `record_shapes` + `group_by_input_shape` give Input
  Shapes as `[[64, 256], [512, 256]]`, `with_stack` + `group_by_stack_n` give Source
  Location (the Python stack that built the operator, `file(line): func`, innermost
  first, without jittor's own frames). Kernel names are demangled.
- `export_chrome_trace` writes Kineto's format (top-level keys, `cpu_op`/
  `user_annotation`/`kernel`/`gpu_memcpy`, ac2g flows, `[memory]` instants,
  `GPU n`/`stream n` metadata, kernel grid/block/registers/shared memory), which
  TensorBoard's profiler plugin, HTA and Perfetto read as a PyTorch trace;
  `export_stacks` writes the flame-graph format; `export_memory_timeline` writes
  `[times, sizes]` (json/json.gz) or the matplotlib plot (html) in PyTorch's layout,
  but jittor does not classify allocations into parameters/gradients/activations,
  so every byte is "Unknown".
- Not there: Python function events (`python_function`), CUDA runtime API events,
  `with_flops`, `with_modules`. Native `jt.profile(stacks=True)` gives the same
  stacks in `result.stacks`.

## Read the report and choose the next step

Start with a working training step and warm it up separately if you want a steady-state profile. `train_step()` in the example stands for that existing function. Export traces outside the source tree.

| Report | Next step |
| --- | --- |
| Host work dominates | Inspect Python construction and executor planning; try host-only profiling with `device=False, memory=False` |
| Device work dominates | Inspect the operator and kernel tables |
| Memory is the constraint | Inspect `prof.memory` and follow [memory optimization](memory-optimization.en.md) |
| Device timing is unavailable | Check CUPTI availability below; host time alone is not measured device time |

Use profiling to attribute cost, then measure the change without the profiler. CUDA device timing needs CUPTI; ROCm/NPU device timing is not implemented here. The reference below explains synchronization, attribution and API fields.

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
| `prof.stacks` | with `stacks=True`, stack id → frames (indexed by `op_records[i]["stack"]`); otherwise `None` |

`prof.step()` marks a step boundary. `jt.profiling.record_function(name)` marks a
host interval in the trace.

## Memory over a whole run: snapshots and allocator statistics

`jt.profile` answers one region. To see how memory grows and fragments over a
whole training run, use PyTorch's tools; the calls and what they produce are
those of PyTorch 2.11:

```python
jt.profiling.record_memory_history(max_entries=100000)   # torch.cuda.memory._record_memory_history
train()
jt.profiling.dump_memory_snapshot("snap.pickle")          # torch.cuda.memory._dump_snapshot
jt.profiling.record_memory_history(None)
print(jt.profiling.memory_summary())                      # torch.cuda.memory_summary's table
```

In Torch compatibility mode `torch.cuda.memory._record_memory_history/_snapshot/_dump_snapshot`,
`torch.cuda.memory_snapshot`, `memory_stats`, `memory_summary` and
`reset_accumulated_memory_stats` are these functions.

- **Snapshot**: the pickle has PyTorch's keys, actions (`segment_alloc`/`alloc`/
  `free_requested`/`free_completed`/`segment_free`/`oom`/`snapshot`) and nesting, so
  it opens in <https://pytorch.org/memory_viz> and reads with `torch.cuda._memory_viz`
  (`segsum`, `trace_plot`, `segment_plot`). The records come from the SFRL pools
  themselves (`src/runtime/profiler/memory_history.{h,cc}`): a segment is what a pool
  got from the driver, a block what was split out of it, and addresses are real
  device addresses. `enabled="state"` keeps only the live blocks, `"all"` the whole
  trace too; `context` chooses whether stacks are kept; `max_entries` keeps the newest
  entries per device; stopping discards the records, as in PyTorch.
  `jt.profiling.record_function` ranges go to `external_annotations`.
- **Stacks**: under lazy execution a block is allocated when the graph runs, and the
  Python stack at that moment is the line that forced execution. A tensor's block
  therefore carries the full Python stack **that built its operator** (innermost
  frame first); a library workspace is charged to the operator running; a free, and
  an allocation outside the executor, carry the stack of the moment (when the thread
  holds the GIL). Frames are Python only, no C++ (`stacks="all"` equals `"python"`).
- **Statistics**: `memory_stats()` has every key of `torch.cuda.memory_stats` --
  allocation, segment, active and inactive-split counts and bytes, reserved and
  requested bytes, each for the all/small/large pools as current/peak/allocated/
  freed, and `num_alloc_retries`, `num_ooms`, `num_device_alloc/free`. They are
  counted on every pool allocation, free and segment change
  (`sfrl_device_memory_stats`); `memory_summary()` matches PyTorch's layout
  character for character (titled Jittor in the native API, PyTorch in Torch mode).

Numbers differ from PyTorch's where the allocator does, not the recording: small
segments are 1 MiB (PyTorch: 2 MiB), blocks are 512-byte aligned and belong to
their segment's pool; frees do not wait behind a stream, so `free_requested` and
`free_completed` coincide; with no `max_split_size` nothing is oversize; cuBLAS
workspaces are not allocated from the pools and do not appear in a snapshot.

## Graph replay

Once `jt.graph_replay` has recorded a device graph, a step is a single
`graph_launch` and the executor does not run — so the old per-operator profile is
empty. Now `graph_launch` itself is a record (`cuda_graph_launch`), and the kernels
inside the graph are attributed to it through correlation; `GraphReplay` instances
register automatically, and the report gives their in-region `stats` deltas and
refusal reasons. Other capture/replay mechanisms can enter the report by calling
`jittor.profiling.register_replay_source(obj)` (the object needs a `stats` dict and
optionally `refused`/`_graph_refused`).

## Account for profiling overhead

CUPTI recording, allocation events and Python sampling add work. Host-bound models can be especially sensitive: profiling overhead contributes to the residual “Python + graph construction” category. For host-only analysis, use `device=False, memory=False`; use multiple steps to amortize first-time CUPTI setup. Do not report profiled latency as the uninstrumented benchmark.


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

## Maintenance reference

- Status: maintained
- Corresponding commit: `wip/profiler-rework` branch (baseline `cb93ba04`)
- Owner: Jittor core maintainers
- Review trigger: executor batch/operator launch paths, allocator (SFRL/Temp)
  accounting, graph replay, or CUPTI version changes
