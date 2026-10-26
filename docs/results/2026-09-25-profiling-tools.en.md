# Performance/memory profiling tools: audit and rewrite

- Status: implemented and verified on a real CUDA device; not merged into `2.0-refactor`
- Date: 2026-09-25
- Baseline commit: `cb93ba04` (branch `wip/profiler-rework`)
- Verification scope: RTX 4090 24 GB (driver 580.178.04), nvcc 12.2.140, CUPTI 12.2.142
  (the `nvidia-cuda-cupti-cu12` wheel, pointed to via `JITTOR_CUPTI_LIB`; the jittor environment does not ship CUPTI by default)
- Maintainer: Jittor core maintainer
- Review conditions: the executor launch path, SFRL/Temp allocator accounting, graph replay, a CUPTI version change
- Mechanism description: [Performance and memory profiling](../notes/profiling.en.md)

## Conclusion

The questions the recent performance work needed to answer — how a step's host time splits among Python, planning, launch, and waiting on the device;
which operators and kernels the device time goes to; whether the bottleneck is host or device; how much the memory peak is, who was alive at the peak, and which line
of code created it; whether a replayed step is really replaying — none of the existing tools could answer fully, and one had to rely on `perf`, `nsys`, and ad-hoc
scripts. Among the existing tools **the counters are correct** (the pool peak `device_memory_peak` and `max_memory_allocated`
have been fixed), but **the profiler is not**: it synchronizes per operator, cannot see replay, and the Torch-side `profiler` is an empty shell.

A single new entry point, `jt.profile`, was added; Torch's `torch.profiler`/`torch.autograd.profiler` were turned into
facades over it; the old tools are kept, with the misleading parts fixed and their positioning corrected in the documentation.

## Audit

For each item a minimal reproduction was made first (the scripts are not committed; they live in `$JITTOR_LAB_ROOT/_state/profiler-rework/repro/`).

| Tool | Claim | Measured | Conclusion |
| --- | --- | --- | --- |
| `jt.profile_scope` / `jt.profiler` (`Profiler::record_and_run`) | Per-operator time within the region | Synchronizes the device after each operator (re-runnable). MLP training step (1024→4096→4096→1024, batch 256, AdamW) steady state 2.289 ms, 3.53–3.76 ms with it on; the total time of 2.71 ms in the table is the sum of "launch + kernel + sync", which is neither device time nor host time; the table header has no host/device columns, row names are raw JIT keys, and there are no Python rows (unless `trace_py_var`) | Kept, repositioned as an operator microbenchmark; whole-step profiling is replaced by `jt.profile` |
| Same, for a graph-replayed step | — | After `jt.graph_replay` captures a CUDA graph, a step shows 0 rows and emits a wrong explanation pointing at `auto_flush_ops` (KI-EXEC-002) | Fixed: when a device graph launch is detected in the region, it now explains that replay bypasses the executor and points to `jt.profile` |
| `profile_memory_enable` + `get_max_memory_info` / `get_max_memory_treemap` / `get_peak_device_used_memory` | Peak value plus the Vars and call sites at the peak | The peak number is correct (4×(2048→8192→2048) MLP, batch 4096: 3456.7 MiB, consistent with `device_memory_peak`). But **re-enabling carries over the previous session's peak and Var list**: enable once, then enable again with `trace_py_var=3`, and the treemap shows the first run's stack-less Vars; call sites require `trace_py_var=3` (that step 68.0 → 84.6 ms), tensors created before enabling have no call site; cuDNN/cub workspaces are not included; the text is separated by `[!@#div1!@#]` | Fixed: clear on enable (`setter_profile_memory_enable`); kept, `jt.profile(memory=True)` is recommended |
| `jt.core.device_memory_used/reserved/peak`, `reset_device_memory_peak`, `device_memory_allocated_total` | Counterparts of torch's `memory_allocated/reserved/max_memory_allocated` | Correct: the peak is recorded on every pool allocation (3456.7 MiB in the example above; NVML 1 ms sampling gives 4112 MiB, the difference being context and caches) | Kept; added `device_memory_reserved_peak`, and `reset_device_memory_peak` resets both peaks |
| `torch.cuda.max_memory_allocated` ("a 22 GB step reports only 0.1–1.3 GB") | — | Fixed (`device_memory_peak`): in the reproduction a step after reset is 140.1 MiB, consistent with the `memory_allocated`/`memory_reserved` readings | Kept |
| `torch.cuda.max_memory_reserved` / `max_memory_cached` | Peak reserved amount | Returns the **allocation** peak: 140.1 MiB, less than `memory_reserved()` 163.0 MiB at the same moment | Fixed: reads the pool's reserved peak |
| `torch.cuda.memory_stats` / `memory_summary` | — | `memory_stats` has only 3 keys; `memory_summary` does not exist | Added: reserved peak, cumulative allocation; added `memory_summary` |
| `torch.profiler.profile` / `record_function` / `schedule` / `tensorboard_trace_handler` | PyTorch profiling API | Empty shell: `profile` records nothing, has no `key_averages`, and `export_chrome_trace` writes no file and reports no error | Replaced with a `jt.profile` facade |
| `torch.autograd.profiler.profile` / `EventList` | Same as above | Empty shell: `key_averages()`/`function_events` return empty tables, `table()` returns an empty string | Replaced |
| `display_memory_info`, `get_mem_info`, `liveness_info`, `jt.introspection.counters` | Debug dumps/counters | Work as documented (`total_cuda_used` sums over all cards, already documented) | Kept, unchanged |
| `graph_build_profile_*` | Time of the C++ graph-construction phase itself | Requires rebuilding the core with `JT_GRAPH_BUILD_PROFILE=1`; correctly reports empty when off | Kept (developer tool); not re-checked this round with a probe build |
| `trace_py_var>=2` + `dump_trace_data` / `print_trace` | Tracing | The executor synchronizes after each operator (`exec_runner.cc`), so it cannot be used for timing | Kept, noted in the documentation |
| `profile_mark` | Sections for `profile_scope` | Not re-checked | Kept |
| The NVML sampler in `benchmarks/torch_compat/worker.py` | Cross-framework process memory peak | Samples at 20 ms intervals and may miss short peaks; still reasonable as one ruler for both sides | Kept |

## What changed

- **Executor step tracing** (`src/runtime/profiler/step_trace.{h,cc}`): batches (planning/compile/finish),
  device waits (`DeviceWaitScope`), every launched operator (start/allocation done/end, readable name, input shapes,
  output bytes, the `Op::launch_origin` call site), device graph launches; during launch the operator sequence number is narrowed into a CUPTI
  external correlation id. When disabled, each hook is a single branch. Hook points: `executor.cc`,
  `exec_runner.cc`, `executor_entry.cc`, `graph_capture.cc`, `var.cc`.
- **Allocator events and reserved peak**: `sfrl_allocator.cc` (allocate/free/request from and return to the driver, plus new
  per-device reserved amounts and their peaks), `temp_allocator.cc` (workspaces), `mem_info.*`
  (`device_memory_reserved_peak`); `launch_diagnostics` gains a call-site id → file line lookup.
- **`jittor.profiling`** (`jt.profile`): CUPTI (ctypes, loaded on demand), host breakdown, device breakdown,
  bottleneck determination, operator/kernel/call-site tables, memory log replay (`MemoryReport`), a Python sampler, NVML,
  Chrome trace export. `GraphReplay` is registered automatically as a replay source.
- **Torch facade**: `compat/torch/profiler.py` (new); `installers/utilities.py`,
  `autograd.py`, `installers/autograd.py` changed to publish it; `installers/cuda/{api,bindings}.py`
  fix `max_memory_reserved`, extend `memory_stats`, add `memory_summary`.
- **Old tools**: `profile_memory_enable` clears on enable; the empty-report warning of `profile_scope` distinguishes device graph
  launches; the documentation and notebook are corrected (the original "the peak has no reset interface" no longer holds).

## Evidence

### Native ResNet18 training (batch 64, 224², SGD momentum), 3 steps

Steady state 30.04 ms/step. `jt.profile(python=True)` over three steps 91.24 ms:

- Host: Python+graph construction 11.05 ms (12.1%), planning 1.37, compile/lookup 3.07, launch 14.38
  (1401 operators, 10.3 µs each, of which allocation 2.6 µs), waiting on the device 61.37 ms (67.3%);
- Device: busy 88.48 ms (97.0%), all 1308 kernels attributed to operators → **device-bound**;
- Top device times: `cudnn_conv_backward_w` 22.79 ms (25.7%), BatchNorm's `code` operator
  21.86 ms (24.6%, with kernels `batch_norm_backward` 10.35 ms and `batch_norm_forward`
  6.47 ms), `cudnn_conv_backward_x` 14.84 ms, `cudnn_conv` 14.37 ms;
- Memory: Var peak 1645.0 MiB (consistent with the pool counter), **including cuDNN workspaces 1929.0 MiB** (none of the previous
  counters could see these 284 MiB), pool reserved peak 2080.0 MiB, NVML 2816 MiB (423 MiB outside the pool);
  alive at the peak: the `cudnn_conv` output 606.4 MiB, the `ternary` operator output 514.5 MiB,
  pre-existing (parameters/momentum) 170.5 MiB; the largest single tensor item is 9 × `float32[64,64,56,56]` totalling
  441.0 MiB; the cache at the end of the region is 1909.4 MiB spread over 122 free blocks, a fragmentation rate of 90%.

### Torch-mode Transformers GPT-2 (4 layers, width 256, batch 8×128, AdamW), 5 steps

Steady state 10.00 ms/step. `jt.profile(python=True, sample_interval=0.0005)` over five steps 63.48 ms:

- **host-bound**: the device is busy only 22.8%; of host time 61.1% is in Python+graph construction, 38% in executor planning/launch,
  0.7% waiting on the device;
- Device time: `cublas_matmul` 6.04 ms (41.7%), `fused_adamw` 2.40 ms (16.6%),
  `cublas_batched_matmul` 1.49 ms;
- Call sites: `transformers/pytorch_utils.py:122` (Conv1D's addmm) created 800 operators, launched
  5.80 ms, device 5.32 ms; `transformers/integrations/sdpa_attention.py:83` 500 operators;
- Memory: peak 220.3 MiB; the largest is the logits `float32[8,128,8192]` at `loss_utils.py:36`,
  32 MiB.

### A replayed step

One MLP inference step captured into a CUDA graph by `jt.graph_replay`: the report shows 1 device graph launch, 6 kernels
(cuBLAS, fused operators) hanging under `cuda_graph_launch`, and a `GraphReplay` counter increment of
`{'replayed': 1, 'graph': 1}`; input/output copies outside the graph are listed separately as "not belonging to any operator". The old tool reports 0 rows in the same
scenario.

### Independent check of device time

With 16 consecutive 2048³ matrix multiplications (device saturated), the per-call `cublas_matmul` device time from CUPTI falls
between 0.5–1.1 times the wall clock per call (regression test `test_device_time_agrees_with_a_device_bound_wall_clock`).
The memory peak is byte-for-byte equal to the pool counter (`test_the_traced_peak_is_the_pools_peak_and_names_what_was_live`).

### Overhead

Same process, same cache, best of three:

| Workload | No profiling | Host only | Default | Default + Python sampling |
| --- | --- | --- | --- | --- |
| MLP training (device-bound, 20 steps) | 2.287 ms | 2.294 | 2.344 (+2.5%) | — |
| Torch-mode GPT-2 (host-bound, 10 steps) | 10.10 ms | 10.31 (+2%) | 11.86 (+17%) | 12.16 (+20%) |

On host-bound workloads the cost is mainly that of recording CUPTI runtime API calls (correlation needs it; driver API
recording is off and does not affect attribution — the attribution rate remains 100% for ResNet18 and the cuDNN case).

## Tests

Failing before the fix and passing after, all executed on an RTX 4090:

- `tests/runtime/test_step_profile.py` (12 items): the host breakdown sums to the wall clock, call sites, device-time
  attribution and validation, Chrome trace, memory peak and live set, reserved peak and reset, old memory profiler clearing on enable,
  replay visibility, the `profile_scope` replay warning, CPU regions. Before the fix: the module failed to import; with
  the import removed, three items failed — old profiler clearing, reserved peak, and the replay warning. The three device-time items are skipped with
  "CUPTI ... cannot be loaded" when CUPTI is not found.
- `compat/tests/torch/test_torch_profiler_facade.py` (4 items): `key_averages`/`table`/
  Chrome trace, `schedule` saves only once, `torch.autograd.profiler`, `max_memory_reserved`
  and `memory_summary`. All 4 items failed before the fix.

Gate results are in the commit message.

## Not done / not verified

- The jittor environment has no CUPTI by default; this verification used a separately downloaded `nvidia-cuda-cupti-cu12==12.2.142`.
  It was not added to the dependencies, and the CUPTI record layout was not verified on other CUDA versions (only fields
  stable since CUPTI 11 are read).
- ROCm, Ascend: the host breakdown and memory follow the same executor path, but they were not run on real hardware; device time is not implemented.
- For native `jittor.models` models the call site lands on the user's line that calls the model (`launch_origin` skips all
  `jittor.*` frames); per-module attribution would need another set of rules.
- `StepCapture` outside the main process (uncommitted in the main checkout) is not wired in; it only needs to call
  `jittor.profiling.register_replay_source(self)` to appear in the report.
- The host breakdown when multiple Python threads drive the executor concurrently is not verified.
