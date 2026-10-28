# Memory Optimization

First use a memory profiler to find the large live tensors, then consider enabling swapping.
**Swapping can lower the memory floor, but it cannot replace fixing a graph that is being
retained by accident.**

## Profile large allocations

Prefer `jt.profile`: it replays the allocator's own event log and gives the exact peak of a
step, the tensors alive at that peak (grouped by producing operator, Python call site, and
shape), workspaces, pool caching and fragmentation, and the process usage seen by NVML --
without needing `trace_py_var`:

```python
with jt.profile() as prof:
    train_step()
print(prof.memory.summary())
```

See [Performance and memory profiling](../notes/profiling.en.md) for details. The older
memory profiler also attributes peak allocations to Python call sites, but requires
`trace_py_var=3` (significantly slower) and only sees tensors created after it is enabled:

```python
import jittor as jt
from jittor import models

net = models.resnet18()
with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
    images = jt.randn((1, 3, 224, 224))
    net(images).sync()
    jt.get_max_memory_treemap()
```

Look at the largest branches in the report. Before changing the global memory limit, do these
three things first: shrink temporary tensors, release the Python references that keep graphs
alive, and split the work into smaller batches.

## Try this switch first: turn off early flushing

The default pipelined execution on CUDA (`auto_flush_ops=128`) splits the graph into
segments, and intermediates that live across segments cannot be released, so the peak is
higher than "launch the whole graph at once" -- measured on a ResNet-50 training step, 5.104
GiB versus 3.574 GiB, at the cost of being 32% slower. **When you hit out-of-memory,
`auto_flush_ops=0` is the first switch to try**, and the direction is counter-intuitive:
making it smaller (rather than setting it to 0) only costs more memory. For the data,
mechanism, and that segmentation table see
[pipelined lazy execution](../notes/pipelined-execution.en.md).

```python
jt.flags.auto_flush_ops = 0
```

Half precision is another orthogonal path: it reduces storage for both activations and
parameters. For how to enable it, loss scaling, and numerical boundaries see
[mixed-precision training](../notes/mixed-precision.en.md).

## Automatic swapping (experimental)

When a limit is reached, Jittor can move data between device memory, host memory, and disk.
Limits are in bytes; `-1` means unlimited.

`JT_SAVE_MEM` is a **build switch, not a runtime flag**: it becomes a compile-time constant so
that builds without it do not pay the cost of a swap branch on every value release. Setting
it therefore selects another build configuration, and the first run after setting or clearing
it recompiles.

**Swapping is not complete yet**: aliased values (`share_with`), cross-device migration, the
dual allocator, and external allocators are still on its own TODO list. Enable it only when
you have **measured that you need it**.

```bash
export JT_SAVE_MEM=1
export cpu_mem_limit=16000000000
export device_mem_limit=8000000000
python train.py
```

PowerShell uses the corresponding `$env:JT_SAVE_MEM`, `$env:cpu_mem_limit`,
`$env:device_mem_limit`. Put swap files on a local disk with enough space, and expect
throughput to drop while swapping.

Clean up leftover swap files:

```bash
python -m jittor_utils.clean_cache swap
```

## Related

- Where memory actually goes and why the caching allocator does not return it to the driver:
  see the [debugging guide](debugging.en.md)
- Moving large tensors back to the host: `.cpu()` allocates its target buffer on the host
  side, see [Devices and placement](../notes/device-placement.en.md)
- Peak and allocation attribution for one step, and which tensors are alive at the peak: see
  [Performance and memory profiling](../notes/profiling.en.md)
- Memory and numerical cost of half-precision training: see
  [mixed-precision training](../notes/mixed-precision.en.md)
