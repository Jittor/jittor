# Memory optimization

Use the memory profiler to identify large live tensors before enabling swapping.
Swapping lowers the minimum device-memory requirement, but it cannot replace
fixing an accidentally retained graph.

## Profile large allocations

The memory profiler attributes peak allocations to Python call sites:

```python
import jittor as jt
from jittor import models

net = models.resnet18()
with jt.flag_scope(trace_py_var=3, profile_memory_enable=1):
    images = jt.randn((1, 3, 224, 224))
    net(images).sync()
    jt.get_max_memory_treemap()
```

Inspect the largest branches in the reported tree. Reduce temporary tensor
sizes, release Python references that keep graphs alive, or split work into
smaller batches before changing global memory limits.

## Enable automatic swapping

Jittor can move values between device memory, host memory, and disk when a
limit is reached. Limits are bytes; `-1` means unlimited.

`JT_SAVE_MEM` is a **build** switch, not a runtime flag: it becomes a
compile-time constant so that the swap branch on every value release costs
nothing in a build that did not ask for it. Setting it therefore selects a
separate build configuration, and the first run after setting or clearing it
recompiles.

Swapping is **experimental and unfinished** -- aliased values (`share_with`),
migration between devices, the dual allocator and foreign allocators are all
still open in its own TODO list. Enable it only if you have measured that you
need it.

```bash
export JT_SAVE_MEM=1
export cpu_mem_limit=16000000000
export device_mem_limit=8000000000
python train.py
```

PowerShell uses the corresponding `$env:JT_SAVE_MEM`, `$env:cpu_mem_limit`, and
`$env:device_mem_limit` variables. Keep swap storage on a local disk with
enough free space and expect lower throughput while swapping is active.

Remove stale swap files with:

```bash
python -m jittor_utils.clean_cache swap
```
