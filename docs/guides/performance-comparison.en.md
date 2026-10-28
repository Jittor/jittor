# Comparing Performance with Other Frameworks

Jittor executes asynchronously and compiles operators on first use. A meaningful benchmark
must therefore **separate compilation from steady-state execution** and synchronize the
device before and after the measurement interval.

When you need reproducible measurements with a commit id, prefer the maintained
[ASV benchmark suite](../performance/benchmarking.en.md); when what you want to compare is
**the end-to-end gap between the same `import torch` code on the compatibility layer and on
real PyTorch**, use
[`benchmarks/torch_compat/`](https://github.com/Jittor/jittor/blob/master/benchmarks/torch_compat/README.md)
-- it already handles two interpreters, warmup to steady state, per-step numerical alignment,
and the memory convention. The current gap table is in the
[downstream library performance ledger](../performance/library-standing.en.md). The pattern
below is suitable for local single-model comparisons:

```python
import time

import jittor as jt
from jittor.models import resnet50

jt.flags.use_cuda = int(jt.has_cuda)
warmup = 10
repetitions = 100
batch_size = 8

data = jt.random((batch_size, 3, 224, 224))
model = resnet50()
model.eval()

jt.sync_all(True)
for _ in range(warmup):
    model(data).sync()
jt.sync_all(True)

start = time.perf_counter()
for _ in range(repetitions):
    model(data).sync()
jt.sync_all(True)
elapsed = time.perf_counter() - start

print("Jittor images/s:", repetitions * batch_size / elapsed)
```

Use **the same** warmup count, input shapes, dtype, eval mode, accelerator, library versions,
and synchronization policy for every framework. **Do not compare Jittor's first compilation
against another framework's warm cache.**

## Verify numerical agreement first

Being faster while computing a different model is meaningless. Before timing:

1. pass identical parameters and input values;
2. compare outputs **outside** the timing interval;
3. choose tolerances appropriate for the dtype and backend;
4. confirm both processes use the intended accelerator;
5. watch memory, so swapping or allocation pressure does not slow only one side.

Call `torch.cuda.synchronize()` before and after the measurement loop on the PyTorch side,
and `jt.sync_all(True)` at the same points on the Jittor side. When reporting results, give
the hardware, software versions, exact command, sample count, and dispersion rather than an
unexplained throughput number.

CPU comparisons **must pin threads** (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`): the same
unpinned step can measure two clusters that differ by 25%, enough to invert the conclusion.
For measured numbers see ["CPU thread pinning" in
Benchmarking](../performance/benchmarking.en.md).

## Why the GPU sits idle

If kernels are faster but the whole step is slower, the usual cause is that CPU graph
construction and device computation do not overlap. See
[pipelined lazy execution](../notes/pipelined-execution.en.md), which explains
`auto_flush_ops` and gives measured data for four transformers.
