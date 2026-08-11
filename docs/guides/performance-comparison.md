# Comparing framework performance

Jittor executes asynchronously and compiles operators on first use. A useful
benchmark therefore separates compilation from steady-state execution and
synchronizes the device around the measured interval.

For repeatable, commit-indexed measurements, prefer the maintained
{doc}`../performance/benchmarking` ASV suite. The following pattern is useful
for a local model comparison:

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

Use the same warm-up count, input shape, dtype, evaluation mode, accelerator,
library versions, and synchronization policy for every framework. Do not
compare Jittor's first compile against another framework's warm cache.

## Validate numerical parity

A faster result is not useful if it computes a different model. Before timing:

1. Transfer identical parameters and input values.
2. Compare outputs outside the timed region.
3. Choose tolerances appropriate for the dtype and backend.
4. Confirm that both processes use the intended accelerator.
5. Monitor device memory so swapping or allocation pressure does not skew one
   side of the comparison.

For PyTorch, call `torch.cuda.synchronize()` before and after the measured loop.
For Jittor, call `jt.sync_all(True)` at the same boundaries. Report the hardware,
software versions, exact command, sample count, and dispersion rather than a
single unexplained throughput number.
