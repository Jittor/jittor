# Performance

```{toctree}
:maxdepth: 1

benchmarking.en
library-standing.en
```

A performance conclusion only means something when it is **reproducible**:
isolated compilation caches, explicit synchronisation, and the exact commit
recorded. [Benchmarking](benchmarking.en.md) describes that protocol, and the
[downstream library ledger](library-standing.en.md) records how far each
library has been measured. They are separate because the methodology rarely
changes while the numbers change on every measurement.

For hands-on tuning see [performance comparison](../guides/performance-comparison.en.md)
and [memory optimization](../guides/memory-optimization.en.md). To understand
why something is fast or slow, see [how Jittor works](../notes/index.en.md).
