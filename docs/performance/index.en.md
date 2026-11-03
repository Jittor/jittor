# Performance

[Documentation](../index.en.md) · [中文](index.md)

Work in this order: establish a comparable baseline, identify the bottleneck, make one change, and measure again. Keep correctness checks outside the timing window and record the exact code, device, precision and inputs.

| Task | Read |
| --- | --- |
| Choose matching workloads and compare frameworks | [Performance comparison](comparison.en.md) |
| Time an operation or retain a comparison across commits | [Benchmarking](benchmarking.en.md) |
| Find where a step spends time and memory | [Profiling](profiling.en.md) |
| Reduce peak memory after identifying its source | [Memory optimization](memory-optimization.en.md) |

Profiling adds measurement overhead. After making a change, rerun the baseline without diagnostic instrumentation. Keep compilation caches separate from tests, and store raw measurements outside the source tree. Record conclusions in the relevant GitHub issue or PR; library-specific benchmarks and support scopes belong to [JittorEcosystem](https://github.com/JittorEcosystem).

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

comparison.en
benchmarking.en
profiling.en
memory-optimization.en
```
-->