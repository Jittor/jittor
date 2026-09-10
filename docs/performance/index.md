# 性能

```{toctree}
:maxdepth: 1

benchmarking
library-standing
```

性能结论只有在**可复现**时才有意义：隔离编译缓存、显式同步、记录确切提交。
[基准测试](benchmarking.md) 说明这套口径，[下游库性能台账](library-standing.md)
记录每个库当前测到哪儿了——两者分开，是因为方法学基本不变而数字每次测量都会变。

动手优化见 [性能对比](../guides/performance-comparison.md) 与
[显存优化](../guides/memory-optimization.md)；想理解为什么快或为什么慢，
见 [机制说明](../notes/index.md)。
