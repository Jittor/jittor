# 性能

[文档首页](../index.md) · [English](index.en.md)

按顺序完成四件事：建立可比较的基线、定位瓶颈、修改一个因素、重新测量。正确性检查放在计时区间之外，记录确切代码、设备、精度和输入。

| 要做什么 | 阅读 |
| --- | --- |
| 选取相同工作负载，对比两个框架 | [性能对比](comparison.md) |
| 测量单个操作，或保存跨提交的对比 | [基准测试](benchmarking.md) |
| 找到一步的时间与显存开销来源 | [性能分析](profiling.md) |
| 确认大块分配后降低峰值显存 | [显存优化](memory-optimization.md) |

性能分析工具本身有开销。改动后，关闭诊断工具重新测量基线。编译缓存与单测分开，原始数据保存在源码树之外。结论写入相关 GitHub issue 或 PR；库专属测速和支持范围由 [JittorEcosystem](https://github.com/JittorEcosystem) 维护。

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

comparison
benchmarking
profiling
memory-optimization
```
-->