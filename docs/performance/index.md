# 性能

```{toctree}
:maxdepth: 1

benchmarking
```

性能测量应隔离编译缓存、显式同步，并记录确切提交、设备和对照环境。
[基准测试方法](benchmarking.md) 说明测量口径；每次实测结果写入相关 GitHub issue/PR，
下游库的测速与支持范围由对应的 [JittorEcosystem 仓库](https://github.com/JittorEcosystem) 维护。
历史性能数字不作为当前版本的支持或速度承诺。

动手优化见 [性能对比](../guides/performance-comparison.md) 与
[显存优化](../guides/memory-optimization.md)；机制说明见 [notes](../notes/index.md)。
