# 使用指南

面向具体任务的操作指南：在特定硬件上运行、调试问题、优化显存与速度。

```{toctree}
:maxdepth: 1

debugging
memory-optimization
performance-comparison
distributed-mpi
ascend-910b
corex
cpp-console
```

| 指南 | 适用场景 |
| --- | --- |
| [调试](debugging.md) | 报错定位、梯度异常、编译失败 |
| [显存优化](memory-optimization.md) | 显存不足、想跑更大的批次 |
| [性能对比](performance-comparison.md) | 与其他框架做可复现的速度对比 |
| [MPI 分布式](distributed-mpi.md) | 多进程、多卡训练 |
| [昇腾 910B](ascend-910b.md) | 在华为昇腾 NPU 上运行 |
| [天数 Corex](corex.md) | 在天数智芯 GPU 上运行 |
| [C++ 控制台](cpp-console.md) | 直接调用 C++ 接口、嵌入部署 |
