# 机制说明

这一节解释 Jittor 的运行方式——那些不看文档就会被行为"意外"到的地方：张量到底
在哪张卡上、低精度累加什么时候生效、两个设备为什么给出不同的数、惰性执行为什么让
GPU 闲着。

写给已经能跑通模型、开始关心结果是否正确、快慢是否合理的人。想了解代码怎么组织、
错误如何分级、算子源码怎么写，见[开发文档](../development/index.md)。

```{toctree}
:maxdepth: 1

device-placement
numerics-contract
float32-precision-policy
complex-dtype
pipelined-execution
```

## 按问题索引

| 遇到的现象 | 看这篇 |
| --- | --- |
| 张量在哪张卡上？`.cpu()` / `.cuda()` 怎么用 | [设备与放置](device-placement.md) |
| CPU 和 CUDA 的结果对不上；NaN、无穷、次正规数、归约精度 | [数值契约](numerics-contract.md) |
| TF32、fp32 矩阵乘精度、`float32_matmul_precision` | [float32 累加精度](float32-precision-policy.md) |
| 复数支持到什么程度 | [复数 dtype](complex-dtype.md) |
| 为什么 GPU 利用率上不去、`auto_flush_ops` 是什么 | [流水式惰性执行](pipelined-execution.md) |