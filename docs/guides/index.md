# 使用指南

[文档首页](../index.md) · [English](index.en.md)

按要完成的任务选择指南。首次使用某个设备时，先确认平台覆盖范围；安装成功本身不能证明数值或训练能力。

## 硬件与运行条件

先查平台支持与已知限制，再配置具体设备。Corex 页面说明探测方法，不代表已通过硬件验收。

- [平台支持](hardware/platform-support.md)
- [已知限制](known-limitations.md)
- [昇腾 910B：配置与验证](hardware/ascend-910b.md)
- [昇腾 310P：配置、能力边界与验证](hardware/ascend-310p.md)
- [天数 Corex：探测与后续验证](hardware/corex.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

hardware/platform-support
known-limitations
hardware/ascend-910b
hardware/ascend-310p
hardware/corex
```
-->

## 分布式训练

新的原生训练脚本从 `jtrun` 单机多卡开始。MPI 指南面向既有 MPI 程序；模型与优化器分片的边界见机制说明。

- [原生单机多卡训练](distributed/distributed-training.md)
- [用 MPI 做分布式训练](distributed/distributed-mpi.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

distributed/distributed-training
distributed/distributed-mpi
```
-->

## 调试与嵌入

报错或数值异常先看调试指南；需要在 C++ 程序中嵌入 Python 运行时时，再看控制台接口。

- [调试 Jittor 程序](debugging.md)
- [C++ 控制台接口](cpp-console.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

debugging
cpp-console
```
-->

## 测量与优化

按[性能工作流](../performance/index.md)建立基线、定位瓶颈，再调整显存或执行设置。

## 分布式状态

先确定要分片的对象：ZeRO 分片优化器状态，张量并行分片线性层特征，FSDP 分片参数，checkpoint 分片文件。各项有独立的后端和组合限制；文件分片不等于训练分片。

- [原生 ZeRO 第一阶段优化器](distributed/native-zero-optimizer.md)
- [原生张量并行线性层](distributed/native-tensor-parallel.md)
- [原生 FSDP 的存储、反向与 checkpoint 归属](distributed/native-fsdp.md)
- [原生分片 checkpoint 文件](distributed/native-sharded-checkpoint.md)

<!-- sphinx-only
```{toctree}
:maxdepth: 1
:hidden:

distributed/native-zero-optimizer
distributed/native-tensor-parallel
distributed/native-fsdp
distributed/native-sharded-checkpoint
```
-->

## 混合精度训练

从[混合精度](mixed-precision.md)了解 autocast 和梯度缩放，再核对设备与算子限制。

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

mixed-precision
```
-->
