---
name: jittor-single-kernel-cold-profile
description: 诊断单个 kernel 的编译、加载与初始化开销，区分合成模板测量和真实 Jittor JIT。
---

# 拆分单核冷编译成本

## 工具与限制

[scripts/profile_single_kernel.py](scripts/profile_single_kernel.py) 可拆分固定 C++ 模板的
写入、编译、加载和调用耗时，支持 `--compiler`、`--json`、`--work-dir`；先查看 `--help`。
结果字段为 `phases_seconds`。工作目录和 JSON 写入 lab。

该模板写文件阶段不是 Jittor 实际 codegen 成本。若要推断真实 JIT，另外记录真实输入、
生成源码、编译命令、配置指纹和实际加载产物，并用 Jittor 的实际路径交叉验证。

比较前分开进程启动、导入、首次编译、动态加载、首次调用和热调用；固定编译器、优化等级
及并行设置。每轮独立输出，保留失败的编译诊断；不将预热后的数据标成 cold。

性能结论注明样本和波动。模板变快不能直接推出模型训练变快；真实工作负载入口见
[性能分析](../jittor-performance-profiling/SKILL.md)。
