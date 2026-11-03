# Jittor 主仓上下文

本页只说明当前维护边界和查找入口。任务进度与验证结果在 GitHub issue/PR；缺陷状态以
[问题总账](known-issues.md) 为准，不在这里复制一份。

## 仓库负责什么

Jittor 以元算子与即时编译执行张量计算。原生 Python API 在 `python/jittor/`，
编译器与运行时 C++ 在 `src/`，设备与通信后端在 `backends/`。

`jittor-compat` 是本仓 `compat/` 维护的独立发行物，安装为 `jittor.compat`，负责通用
Torch API 和 shim。它与核心共享运行时，但不是某个下游库的 adapter。

[独立生态仓库](https://github.com/JittorEcosystem) 负责下游版本组合、注册与适配、官方测试、
教程、测速和功能/芯片支持表。主仓不再保存 `adapters/` 合集，也不为每个下游库维护专属 skill。
由下游暴露的核心/compat 缺陷仍修回主仓，并保留不依赖完整下游安装的最小回归。

## 按任务找入口

| 任务 | 入口 |
| --- | --- |
| 开始修改、同步与提交 | [`AGENTS.md`](../../AGENTS.md)、[协作手册](collaboration.md) |
| 准备环境、选择缓存和后端 | [environment.md](environment.md) |
| 确定源码与资源归属 | [源码架构](../../docs/development/source-architecture.md)、[仓库布局](../../docs/development/repository-layout.md) |
| 选择测试范围 | [测试体系](../../docs/development/testing/test-system.md)、[`tools/run_test_suite.py`](../../tools/run_test_suite.py) |
| 修改通用 Torch 兼容语义 | [兼容原则](../../docs/development/compatibility/principles.md)、[兼容架构](../../docs/development/compatibility/torch-architecture.md) |
| 从下游问题分流到主仓 | [适配分流流程](../skills/downstream-library-adaptation/SKILL.md) |
| 查未解决缺陷 | [known-issues.md](known-issues.md) |
| 补真实设备验收 | [deferred-hardware.md](deferred-hardware.md) |
| 找验证与诊断方法 | [agent-index.md](agent-index.md) |
| 用户文档与升级说明 | [文档首页](../../docs/index.md)、[2.0 升级说明](../../docs/project/releases/2.0.md) |

## 使用证据

当前源码结构与测试入口从目标远端最新提交核对。旧报告只作为定位线索，不表示当前版本已经通过；
需要历史证据时从问题条目、GitHub issue/PR 或 Git 历史进入。
不得用导入成功、CPU fallback、静态检查或全 skip 代替真实设备执行。
