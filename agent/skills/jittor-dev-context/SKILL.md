---
name: jittor-dev-context
description: Jittor 主仓的上下文入口；按任务定位核心、后端、通用 compat、环境和验证规则，并区分 JittorEcosystem 的下游责任。
---

# Jittor 主仓上下文入口

开始或恢复任务时，先遵守根 [`AGENTS.md`](../../../AGENTS.md) 的同步要求，再依次读取：

1. [协作手册](../../manuals/collaboration.md)：主仓与生态仓库的责任边界。
2. [项目上下文](../../manuals/project-context.md)：当前源码、测试和文档入口。
3. 与任务相关的 [环境规则](../../manuals/environment.md)、[问题总账](../../manuals/known-issues.md)、
   [待硬件验收项](../../manuals/deferred-hardware.md) 和公开开发文档。

从 [skill 索引](../../manuals/agent-index.md) 选择已有方法，不要求每个任务通读所有材料。
下游专属安装与验收到对应 JittorEcosystem 仓库查阅；问题需要改动核心或 compat 时，
按 [适配分流流程](../downstream-library-adaptation/SKILL.md) 建立主仓最小复现。

本 skill 不重复同步、提交、测试预算或缓存规则，也不另设与 `AGENTS.md` 冲突的要求。
更新它仅用于修正入口与归属；执行记录写入 GitHub issue/PR。
