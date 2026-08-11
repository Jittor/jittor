---
name: jittor-dev-context
description: Jittor 开发任务的上下文路由入口。用于定位当前状态、架构契约、环境规则、活跃问题和验证纪律，适用于修复、重构、对拍、后端复验和模型覆盖。
---

# Jittor → Torch-Grade 项目上下文

**用途**：开始在 Jittor 仓库工作前，用本 skill 定位所需上下文，避免读取或复制一份
不断增长的会话历史。

## 必读

开工前**必须阅读** [`../../manuals/project-context.md`](../../manuals/project-context.md)。
它是短索引，不是完整历史。再按任务选择：

- 环境、缓存和后端前置：[`../../manuals/environment.md`](../../manuals/environment.md)
- 活跃缺陷与 workaround：[`../../manuals/known-issues.md`](../../manuals/known-issues.md)
- 架构与模块边界：[`../../../docs/architecture/source-architecture.md`](../../../docs/architecture/source-architecture.md)
- Torch 兼容验收：[`../../../docs/architecture/torch-compatibility-principles.md`](../../../docs/architecture/torch-compatibility-principles.md)
- 测试体系：[`../../../docs/testing/test-system.md`](../../../docs/testing/test-system.md)
- 已有验证报告：[`../../results/README.md`](../../results/README.md)

## 工作纪律速记

1. **宁可响亮崩也不静默错**
2. **verify-then-fix**（~75% 审计是误报，先复现再修）
3. 改动必须验证所有声明支持的后端；无目标硬件时明确报告未验证
4. 对拍/调试工具沉淀到 `agent/skills/`
5. 提交信息结尾可选加 `Co-Authored-By: ...`（非必须）
6. 只在用户要求时 push
7. **效率优先**：先跑最小复现和定向测试，再逐层扩大门禁
8. **计算跑在 device 上**：torch_compat 里新加的计算要考虑效率，至少能跑在 device（GPU/NPU）上，不能只在 CPU
9. 当前状态入口变化才更新 `project-context.md`；活跃缺陷更新 `known-issues.md`；
   详细证据写入 `agent/results/`
10. **多用 subagent 提升效率**（并行验证、分卡并行等）

## 快速开工

1. 读 `agent/manuals/project-context.md` 并打开任务相关链接
2. 确认分支、提交和 dirty state：`git status --short --branch`
3. 按 `agent/manuals/environment.md` 隔离缓存和运行状态
4. 搜索 `agent/manuals/known-issues.md` 与 `agent/results/` 中的既有证据
5. 最小复现后再修改，按变更风险逐层扩大测试
