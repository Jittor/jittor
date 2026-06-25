---
name: jittor-dev-context
description: Jittor torch-grade 项目的完整背景、进度、环境、bug 总账和待办。任何 agent 在开始工作前必须先读此 skill，了解目标、验收门槛、已完成工作、已知陷阱和工作纪律。适用于：接手 jittor 开发任务、修 bug、做对拍验证、NPU/CUDA 复验、扩展模型覆盖等。
---

# Jittor → Torch-Grade 项目上下文

**用途**：任何 agent 开始在 jittor 仓库工作前，先读本 skill 获取完整上下文。

## 必读

开工前**必须阅读**同目录下的 `CONTEXT.md`，它包含：

- §0 目标与 5 条硬验收门槛 (G1–G5)
- 整体进展与现状（已完成 / 在飞 / 待做）
- §1 环境、机器、命令、验证回路
- §2 21 条任务现状速览
- §3 已完成清单
- §4 Bug 清单 + 内核陷阱（最值钱的交接知识）
- §5 待办
- §6 用户批注（完成标准）
- §7 建议下一步

## 工作纪律速记

1. **宁可响亮崩也不静默错**
2. **verify-then-fix**（~75% 审计是误报，先复现再修）
3. 改动必须 **双卡验证**（CPU + CUDA + NPU）
4. 对拍/调试工具沉淀到 `agent/skills/`
5. 提交信息结尾可选加 `Co-Authored-By: ...`（非必须）
6. 只在用户要求时 push
7. **效率优先**：先在 GPU 上跑通所有流程，再进行 NPU 验证
8. **计算跑在 device 上**：torch_compat 里新加的计算要考虑效率，至少能跑在 device（GPU/NPU）上，不能只在 CPU
9. **有进展及时更新** `CONTEXT.md`，细节可加子文档
10. **多用 subagent 提升效率**（并行验证、分卡并行等）

## 快速开工

1. 读同目录 `CONTEXT.md`（60 秒速读顶部 → 详细按需）
2. 确认分支：`git branch --show-current` → 应为 `2.0`
3. 确认环境：见 CONTEXT.md §1.2 机器与环境表
4. 按 §1.1 主循环开展工作
