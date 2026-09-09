# 整改期临时文档（完成后整个目录删除）

本目录存放 `2.0-refactor` 整改期间的**过程文档**：任务看板、计划、交接、分工、
审计快照，以及迁移期的 API owner 对照表。

它们是脚手架，不是 Jittor 的对外文档。整改收口后**整个 `refactor-wip/` 目录直接
删除**，届时不需要把其中任何内容迁回 `docs/`。

对外文档只在 [`docs/`](../docs/) 一棵树里：安装与上手、API 参考、指南、教程、
兼容性说明、发布说明和贡献指南。长期有效的设计说明留在
[`docs/architecture/`](../docs/architecture/)，测试契约留在
[`docs/testing/`](../docs/testing/)。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`architecture/refactor-board.md`](architecture/refactor-board.md) | 整改任务的唯一状态源 |
| [`architecture/refactor-plan.md`](architecture/refactor-plan.md) | 任务清单与编号 |
| [`architecture/refactor-handoff.md`](architecture/refactor-handoff.md) | 当前交接与证据 |
| [`architecture/refactor-dispatch.md`](architecture/refactor-dispatch.md) | 分工方式 |
| [`architecture/target-layout.md`](architecture/target-layout.md) | 目标目录布局 |
| [`architecture/codebase-audit/`](architecture/codebase-audit/) | 分域审计快照 |
| [`architecture/system-design-audit.md`](architecture/system-design-audit.md) | 系统设计审计 |
| `architecture/*-owners.md` | 迁移期 API owner 对照表 |

## 未迁入的部分

`docs/results/` 的历史验证报告仍在原处。它属于同一类过程材料，但迁移时有并发写入，
留待整改收口时与本目录一并处理。
