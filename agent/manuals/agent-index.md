# Jittor Agent 文档索引

`agent/` 只保存操作手册与可复用 skill，仓库维护命令统一在 `tools/`。
设计、看板及验证结论的唯一权威分别位于 `docs/architecture/` 与 `docs/results/`。

## 开始工作

1. 阅读[协作手册](collaboration.md)。
2. 通过 [jittor-dev-context skill](../skills/jittor-dev-context/SKILL.md) 阅读
   [项目上下文](project-context.md)。
3. 按任务读取[环境规则](environment.md)、
   [已知问题总账](known-issues.md)或对应的 `docs/` 文档。
4. 在[结果索引](../../refactor-wip/results/README.md)中查找已有验证和性能结论，
   在[设计文档](../../refactor-wip/architecture/README.md)中查找某个机制为什么是现在这样。
5. 领取工作前读取[看板](../../refactor-wip/architecture/refactor-board.md)、
   [交接](../../refactor-wip/architecture/refactor-handoff.md)和
   [分工](../../refactor-wip/architecture/refactor-dispatch.md)，只更新这三处权威文件。
6. 需要对拍或专项基准时，优先复用 `agent/skills/` 中已有工具。

## 目录

```text
agent/
├── manuals/                  # 协作、环境、问题总账和上下文索引
│   ├── agent-index.md         # 本入口
│   ├── collaboration.md
│   ├── environment.md
│   ├── known-issues.md
│   ├── project-context.md
└── skills/                   # SKILL.md 与可复用工具

docs/
├── architecture/             # 设计、整改计划与唯一看板
└── results/                  # 带基线的历史验证结论与只读 baselines
```

原始日志、缓存、生成文件和大体积 benchmark 数据不放入主仓库，统一放在
`$JITTOR_LAB_ROOT`。未设置时使用仓库同级的 `jittor-lab/`。报告可以记录这些
本地产物的相对位置与哈希，但必须明确它们不是版本化文档。详细边界见
[协作手册](collaboration.md#工作区边界)和
[环境规则](environment.md)。
