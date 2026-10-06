# Jittor Agent 文档索引

`agent/` 只保存操作手册与可复用 skill，仓库维护命令统一在 `tools/`。
设计与机制说明位于 `docs/development/`、`docs/notes/` 和 `docs/compatibility/`，验证结论位于
`docs/results/`；任务状态、交接和验证证据记录在 GitHub issue 与 PR 中。

## 开始工作

1. 阅读[协作手册](collaboration.md)。
2. 通过 [jittor-dev-context skill](../skills/jittor-dev-context/SKILL.md) 阅读
   [项目上下文](project-context.md)。
3. 按任务读取[环境规则](environment.md)、
   [已知问题总账](known-issues.md)或对应的 `docs/` 文档。
4. 在[结果索引](../../docs/results/index.md)中查找已有验证和性能结论，
   在[源码架构](../../docs/development/source-architecture.md)与[机制说明](../../docs/notes/index.md)中
   查找某个机制为什么是现在这样。
5. 领取工作前查看对应的 GitHub issue/PR；等硬件验收的事项见
   [硬件延迟清单](deferred-hardware.md)，缺陷与限制只记入[已知问题总账](known-issues.md)。
6. 需要对拍或专项基准时，优先复用 `agent/skills/` 中已有工具。

## 目录

```text
agent/
├── manuals/                  # 协作、环境、问题总账和上下文索引
│   ├── agent-index.md         # 本入口
│   ├── collaboration.md
│   ├── deferred-hardware.md   # 等硬件验收的任务与上机命令
│   ├── environment.md
│   ├── known-issues.md
│   ├── project-context.md
└── skills/                   # SKILL.md 与可复用工具

docs/
├── development/              # 仓库布局、源码架构、测试体系与后端契约
├── notes/                    # 长期机制说明
├── compatibility/            # Torch 兼容原则与说明
├── research/                 # 研究提案
└── results/                  # 带基线与复查条件的验证结论
```

原始日志、缓存、生成文件和大体积 benchmark 数据不放入主仓库，统一放在
`$JITTOR_LAB_ROOT`。未设置时使用仓库同级的 `jittor-lab/`。报告可以记录这些
本地产物的相对位置与哈希，但必须明确它们不是版本化文档。详细边界见
[协作手册](collaboration.md#工作区边界)和
[环境规则](environment.md)。
