# Jittor Agent 文档索引

`agent/` 保存 Jittor 内部协作手册、可复用 skill 和人工整理的验证结论，
不属于公开 Sphinx 用户文档。

## 开始工作

1. 阅读[协作手册](manuals/collaboration.md)。
2. 通过 [jittor-dev-context skill](skills/jittor-dev-context/SKILL.md) 阅读
   [项目上下文](manuals/project-context.md)。
3. 在[结果索引](results/README.md)中查找已有验证和性能结论。
4. 需要对拍或专项基准时，优先复用 `skills/` 中已有工具。

## 目录

```text
agent/
├── README.md                 # 总入口
├── manuals/                  # 协作、上下文和设计手册
│   ├── collaboration.md
│   ├── project-context.md
│   └── design/
├── results/                  # 人工整理的验证/性能报告
│   └── transformers/         # Transformers 专项报告
├── scripts/                  # 仓库维护检查
└── skills/                   # SKILL.md 与可复用工具
```

原始日志、缓存、生成文件和大体积 benchmark 数据不放入主仓库，统一放在
`${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}`。报告可以链接这些本地产物，
但必须明确它们不是版本化文档。仓库与实验工作区的详细边界见
[协作手册](manuals/collaboration.md#工作区边界)。
