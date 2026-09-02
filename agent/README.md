# Jittor Agent 文档索引

`agent/` 保存 Jittor 内部协作手册、可复用 skill 和人工整理的验证结论。
长期有效的架构、测试和开发文档统一放在根目录 `docs/`。

## 开始工作

1. 阅读[协作手册](manuals/collaboration.md)。
2. 通过 [jittor-dev-context skill](skills/jittor-dev-context/SKILL.md) 阅读
   [项目上下文](manuals/project-context.md)。
3. 按任务读取[环境规则](manuals/environment.md)、
   [已知问题总账](manuals/known-issues.md)或对应的 `docs/` 文档。
4. 在[结果索引](results/README.md)中查找已有验证和性能结论，
   在[设计文档](design/README.md)中查找某个机制为什么是现在这样。
5. 需要对拍或专项基准时，优先复用 `skills/` 中已有工具。

## 目录

```text
agent/
├── README.md                 # 总入口
├── design/                   # 设计与分析文档（成因、机制、待决策方案）
├── manuals/                  # 协作、环境、问题总账和上下文索引
│   ├── collaboration.md
│   ├── environment.md
│   ├── known-issues.md
│   ├── project-context.md
├── results/                  # 人工整理的验证/性能报告
│   └── transformers/         # Transformers 专项报告
├── scripts/                  # 仓库维护检查
└── skills/                   # SKILL.md 与可复用工具
```

原始日志、缓存、生成文件和大体积 benchmark 数据不放入主仓库，统一放在
`$JITTOR_LAB_ROOT`。未设置时使用仓库同级的 `jittor-lab/`。报告可以记录这些
本地产物的相对位置与哈希，但必须明确它们不是版本化文档。详细边界见
[协作手册](manuals/collaboration.md#工作区边界)和
[环境规则](manuals/environment.md)。
