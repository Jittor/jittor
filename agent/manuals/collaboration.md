# Jittor Agent 协作手册

本手册说明 AI agent 在 Jittor 仓库内的协作、验证和文档规则。

## 开工流程

1. **先读上下文**：阅读 [`../skills/jittor-dev-context/SKILL.md`](../skills/jittor-dev-context/SKILL.md)，
   再通过 [`project-context.md`](project-context.md) 定位相关架构契约。
2. **确认环境与问题**：按需阅读 [`environment.md`](environment.md) 和
   [`known-issues.md`](known-issues.md)，不要依赖个人机器路径或过期会话记录。
3. **确认结果文档**：已有主题继续更新 `../results/` 中的报告；新主题使用
   `YYYY-MM-DD-topic.md`。
4. **开始工作**，遵循下面的协作规范。

## 协作规范

### 文档更新（核心纪律）

- **`project-context.md`** 是当前状态索引，只在目标、状态或文档入口变化时更新。
- **`known-issues.md`** 是活跃问题总账。新增问题要有 owner、可执行证据、workaround
  和退出条件；问题修复后删除条目，历史由 Git 和结果报告保留。
- **`docs/`** 保存长期架构决策、测试契约、开发指南和研究提案。
- **结果报告**记录验证口径、命令、结果、结论和已知边界。原始日志、缓存、
  二进制和大体积结果放在 `$JITTOR_LAB_ROOT`，
  报告中标明“未版本化”，不要放进 Jittor 主仓库。
- 稳定文档注明状态、复查日期、对应基线、owner 和复查触发条件。

### Skill 沉淀

工作过程中写出的**可复用工具**（对拍脚本、验证 harness、调试探针等），沉淀为 skill：

- 在 `../skills/` 下创建新目录，包含 `SKILL.md`（说明用途和用法）与工具文件。
- 已有 skill 直接复用，别重新造轮子。

### 效率原则

- **按目标后端验证**：先在最容易定位问题的后端跑通，再验证所有声明支持的设备。
- **计算在 device 上**：torch_compat 里新加的计算必须能跑在 GPU/NPU 上，不能只支持 CPU。
- **多用 subagent**：互不依赖的验证任务并行展开（分卡并行、多模型并行、三后端并行）。
- **verify-then-fix**：~75% 的审计是误报，先复现再修。

### JIT 并发规则

Jittor 使用文件锁串行化 JIT 编译。多个进程共享缓存并首次编译时可能长期等待：

1. 新算子或新扩展的首次编译必须串行完成。
2. 并行任务必须使用不同的 `JITTOR_HOME` 或 `cache_name`。
3. unittest 与 benchmark 不得共享同一缓存并行运行。
4. 进程长时间无输出时，先检查 `jittor.lock` 持有者和编译子进程，再判断
   是否为模型运行卡住。

### 工作区边界

- 主仓库：源码、测试、文档、人工报告和可复用 skill。
- `$JITTOR_LAB_ROOT/<topic>/`：独立实验、下游 checkout、脚本运行目录和产物。
- `$JITTOR_LAB_ROOT/_state/<topic>/<run>/`：`HOME`、`JITTOR_HOME`、`TMPDIR`、
  wheel/模型/编译缓存与原始日志。
- `$JITTOR_LAB_ROOT/worktrees/`：并行 agent 的 Git worktree。
- 不在主仓库顶层新建 `jittor_fsdp2`、`*_work`、`*_probe` 等实验目录。
- 提交前运行 `agent/scripts/check_repo_layout.sh`，直接检查工作区顶层是否越界。

## 目录结构

```
agent/
├── README.md                 # 总索引
├── manuals/                  # 协作、环境、问题总账和上下文索引
├── results/                  # 人工整理的验证与性能报告
└── skills/                   # SKILL.md 与可复用工具
```

长期设计资料不放在 `agent/manuals/` 的主题子目录，按语义写入根目录 `docs/`。
