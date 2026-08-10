# Jittor Agent 协作手册

本手册说明 AI agent 在 Jittor 仓库内的协作、验证和文档规则。

## 开工流程

1. **先读上下文**：阅读 `../skills/jittor-dev-context/SKILL.md`，再按入口阅读
   `project-context.md`，了解目标、进度、环境和已知问题。
2. **确认结果文档**：已有主题继续更新 `../results/` 中的报告；新主题使用
   `YYYY-MM-DD-topic.md`。
3. **开始工作**，遵循下面的协作规范。

## 协作规范

### 文档更新（核心纪律）

- **`project-context.md`** 是全局进度文档。完成有意义的进展后，更新其中
  对应的状态或索引，不要把完整实验流水重复粘贴进去。
- **结果报告**记录验证口径、命令、结果、结论和已知边界。原始日志、缓存、
  二进制和大体积结果放在 `${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}`，
  报告中标明“未版本化”，不要放进 Jittor 主仓库。
- 更新文档时注明状态（✅ 已完成 / 🟡 进行中 / 🔴 有问题），便于他人接手。

### Skill 沉淀

工作过程中写出的**可复用工具**（对拍脚本、验证 harness、调试探针等），沉淀为 skill：

- 在 `../skills/` 下创建新目录，包含 `SKILL.md`（说明用途和用法）与工具文件。
- 已有 skill 直接复用，别重新造轮子。

### 效率原则

- **GPU 先行**：先在 GPU 上跑通全流程，再做 NPU 验证。
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
├── manuals/                  # 协作、上下文和设计手册
├── results/                  # 人工整理的验证与性能报告
└── skills/                   # SKILL.md 与可复用工具
```
