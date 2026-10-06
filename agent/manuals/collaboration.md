# Jittor Agent 协作手册

本手册说明 AI agent 与维护者在 Jittor 仓库内的协作、验证和文档规则。入口索引见
[agent-index.md](agent-index.md)。

## 开工流程

1. **先同步**：按 [`AGENTS.md`](../../AGENTS.md) 获取并整合目标远端分支，记录同步后的提交 SHA；
   保留本地工作，不用 stash 或丢弃改动来清场。
2. **再读上下文**：阅读 [`jittor-dev-context`](../skills/jittor-dev-context/SKILL.md)，
   再通过 [`project-context.md`](project-context.md) 定位相关文档。
3. **确认环境与问题**：按需阅读 [`environment.md`](environment.md) 和
   [`known-issues.md`](known-issues.md)，不要依赖个人机器路径或过期会话记录。
4. **确认既有证据**：在 [`docs/results/`](../../docs/results/index.md) 查找已有结论；新主题的
   报告使用 `YYYY-MM-DD-topic.md`。
5. **开始工作**，遵循下面的协作规范。

## 协作规范

### 文档更新（核心纪律）

- **`project-context.md`** 是当前状态索引，只在目标、状态或文档入口变化时更新。
- **`known-issues.md`** 是未关闭缺陷与已知限制的总账。新增条目按其头部的格式写清
  severity、owner、可执行证据、workaround 和退出条件；修复后删除条目，修复提交在提交信息或
  PR 中注明条目编号。
- **`deferred-hardware.md`** 只记等硬件验收的事项与上机命令，不记进度。
- **`docs/`** 保存长期架构决策、测试契约、开发指南和研究提案。
- **结果报告**记录验证口径、命令、结果、结论和已知边界。原始日志、缓存、
  二进制和大体积结果放在 `$JITTOR_LAB_ROOT`，
  报告中标明“未版本化”，不要放进 Jittor 主仓库。
- 稳定文档注明状态、复查日期、对应基线、owner 和复查触发条件。
- 任务领取、状态、交接与验收证据记在 GitHub issue 与 PR 中，仓库不保留看板或交接文档。

### Skill 沉淀

工作过程中写出的**可复用工具**（对拍脚本、验证 harness、调试探针等），沉淀为 skill：

- 在 `../skills/` 下创建新目录，包含 `SKILL.md`（说明用途和用法）与工具文件，并在
  [agent-index.md](agent-index.md) 加一行。
- skill 里的命令要能在当前树上运行：路径用 `git ls-files` 核对，机器相关的位置写成
  `$JITTOR_LAB_ROOT/...`、`<jittor-python>`、`CUDA_VISIBLE_DEVICES=<gpu>` 这类占位，
  脚本不带个人默认路径（未设置时报错退出）。
- 已有 skill 直接复用，别重新造轮子；方法不再适用于当前树的 skill 应删除或并入相关 skill。

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
- 提交前运行 `bash tools/check_repo_layout.sh`，直接检查工作区顶层是否越界；
  `tests/structure/test_no_private_paths.py` 拒绝个人目录、主机地址和按主机名分支的代码。

`agent/` 只有 `manuals/` 与 `skills/` 两个目录；长期设计资料按语义写入根目录 `docs/`，
仓库维护命令统一放在 `tools/`，wheel 内容基线位于 `tools/release/baselines/`。
