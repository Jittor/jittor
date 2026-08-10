# Project Rules

## Documentation
完成任何功能、修复或重要改动后，必须在 `/home/zy/projects/doc/` 目录下创建或更新面向项目使用者的中文交付文档，包含：
- 改动的内容摘要
- 用法说明和示例
- 如果涉及新的依赖或环境要求，需要注明

文件命名格式：`<日期>-<主题>.md`，例如 `2026-07-03-trellis2-jittor.md`

## Code Quality
- 主仓库只保留源码、测试、公开文档、`agent/` 协作文档和可复用工具。
- 独立实验、下游仓库和大体积产物统一放在
  `${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}/<topic>/`。
- `HOME`、`JITTOR_HOME`、`TMPDIR`、模型缓存、编译缓存和原始日志统一放在
  `${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}/_state/<topic>/<run>/`，
  不要写入 Jittor 主仓库，也不要使用 `/tmp`、`/dev/shm`。
- Git worktree 放在 `${JITTOR_LAB_ROOT:-/home/zy/projects/jittor-lab}/worktrees/`。
- 提交前运行 `agent/scripts/check_repo_layout.sh`；顶层目录有意调整时，同步更新其允许名单。
- 只 `git add` 当前任务明确涉及的文件，禁止 `git add -A`；完成后提交。
- commit message 用中文，简明扼要

---

## Agent 协作工作流

内部文档入口是 `agent/README.md`。开始任务前先阅读：

1. `agent/manuals/collaboration.md`：协作、验证和 JIT 并发规则。
2. `agent/skills/jittor-dev-context/SKILL.md`：上下文 skill 入口。
3. `agent/manuals/project-context.md`：当前进度、环境、已知问题和待办。

`agent/results/` 只保存供仓库维护者接力的内部验证与性能总结；项目使用者需要
长期查阅的交付说明仍放在 `/home/zy/projects/doc/`。原始日志、缓存、
二进制和大体积 benchmark 结果不进入文档树。可复用工具继续放在
`agent/skills/<name>/`，每个 skill 以 `SKILL.md` 为入口。

必须遵守：

- verify-then-fix，先复现再修改。
- 新增兼容计算至少能在目标 device 上运行，不能只提供 CPU fallback。
- 首次 JIT/扩展编译串行执行；并行任务在 `jittor-lab/_state/` 下使用不同的 `JITTOR_HOME` 或
  `cache_name`，禁止共享编译缓存。
- 不要让 unittest 与 benchmark 共用同一 Jittor cache 并行运行。
- 有稳定结论后更新 `agent/manuals/project-context.md`，详细证据写入
  `agent/results/YYYY-MM-DD-topic.md`。

---

# cc-connect Integration

This project is managed via cc-connect, a bridge to messaging platforms.

## Scheduled tasks (cron)
When the user asks you to do something on a schedule (e.g. "every day at 6am",
"every Monday morning"), use the Bash/shell tool to run:

  cc-connect cron add --cron "<min> <hour> <day> <month> <weekday>" --prompt "<task description>" --desc "<short label>"

Environment variables CC_PROJECT and CC_SESSION_KEY are already set — do NOT
specify --project or --session-key.

Examples:
  cc-connect cron add --cron "0 6 * * *" --prompt "Collect GitHub trending repos and send a summary" --desc "Daily GitHub Trending"
  cc-connect cron add --cron "0 9 * * 1" --prompt "Generate a weekly project status report" --desc "Weekly Report"

To list, run, edit, or delete cron jobs:
  cc-connect cron list
  cc-connect cron exec <job-id>
  cc-connect cron edit <job-id> <field> <value>
  cc-connect cron del <job-id>

Use `cron exec <job-id>` to run an existing scheduled task immediately; this is different from the `--exec <command>` flag used when creating a shell-command cron job.
Use `cron edit` to modify a single field instead of delete-and-recreate.
Common editable fields: cron_expr, prompt, exec, description, enabled (true/false), mute (true/false), timeout_mins (int).
Run `cc-connect cron edit --help` for the full field list.

Examples:
  cc-connect cron exec abc123
  cc-connect cron edit abc123 cron_expr "0 9 * * *"
  cc-connect cron edit abc123 enabled false
  cc-connect cron edit abc123 prompt "Updated daily summary task"

## Send message to current chat
To proactively send a message back to the user's chat session (use --stdin heredoc for long/multi-line messages):

  cc-connect send --stdin <<'CCEOF'
  your message here (any special characters are safe)
  CCEOF

For short single-line messages:

  cc-connect send -m "short message"
