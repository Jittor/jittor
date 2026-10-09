---
name: github-collaboration-commit
description: 多人或多 agent 在 Jittor 仓库协作时的同步、提交、推送与 Pull Request 流程：开工前以远端为准同步并记录 SHA、只暂存当前任务的文件、简明中文提交信息、只做快进推送、PR 描述写清验证证据与未完成项。开始或恢复一个任务、准备提交或推送、发起或更新 PR、交接未完成工作时使用。
---

# 协作、提交与 Pull Request

本 skill 把 `AGENTS.md` 的协作规则落成命令。下文 `<target>` 指本次任务的目标远端分支
（默认分支是 `master`；以任务说明为准），`origin` 指它所在的远端。

## 1. 开工或恢复任务：先以远端为准同步

```bash
git status --short --branch            # 先看有没有未提交改动
ps -eo pid,etimes,args | grep -E "[p]ytest|[n]ox"   # 同一工作树里有没有在跑的测试
git fetch origin <target>
git log -1 --format='%H %s' origin/<target>          # 记下同步基线的 SHA
```

- **保留本地工作，不清场**：禁止 `git stash`（stash 栈是所有 worktree 共用的，见
  `git-worktree-shared-state`），禁止用 `git checkout -- .`、`git reset --hard` 丢弃改动来同步。
- **整合期间冻结同一工作树**：停下本工作树里的测试与协作者写入，整合完再继续。
- 本地改动已提交在个人分支上：`git rebase origin/<target>`（分支未共享时）或
  `git merge origin/<target>`（已被他人引用时）。还未提交的改动先在个人分支上提交成一个
  WIP 提交（只暂存自己的文件），再 rebase 或 merge，整合完再整理提交信息：

  ```bash
  git add <你的文件...> && git commit -m "WIP：<主题>"   # 显式路径，不用 -A
  git rebase origin/<target>                             # 冲突逐个解决
  ```

  不要用 `git checkout -- <文件>` 或 `git reset` 把工作树清空后再同步。

- **重叠处以远端实现为准**，再把本地意图重新落在它上面；不得用旧的本地版本覆盖远端更新。
  冲突文件逐个读，必要时对比合并基线与双方提交（`git merge-file`，见 `codebase-wide-fix-as-rule`）。
- 同步前跑出的结果属于旧基线：在 PR 或交接里按旧 SHA 记录，并**按实际带进来的变更复验受影响范围**，
  不要把旧结果当成新基线的证据。

并行任务用独立 worktree，放在 `$JITTOR_LAB_ROOT/worktrees/`：

```bash
git worktree add "$JITTOR_LAB_ROOT/worktrees/<topic>" -b <type>/<short-topic> origin/<target>
```

分支名小写、短、说明目的（`fix/npu-reduction-gradient`、`docs/agent-manuals`），不带机器路径或实验编号。

## 2. 提交

```bash
git status --short          # 只能出现本任务碰过的文件；不认识的路径先查清来源
git diff -- <文件...>
git add <文件1> <文件2>      # 显式路径，不用 git add -A / git add . / git commit -a
git diff --cached --stat
git commit                  # 简明中文提交信息
git show --stat --format="" HEAD   # 确认提交里没有你没碰过的路径
```

- 一个提交表达一个可独立理解的逻辑步骤，不混入无关格式化、缓存、日志、模型或个人环境文件。
  拆开会造出比修复前更糟的中间状态时才合并，并在提交信息里写明为什么不能拆。
- 提交信息标题用简明中文，以动词或变更对象开头，例如「修复 NPU 归约反向的 dtype 处理」。
  改动大或原因不明显时在正文写：问题、原因、改动、验证（命令与 device）、限制。
- 提交前运行仓库约定的检查：

  ```bash
  bash tools/check_repo_layout.sh
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
  python tools/run_test_suite.py --tier core    # 改了代码时
  ```

## 验证成本与续接边界

自动监督中的同一运行键轮询不是新任务；作业状态无变化时不重新 fetch、不重复跑门禁或创建进度提交。新任务开始、集成前与推送前仍按本 Skill 和 `AGENTS.md` 获取并核对远端 SHA。个人 fork 的拉取和推送优先用同一已验证 SSH 身份；短暂网络故障保留上次可信 SHA，不把过期 tracking ref 说成实时远端。

原始进度和编译日志留在 state；一个逻辑问题的结论、根因或覆盖边界变化时再整理仓库结果。多个纯文档更新先合并为一次可审阅提交，避免每条中间观察都重复完整结构门禁。提交前仍执行仓库 `AGENTS.md` 强制的布局和结构检查；测试未完成或失败要如实记录，不能用定向通过代替全量通过。

## 3. 推送：只做快进

```bash
git fetch origin <target>
git merge-base --is-ancestor origin/<target> HEAD && git push origin HEAD:<target>
```

- 推送被拒（远端前进了）就回到第 1 节重新整合、复验，再推。**不 force push**，
  不对共享分支 `reset` 或改写历史。
- 个人分支推到自己的分支名（`git push origin HEAD:<type>/<short-topic>`），同样不 force；
  确需整理个人分支历史时，先确认没有别人基于它工作。

## 4. Pull Request

只有问题已经完整解决、实现与测试可审阅时才发 PR。描述至少写：

1. **问题和结果**：触发场景，合并后行为如何变化；
2. **范围**：改了哪些模块，哪些有意不包含；
3. **验证证据**：实际执行的命令、结果计数、device、同步基线 SHA 与缓存说明；
   没执行的（缺 CUDA/NPU/ROCm、缺依赖）写成「未执行」而不是通过；
4. **兼容与风险**：导入路径、序列化、后端、下游库、性能；
5. **未完成事项**：未覆盖的后端、已知限制、后续工作。缺陷记入
   `agent/manuals/known-issues.md`，等硬件的验收记入 `agent/manuals/deferred-hardware.md`。

发 PR 前的最终检查：

```bash
git fetch origin <target>
git log --oneline origin/<target>..HEAD
git diff --stat origin/<target>...HEAD
git diff --check origin/<target>...HEAD
```

review 意见用新的提交处理，并在 PR 里说明对应的验证。

## 5. 交接

暂停或转交时在对应的 GitHub issue 或 PR 里记录：分支与最新提交、同步基线 SHA、已完成与未完成的部分、
已运行的命令与结果、已知失败及判断、需要注意的冲突区域。不要修改别人的 worktree；
复用别人的代码时引用具体提交或 PR，不复制工作区文件。
