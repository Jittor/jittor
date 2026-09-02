---
name: git-worktree-shared-state
description: 多个 agent 各自在一个 git worktree 里并行改同一个仓库时，哪些 git 状态是共用的、哪些不是。用于任何会写 .git 的操作前（stash、branch、reset、gc、rebase 中断处理），尤其是 rebase 前想临时搁置改动的时候。git stash 是全仓库唯一一个栈，两个 agent 交错 push/pop 会互相拿走对方的改动，且不报任何错。
---

# 多 worktree 并行时，git 的哪些东西是共用的

`git worktree add` 只复制**工作区和索引**，`.git` 目录是**共用的一份**。
所以「我在自己的 worktree 里」并不等于「我的 git 操作只影响自己」。

| 共用（写它就影响所有人） | 不共用（每个 worktree 一份） |
| --- | --- |
| 对象库、`refs/heads/*`、`refs/tags/*`、`refs/remotes/*` | `HEAD`、当前分支 |
| **`refs/stash`（stash 是全仓库唯一一个栈）** | 索引（index）与工作区文件 |
| `config`、`hooks/`、`packed-refs` | `.git/worktrees/<name>/` 下的 ORIG_HEAD、rebase 状态 |
| `git gc` / `prune` 的效果 | `git status` 的结果 |

## 已经踩过的事故：stash 对调

两个 agent 各自在 rebase 前用 `git stash push <自己的文件>` 搁置改动：

```
dist:    stash push (5 个 dist 的文件)     -> stash@{0} = dist
pyother: stash push (2 个 pyother 的文件)  -> stash@{0} = pyother, stash@{1} = dist
dist:    stash pop                          -> 拿到 pyother 的 2 个文件
pyother: stash pop                          -> 拿到 dist 的 5 个文件
```

两边的改动**完整对调**，git 全程零报错零警告。发现它的唯一途径是
`git status` 里冒出自己没碰过的文件，或者自己刚写的测试类突然「不存在」。

## 规则

1. **禁止 `git stash`**（包括 `git stash push/pop/apply/list`）。没有例外——
   带路径参数、带 `-m` 消息都一样共用同一个栈。
2. rebase 前清理工作区的正确做法是**一次只做一个任务**：改完 → 测完 → 提交 →
   再 `git fetch && git rebase && git push`。不要让无关的 WIP 留在树里。
3. 确实要临时搁置：存到**自己目录下**的补丁，再 checkout 掉。

   ```bash
   git diff -- <你的文件> > <你自己的目录>/wip.patch
   git checkout -- <你的文件>
   # 需要时再 git apply <你自己的目录>/wip.patch
   ```

4. 禁止 `git add -A`：一旦发生过 stash 对调，别人的文件就躺在你的工作区里，
   `-A` 会把它们提交进你的提交。始终显式列出自己改的文件。
5. 禁止 `git gc --prune=now`、`git reflog expire`、`git branch -f/-D` 别人的分支、
   `git checkout` 别人的分支（worktree 之间同一个分支不能同时检出，git 会拒绝，
   但 `-f` 能绕过去，别绕）。
6. `git fetch` / `rebase` / `push origin HEAD:<branch>` 是安全的：远端的
   非快进保护会把并发推挤成串行，推被拒就 `fetch && rebase` 再推。

## 万一已经拿到了别人的东西

按这个顺序做，先保存再清理，**不要直接 `git checkout --` 丢掉**：

1. 存补丁：`git diff > <公共救援目录>/<对方分区>-<时间戳>.patch`
2. 原样放回（让对方能拿到），并**明确告诉对方用 `git apply` 而不是 `git stash pop`**，
   避免再错位一次。
3. 通知协调者与对方 agent，说清哪些文件、存在哪里。
4. 确认 `git status` 干净后再继续自己的活。**任何时候都不要把别人的文件混进自己的提交。**

## 自检

提交前一定看一眼 `git status --short`：里面**只能**有你这个任务碰过的文件。
出现你不认识的路径，先按上面的抢救流程处理，再继续。
