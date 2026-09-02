---
name: jittor-worktree-verification
description: 在 git worktree 里验证 Jittor 改动时，确认跑的确实是本 worktree 的代码。用于任何在 worktree（而非主树）中改 python/jittor 或 python/jittor/src 后要跑脚本、python -c、mpirun、benchmark 或子进程验证的场合。裸 python 命令会静默导入主树，测出来的绿是别人的绿。
---

# 在 worktree 里验证 Jittor 改动

**核心事实**：Jittor 通常以 **editable 方式**装进 conda 环境。`site-packages` 里是一个
`.pth` 文件，指向**主树**的 `python/` 目录，不是你的 worktree。所以在 worktree 里执行
裸 `python -c "import jittor"`，导入的是**主树**代码。你的改动一行都没生效，测出来的
结果与你无关。

这条不会报错、不会警告、不会有任何迹象。唯一的症状是「改了没反应」或「修完还是错」。

## 开工自检（每个新 worktree 做一次，每次换验证方式再做一次）

```bash
# 必须打印出你自己的 worktree 路径
JITTOR_HOME=<你的 _home> TMPDIR=<你的 _tmp> \
PYTHONPATH=<你的 worktree>/python \
python -c "import jittor, os; print(os.path.dirname(os.path.dirname(jittor.__file__)))"
```

把它和 `git rev-parse --show-toplevel` 比一下。不相等就说明你在测别人的代码。

想看 editable 指向哪里：

```bash
cat "$(python -c 'import site;print(site.getsitepackages()[0])')"/__editable__*jittor*.pth
```

## 三件套环境变量：少一个会发生什么

| 变量 | 作用 | 少了会怎样 |
| --- | --- | --- |
| `PYTHONPATH=<worktree>/python` | 让 `import jittor` 命中本 worktree | **静默导入主树**，改动完全不生效 |
| `JITTOR_HOME=<自己的缓存目录>` | 隔离 JIT 缓存 | 与别的 agent 共用缓存 → 缓存损坏 → 在毫不相干的算子上大面积报错 |
| `TMPDIR=<自己的临时目录>` | 隔离编译临时文件 | 同上，且可能写满 `/tmp` |

三个一起写，不要只写一个。

## pytest 是对的，手写命令不是

`tests/conftest.py` 会把 **本 checkout 的 `python/`**
插到 `sys.path[0]`。所以在 worktree 里跑 pytest 天然导入本 worktree——**这是唯一
自动正确的入口**。

```bash
# 正确：在 worktree 里直接跑，不需要 PYTHONPATH
cd <worktree> && JITTOR_HOME=... TMPDIR=... pytest tests/<...> -x -q
```

要用 pytest 测**另一个**目录的副本：`JITTOR_SOURCE_ROOT=<那个目录> pytest ...`
（以前要在每条命令上写 `-o pythonpath=<那个目录>/python`）。

会**静默测错代码**的入口（都要手动加 `PYTHONPATH`）：

- `python -c "..."` 和 `python some_script.py`
- `mpirun ... python ...`（每个 rank 各自 import，全都会走主树）
- 任何 `subprocess` / `os.system` 起的子进程
- benchmark 脚本、profile 脚本、复现脚本
- `python -m` 起的模块

### 测试里起子进程：只有一个入口，而且是强制的

`pytest` 起的子进程也一样：pytest 改的是自己进程的 `sys.path`，**不会**写
`os.environ["PYTHONPATH"]`，所以子进程不继承。

不要再自己拼 `subprocess.run([sys.executable, ...])`。全树统一走
`tests/_helpers/child_process.py`：

```python
from _helpers.child_process import run_python_child, run_child_script, run_mpi_python

r = run_python_child(["-c", src], env={"use_cuda": "0"}, merge_stderr=True)
r = run_child_script(source_text)                    # 写成文件再跑，traceback 有真行号
r = run_mpi_python(2, [script_path])                 # mpirun -np 2 python script
```

要点：

- `PYTHONPATH` 由 `child_env()` 钉住，且钉的是 `conftest.source_python_dir()` ——
  和父进程 `sys.path[0]` **同一个函数**，两边不可能漂移。
- `env=` 是**叠加**在 `os.environ` 上的。调用方如果是先 `dict(os.environ)` 再
  **删掉**某个变量，必须传 `inherit=False`，否则合并会把它原样加回来。
- 默认超时 900 s，`JITTOR_TEST_CHILD_TIMEOUT` 可调。冷子进程要编译整个核心，
  按空闲机器调的 180 s 在有负载时必然假红。
- 需要进程句柄（`Popen` 让子进程保持活着）时才自己起，但必须
  `env=child_env()`。
- **子进程可能被信号杀死（段错误 / abort）时必须传 `crash_isolated=True`。**
  jittor 装了一个进程级 SIGCHLD 处理器（`src/utils/log.cc`）：直接子进程**非正常退出**
  时它让父进程 quick-exit。于是「把会崩的用例放子进程里跑」这个标准做法**反过来生效**——
  子进程 abort，处理器在 pytest 里触发，pytest 中途消失、`-q` 缓冲里的输出全丢。
  看起来是「runner 坏了」，不是「某条测试失败了」（6.C31，两个分区各栽过一次）。
  `crash_isolated=True` 在中间隔一层 `sh`：pytest 的直接子进程永远正常退出
  （`128+signo`，属 `CLD_EXITED`，处理器不理），`returncode` 仍是 134/139，崩溃照样可断言；
  同时把 `gdb_path` 清空，免得崩溃处理器 fork 出的 gdb 把子进程 ptrace-stop 在那里。
  **是 opt-in 不是默认**：包一层 shell 之后 `subprocess.run` 超时只杀得掉 `sh`，
  孙进程会变孤儿，这个代价只该由崩溃测试付。

`tests/structure/test_child_process_contract.py` 会在门禁里静态扫全树，两条规则：
`tests/` 下不许出现 `sys.executable`（改用 `child_process.PYTHON`）；任何起进程的调用
只要碰到解释器或 `mpirun`，就必须走 helper 或显式 `env=child_env(...)`。

这不是洁癖：`[0.08]` 把 core 的 `set_lock_path` 改名成 `set_lock_fd` 之后，
`test_tracer` 的子进程加载的是**分支编的 core**、导入的是**主树的 `compiler.py`**，
`AttributeError` 指向的两棵树都不是问题所在。安静的那一半更贵——子进程照样跑通，
测试照样绿，只是它验证的是另一棵树。

## 判据

「跑绿了」不等于通过。还要能回答：

1. `jittor.__file__` 在我的 worktree 里吗？
2. 如果验证涉及子进程（mpirun / launch / subprocess），**子进程**里的
   `jittor.__file__` 也在我的 worktree 里吗？
3. 缓存目录是我自己的 `JITTOR_HOME` 吗？（日志里的 `cache_path:` 一行会打出来，看一眼）

任何一条答不上来，这次验证不算数。

## 多个 worktree 并行时，`.git` 里哪些是共用的

同一个仓库的所有 worktree **共用一个 `.git` 目录**。哪些状态是共用的、哪些是本 worktree
私有的，决定了你能不能把某个操作当成"只影响我自己"。

| 东西 | 共用还是私有 | 后果 |
| --- | --- | --- |
| `refs/stash`（stash 栈） | **共用** | **别人的 `git stash pop` 会拿走你的改动** |
| `refs/heads/*`、`refs/remotes/*`、所有 tag | 共用 | 你能看见别人的分支；`git fetch` 互相影响 |
| objects（提交、blob） | 共用 | 好事：别人的悬垂 commit 你也能找回 |
| `HEAD`、index（暂存区） | 每 worktree 私有 | `git add` 只影响自己 |
| HEAD 的 reflog | 每 worktree 私有（在 `.git/worktrees/<name>/`） | |

**因此：worktree 里禁止 `git stash`。** stash 栈是全局的一个栈，两个 agent 各自
`push` 再各自 `pop`，拿回来的是对方的改动——这不是理论风险，已经真实发生过一次完整对调。

### 要做「修前失败、修后通过」怎么办

用补丁文件，存在**自己的 `TMPDIR`** 里：

```bash
git diff -- <你改的文件> > "$TMPDIR/fix.patch"   # 存下修后状态
git checkout -- <你改的文件>                      # 回到修前
<跑测试，确认它失败——这一步才是重点>
git apply "$TMPDIR/fix.patch"                     # 回到修后
<再跑一次，确认它通过>
```

新增的文件不在 `git diff` 里，用 `mv` 挪走再挪回来，**不要** `git stash -u`。

### 万一已经被 stash 串号了

stash commit 只是变成悬垂对象，没被删。按 message 找回来：

```bash
for c in $(git fsck --unreachable --no-progress 2>/dev/null | grep commit | awk '{print $3}'); do
  echo "$c :: $(git log -1 --format='%ci %s' $c)"
done | sort -k2
```

stash 的 message 形如 `On <分支>: <你写的 -m 文本>` 或 `WIP on <分支>: <提交标题>`。
认出自己那条之后 `git stash apply <commit>`（先把误 pop 进来的别人的文件
`git checkout --` 还原掉，别提交它们；还原前先 `git diff > 一份patch` 存起来还给对方）。

**所以万一非要 stash，也一定要 `-m "<任务编号>"`**——没有 message 的话事后分不清哪条是自己的。

## 改了 C++ 之后

改 `python/jittor/src/**` 或 `python/jittor/extern/**` 之后，**每个新进程**都要重编，
第一次约 10 分钟（核心约 156 个 TU）。所以：

- C++ 改动攒着一次验证，不要改一行跑一次。
- 编译期间**不要 `kill -9`**。真要杀，先把整个 `JITTOR_HOME` 删掉再重跑。
- 多进程验证（mpirun）第一次会串行编译（`jittor.lock` 是 flock 互斥的），时间是单进程
  的 N 倍，别按单进程的耗时设超时。
