---
name: jittor-worktree-verification
description: 在 git worktree 里验证 Jittor 改动时，确认跑的确实是本 worktree 的代码。用于任何在 worktree（而非主树）中改 python/jittor、src/、backends/ 或 compat/ 后要跑脚本、python -c、mpirun、benchmark 或子进程验证的场合。源码与安装产物分别核对来源，避免 editable 安装或路径注入让验证对象漂移。
---

# 在 worktree 里验证 Jittor 改动

## 先选择验证对象

| 验证对象 | 来源契约 |
| --- | --- |
| 当前 worktree 源码 | 使用仓库测试入口，或显式设置 `PYTHONPATH=<worktree>/python`；确认 `jittor.__file__` 来自目标树 |
| 当前提交构建的 wheel | 在隔离环境安装产物，清除源码路径注入；确认 `jittor.__file__` 来自安装目录并记录构建提交 |

editable 安装可能指向另一棵工作树，也可能已经正确指向当前树；普通 Python 命令是否
正确取决于环境，不能一概判断。源码 checkout 的 pytest 会配置源码路径，nox 各会话
则有自己的源码/产物约定。每次换验证方式都重新核对。

源码自检：

```bash
JITTOR_HOME=<隔离缓存> TMPDIR=<隔离临时目录> \
PYTHONPATH=<worktree>/python \
python -c "import jittor; print(jittor.__file__)"
```

输出应位于 `<worktree>/python/jittor/`。安装产物验证不要使用这条源码命令；在安装环境
中直接打印来源，确认不落在 checkout。核心和 compat 是独立发行物，单独注入核心源码
不会安装 compat。

## 缓存与子进程

每个并发任务使用独立 `JITTOR_HOME` 与 `TMPDIR`，放在 lab 的状态目录。默认缓存位置
可能被其它任务共享；首次 JIT 串行预热，测试与 benchmark 不共享缓存并发运行。
普通脚本、mpirun 和子进程也要验证各自来源；不能假设它们继承了父进程的 `sys.path`。

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
  看起来是「runner 坏了」，不是「某条测试失败了」（这个坑已经有两个人各栽过一次）。
  `crash_isolated=True` 在中间隔一层 `sh`：pytest 的直接子进程永远正常退出
  （`128+signo`，属 `CLD_EXITED`，处理器不理），`returncode` 仍是 134/139，崩溃照样可断言；
  同时把 `gdb_path` 清空，免得崩溃处理器 fork 出的 gdb 把子进程 ptrace-stop 在那里。
  **是 opt-in 不是默认**：包一层 shell 之后 `subprocess.run` 超时只杀得掉 `sh`，
  孙进程会变孤儿，这个代价只该由崩溃测试付。

`tests/structure/test_child_process_contract.py` 会在门禁里静态扫全树，两条规则：
`tests/` 下不许出现 `sys.executable`（改用 `child_process.PYTHON`）；任何起进程的调用
只要碰到解释器或 `mpirun`，就必须走 helper 或显式 `env=child_env(...)`。

这不是洁癖：一次把 core 的 `set_lock_path` 改名成 `set_lock_fd` 的改动之后，
`test_tracer` 的子进程加载的是**分支编的 core**、导入的是**主树的 `python/jittor/build/compiler.py`**，
`AttributeError` 指向的两棵树都不是问题所在。安静的那一半更贵——子进程照样跑通，
测试照样绿，只是它验证的是另一棵树。

## 判据

「跑绿了」不等于通过。还要能回答：

1. `jittor.__file__` 是否符合本次选定的源码或 wheel 来源契约？
2. 如果验证涉及子进程（mpirun / launch / subprocess），**子进程**里的
   `jittor.__file__` 是否也符合相同来源契约？
3. 缓存目录是我自己的 `JITTOR_HOME` 吗？（日志里的 `cache_path:` 一行会打出来，看一眼）
4. 除了本任务的新测试，我跑了**受影响目录的既有测试**吗？

任何一条答不上来，这次验证不算数。

第 4 条最容易被跳过，因为前三条答对之后「新测试全绿」感觉已经够了。它不够：
新测试只覆盖你**已经理解**的那半契约，你改错的往往是你没意识到存在的另一半。
2026-09-03 的一次改动就是这样——为它新写的三条测试全绿，抓到回归的是既有的
`tests/core/test_array.py::TestArray::test_data`，它断言的是同一个契约里
作者没想到的那一半（`Var.data` 的视图要让**整个 Var 对象**活着，不只是那块内存）。
少跑那一组，回归会带着「三条新测试全绿」的提交说明进主干。

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

### 修前与修后对照

为基线与修复准备不同的 worktree 或只读快照，分别记录提交并配置隔离缓存。
不要在当前工作树中反复 checkout、apply 来切换基线，也不要用共享 stash 暂存改动。
Git 共享状态和恢复流程见 [git-worktree-shared-state](../git-worktree-shared-state/SKILL.md)。

## 跑套件期间不要改源码

pytest 是**边跑边 import** 的：模块在第一次被用到时才加载。整套 `compat/tests/torch`
要 15–25 分钟，这期间你对 `python/jittor/**` 的任何一次写入，都会被后面才 import 的
测试读到——包括你用来做「修前失败」验证的 `git checkout --` 与 `git apply`。

实测：一次 `compat/tests/torch` 在后台跑着，我在中途改了两个 shim 文件、又
checkout 回去、再 apply 回来，结果是 **373 failed / 563 passed**；同一棵树静止不动
重跑是 **1 failed / 943 passed**。那 373 条与改动毫无关系，全是半截源码造成的假失败。

所以：

- **后台跑套件时，工作区必须静止。** 要么等它跑完，要么先把要做的编辑做完再启动。
- 修前与修后分别在冻结的工作树运行，避免后台测试读到混合版本。
- 看到「大面积、跨文件、与改动无关」的失败，第一反应先问「刚才这棵树动过吗」，
  再去查缓存损坏或并发。静止重跑一次比读 373 条 traceback 快得多。

## 改了 C++ 之后

C++ 改动触发受影响单元的增量编译；是否需要编译由源码、缓存和构建配置决定，
不是每个新进程都重编。首次构建单列计时，并为运行预留实际冷编译预算。

- 首次 JIT/扩展编译串行完成，随后按隔离缓存规则运行多进程测试。
- 不要在编译进程存活时删除缓存。需要中止时先停止本任务进程及子进程，保留诊断，
  再用新的隔离缓存复现；不清理其他任务状态。
- 同步或换配置后的结果只代表实际测试的提交与缓存条件，不能直接复用旧结论。
