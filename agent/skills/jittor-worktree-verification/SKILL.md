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

`pyproject.toml` 里有 `pythonpath = ["python"]`，pytest 会把 **rootdir 下的 `python/`**
插到 `sys.path[0]`。所以在 worktree 里跑 pytest 天然导入本 worktree——**这是唯一
自动正确的入口**。

```bash
# 正确：在 worktree 里直接跑，不需要 PYTHONPATH
cd <worktree> && JITTOR_HOME=... TMPDIR=... pytest tests/<...> -x -q
```

要用 pytest 测**另一个**目录的副本，必须显式覆盖：`-o pythonpath=<那个目录>/python`。

会**静默测错代码**的入口（都要手动加 `PYTHONPATH`）：

- `python -c "..."` 和 `python some_script.py`
- `mpirun ... python ...`（每个 rank 各自 import，全都会走主树）
- 任何 `subprocess` / `os.system` 起的子进程
- benchmark 脚本、profile 脚本、复现脚本
- `python -m` 起的模块

`pytest` 起的子进程也一样：pytest 改的是自己进程的 `sys.path`，**不会**写
`os.environ["PYTHONPATH"]`，所以子进程不继承。测试里 spawn 子进程时有两条出路：

- **子进程还是 pytest，且 `cwd` 设成仓库根**——rootdir 会重新解析到这个仓库，
  `pythonpath = ["python"]` 再次生效，天然正确。`tests/_helpers/distributed.py`
  的 `run_mpi_test()` 走的就是这条（`mpirun -np N python -m pytest <abs path>`，
  `cwd=repo_root`），所以它是对的。
- **子进程是裸 python**——必须显式传 `env["PYTHONPATH"] = <repo_root>/python`，
  否则父进程测的是 worktree、子进程测的是主树，两边结论会打架。

## 判据

「跑绿了」不等于通过。还要能回答：

1. `jittor.__file__` 在我的 worktree 里吗？
2. 如果验证涉及子进程（mpirun / launch / subprocess），**子进程**里的
   `jittor.__file__` 也在我的 worktree 里吗？
3. 缓存目录是我自己的 `JITTOR_HOME` 吗？（日志里的 `cache_path:` 一行会打出来，看一眼）

任何一条答不上来，这次验证不算数。

## 改了 C++ 之后

改 `python/jittor/src/**` 或 `python/jittor/extern/**` 之后，**每个新进程**都要重编，
第一次约 10 分钟（核心约 156 个 TU）。所以：

- C++ 改动攒着一次验证，不要改一行跑一次。
- 编译期间**不要 `kill -9`**。真要杀，先把整个 `JITTOR_HOME` 删掉再重跑。
- 多进程验证（mpirun）第一次会串行编译（`jittor.lock` 是 flock 互斥的），时间是单进程
  的 N 倍，别按单进程的耗时设超时。
