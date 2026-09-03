# 交接：`2.0-refactor` 的当前状态

2026-09-02。上一轮并行执行在此中断。本文写给接手的人，只回答三个问题：**分支停在哪、活的残留在哪、
下一步从哪开始**。任务本身在 [整改计划](refactor-plan.md)，进度在 [看板](refactor-board.md)，目录终点在
[目标布局](target-layout.md)，本文不重复它们。

下文用 `$LAB` 指代实验根目录下的 `refactor/`（即 `$JITTOR_LAB_ROOT/refactor`）。所有 worktree、缓存与
共用简报都在那里，不在仓库里。

## 0. 从哪起步

| | |
| --- | --- |
| 分支 | **`2.0-refactor`**，不是 `2.0` |
| 工作目录 | **`$LAB/coord`**——`2.0-refactor` 上一个干净的 checkout，有 `AGENTS.md` 与全套文档 |
| 六处未提交残留 | 同级目录：`../dist`、`../pyother`、`../cudabk`、`../gates`、`../pyops`、`../bindings` |
| 共用简报 | `$LAB/AGENT-BRIEF.md`（在仓库外，因为含机器相关路径） |

**不要用主树。** 它在 `2.0` 上，那里看不到本文与 [怎么派活](refactor-dispatch.md)，而且——

> **`2.0` 分支上的 `refactor-board.md` 是写计划时的初版，265 行全标着「待领」。** 照它干活会把已经
> 合并的 132 条重做一遍。看板只认 `2.0-refactor` 上的那份。

主树里还躺着三个属于别人的未提交文件（见第 9 节），切分支会带上它们。

## 1. 状态

| | |
| --- | --- |
| 分支 | `2.0-refactor`，head `9a9ea4f8` |
| 相对 `2.0` 的提交 | 345 |
| 提交里出现过的任务号 | 159 |
| 看板 | 已合并 **132** / 进行中 **16** / 待领 **120** |
| 沉淀的 skill | `agent/skills/` 下 **28** 个 |

**整改没有做完，是中途停的。** 265 行任务里 132 行合并、120 行从未开始，停的原因是执行额度用尽而不是
收敛。这个分支不是终态，别按「已完成」读它。

看板的「已合并」是权威。提交里的任务号更多，是因为一个任务常有补充提交、改判提交与「更正前一个提交」
的提交；数任务要数看板行，不要数提交。

## 2. 保存点：五个 `wip/*` 分支

中断时有五个 worktree 有**已提交但未推**的工作。它们已被原样推到各自的分支，不混进 `2.0-refactor`：

| 分支 | 提交 | 内容 |
| --- | --- | --- |
| `wip/bindings` | `9acf74c4`、`1ea98107` | 2.19 之一（生成的 `tp_dealloc` 不得抛、不动解释器异常状态）；2.16 补（两个标量相加的结果不再取决于谁在右边） |
| `wip/build` | `34f19fa2` | 9.21 前半（拆开 `process()` 的两个职责：`JT_*` 宏发现移到 Python，扫描器不再改写命令行） |
| `wip/codegen` | `3d4183f7` | 3.12（代码生成 pass 不再 `catch(...)` 吞掉自己的失败） |
| `wip/gates` | `c8de551e` 等 4 个 | 0.15 分层清单实测秒数与并行度单点声明、0.16 改判（设备对拍不分片）、一次自我更正 |
| `wip/pyops` | `6ed45cea` | 5.04（参数模型：一份遍历、一张角色表，五个方法都是它的视图） |

**这五个分支没有跑完各自的验证就被中断了。** 接手的第一件事不是直接合，是按简报第 2 节把每个提交的
验证补完（改了 C++ 的要各跑一次 CPU 与 CUDA），再 `git rebase origin/2.0-refactor` 合入。合完删分支。

## 3. 未提交的残留

六个 worktree 里有未提交改动，都还在磁盘上，`git status` 能看到全貌：

| worktree | 文件数 | 是什么 |
| --- | --- | --- |
| `$LAB/dist` | 8 | MPI/NCCL reduce 算子（`mpi_reduce_op.{cc,h}`、`mpi_broadcast_op.cc`、`nccl_reduce_op.cc`）、`compat/fsdp2/common.py`，外加两个新测试：`tests/distributed/test_fsdp2_flat_policy.py`、`test_mpi_graph_isomorphism.py` |
| `$LAB/pyother` | 9 | 5.25 拆 `utils/`：`jtune`/`nvtx` 迁入 `python/jittor/tools/`，`gen_pyi`/`local_doc_builder`/`bench_klo` 迁入顶层 `tools/`，配套结构测试同步 |
| `$LAB/cudabk` | 6 | wheel 与 `compile_extern` 一线（`cuda_wheel.py`、`compile_extern.py`、`pyproject.toml`、`tests/structure/test_cuda_wheel.py`、README 与 torch 兼容文档） |
| `$LAB/gates` | 3 | `.github/workflows/cpu.yml`、`tests/_helpers/tiers.py`、`tests/structure/test_gate_tiers.py`（0.15 的收尾） |
| `$LAB/pyops` | 1 | 新测试 `tests/nn/test_module_dtype_cast.py`（5.19 的一半） |
| `$LAB/bindings` | 1 | `optim_test.tar`，是调试产物，删掉即可 |
| `$LAB/mem` | 1 | 只有看板的一行状态改动 |

`compat`、`coreops`、`device`、`gatecheck`、`coord` 五个 worktree 干净，无残留。

**注意 `cudabk` 与 `dist` 有已 `git add` 进索引的文件**（`git status` 的第一列是 `M`）。这是 `git apply --3way`
留下的，直接 `git commit` 会把不相干的东西一起带进去。提交前先看 `git diff --cached --stat`。

## 4. 停在半路的线索

中断时各分区正在做的事，按对接手人的价值排序：

1. **compat：FSDP2 的内存增长找到了复现口径。** 最后一步的结论是「每步加一次 `gc.collect()` 内存就稳住
   了」，正在把它收敛成最小复现。这是一条真线索——说明泄漏是引用环而非分配器，接着往下查引用环的持有者
   就行，不必从头找。
2. **gates：0.15 的两个方向都已验证**，只差再跑一次 `tests/structure` 就能提交（残留见上表）。
3. **bindings：上游 `786658d9` 已经修掉了它正在修的东西**，正准备 fetch、rebase、重验。合 `wip/bindings`
   之前先确认这一条，避免重复修。
4. **pyops：5.04 的实现已提交（`wip/pyops`），测试刚开始写。**

## 5. 看板上 16 条「进行中」

`0.06 0.15 5.19 6.C10 6.C11 6.C12 6.C13 6.C14 6.C17 6.C18 6.C19 6.C20 6.C21 7.09 7.10 7.11`

其中 `6.C10`–`6.C21` 是同一个分区的一串连号，`7.09`–`7.11` 同理。接手时先把这些行的状态按第 2、3 节的
实际情况改对（有提交的写提交号，只有未提交残留的退回「待领」并在备注里写清残留在哪个 worktree），
不要留着「进行中」——没有 agent 在跑它们了。

## 6. 从未重启的分区

`dist`、`device`、`mem`、`build` 四个分区在最后一次中断后没有再启动。它们名下的待领任务是**下一轮的
起点**，因为其它分区的待领任务多半有前置依赖还没落地。

## 7. 接手怎么开始

0. 派活的话术、验收该问什么、哪些说法会让它跑偏，在 [怎么派活](refactor-dispatch.md)。
1. 按第 0 节确认分支与工作目录，读 `$LAB/AGENT-BRIEF.md`（所有 agent 共用的规则：worktree 与缓存约定、提交纪律、会制造假失败的坑）。
   它在仓库外，因为里面有机器相关的路径。
2. 读本文第 2、3 节，把五个 `wip/*` 验证后合入，把六处残留各自收尾或丢弃。
3. 按第 5 节修看板状态。
4. 再按计划的波次派活。`$LAB/clean-idle-caches.sh` 按 mtime 清空闲缓存，长跑期间挂着跑——**不要按体积
   挑**，体积大的往往正是在跑的那个。

## 8. 这轮的教训已经在 skill 里

28 个 skill 覆盖了这轮踩过的坑。跟并行执行本身最相关的三个，派活前让每个 agent 都读：

- `git-worktree-shared-state`——`git stash` 的栈是所有 worktree 共用的，两个 agent 的 WIP 会整包对调。
- `jittor-worktree-verification`——可编辑安装的 `.pth` 指向主树，worktree 里裸跑 `python -c "import jittor"`
  导入的是主树而不是你改的代码。
- `verifying-a-gate-actually-ran`——绿不等于跑过；先确认用例真的被实例化了。

还有一条不是 skill 而是纪律：**rebase 冲突要逐块解，不要整文件取一侧**。这轮有一次整文件解冲突静默
回退了 40 个已合并的 handler，是靠 `tests/structure` 才发现的。

## 9. 环境残留

- 主树 `/`（`2.0` 分支）上有**三个属于别人的文件**不要提交：`agent/manuals/README.md`、
  `tests/core/test_setitem.py`、`agent/results/2026-08-12-repository-modernization-review.md`。
- 停 agent 时清掉了 19 个超时的 python/pytest 残留进程。若发现构建整体变慢而非失败，先查有没有进程
  卡在编译锁上（ptrace 停住的 gdb、`futex_wait_queue` 里的 pytest 都出现过），症状是「所有人都变慢」。
