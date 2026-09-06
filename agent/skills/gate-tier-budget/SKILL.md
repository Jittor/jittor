---
name: gate-tier-budget
description: 把一套门禁拆成快慢两层的做法：怎么量每个文件的耗时、怎么选该推迟的文件、以及为什么预算必须用算术校验而不是断言墙钟。要给 PR 门禁提速、要加 xdist、或者要写「smoke < N 分钟」这类承诺时读这一篇。
---

# 门禁分层：量、选、校验

## 1. 量：`--durations=0`，并且记下机器当时的负载

```bash
pytest <门禁选择集> -q --durations=0 -p no:randomly > run.log
```

按**文件**聚合，不要按 nodeid——推迟清单里写 nodeid 会随参数化改名而静默失效，
写文件不会。解析脚本见 `_home/<分区>/runs/durations.py`（正则
`^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)`，按 `::` 前缀归并）。

**三个必须一起记下来的条件，少一个这份数就不能用**：

- `uptime` 的 load average。十几个 agent 并行的机器上，同一套测试可以差两倍。
- **缓存是冷是热。** Jittor 第一次跑要编 kernel。冷缓存量出来的是「编译时间的分布」，
  热缓存量出来的才是「测试时间的分布」，而 CI 会 restore 缓存，所以**热的那份才是承诺**。
  改过 `python/jittor/src/**` 或 rebase 过之后，第一轮一定是冷的。
- `use_parallel_op_compiler` 的值。它决定编译时间除以几。

### 1.1 用门禁自己的口径量，否则排名会整个错掉

**这一条比上面三条更容易漏，而且错得更狠。** 一个目录「跑一遍多久」和「它在门禁里
多久」可以差 50 倍，因为门禁的环境不是你手跑的环境：

|  | 手跑 `pytest tests/structure -q` | 门禁口径 |
| --- | --- | --- |
| 设备 | 这台机器有 CUDA，就用 CUDA | `nvcc_path=""`、`JITTOR_TEST_DEVICES=cpu`，CUDA 用例全 skip |
| 并行 | 串行，独占全部核 | `-n 4`，每 worker 分到 1/4 的核与线程 |
| 进程模式 | 你手上那个 | 由 `TORCH_MODE_PATHS` 决定，两条命令各管一半 |

实测（2026-09-06，`tests/structure`）：串行且开着 CUDA，
`test_process_mode_contract.py::test_naming_a_torch_path_alongside_a_native_one_does_not_change_its_meaning`
是 **269.4 s**、占该目录 44.9%；同一条在门禁口径下是 **4.8 s**。那份序贯数据把三个文件
排到前三，而它们在门禁里分别是第 20 名开外。**照它做分层会三个都选错。**

所以：量的那条命令要和门禁跑的那条命令一样。最省事的做法是直接用门禁的选择集函数
（`_helpers.gate_scope.native_arguments()` / `torch_arguments()`）加上层的筛选参数，
而不是自己写一个路径。

### 1.2 两轮之间不要 rebase：它会把缓存变冷

已经知道「不要在验证跑着的时候 rebase」。相邻的那个坑是**在 before 和 after 之间
rebase**：只要带进来的提交碰了 `python/jittor/src/**`，配置指纹就变了，after 那轮
从冷缓存开始，于是你量到的是编译时间的差，不是改动的差。

实测：一次 rebase 只带进 `backend.cc`/`backend.h` 共 +37 行，下一轮就重编了 900 多个
kernel，native 半边从热的 406 s 变成 1697 s（4.2 倍）。**先跑完两轮再 rebase**；
非要 rebase 就把 before 也在新 HEAD 上重跑一遍，并且两轮都要先跑一次把缓存捂热。

## 2. 选：默认包含，推迟的要写出代价和理由

清单放一个地方（`tests/_helpers/tiers.py`），一行一个
`(路径, 实测秒数, 理由)`。**默认是包含**——新写的测试自动进快门禁，慢的那个要来
清单里解释自己。反过来（维护一份「快门禁跑哪些」的白名单）一定会漂移，0.04 已经
用 98/332 证明过一次。

理由不能写「慢」。「因为要编两百个 kernel 而慢」和「因为 sleep 而慢」是两个不同的
决定，只有后者是 bug。

### 2.1 行数不是耗时，别拿行数当选择依据

「哪个目录最大」和「哪个目录最贵」经常不是同一个答案，而**行数是免费就能看到的那个**，
所以它会被当成耗时的代理用。实测反例（`tests/structure`，19774 行 107 文件）：

- 16 个 ACL 静态合同共 **3292 行**（占该目录 16.6% 的行），实测合计 **13.2 s**；
- 序贯量出来最贵的三个文件是 **116、68、127 行**。

原因是静态合同就是读文件加子串断言，几毫秒一条；贵的是**起子进程**和**编 kernel**，
而这两件事和文件多长没关系。

推论有两条。一，**按行数删测试不会让门禁变快**——「把这个目录压到 N 行」和
「把这一层压到 N 分钟」是两个独立目标，不要拿一个去论证另一个。二，**先量再选**：
`--durations=0` 一轮的成本远小于按行数猜错之后重做一次分层。

## 3. 校验：算术，不是墙钟

**不要写 `assert 这一轮耗时 < 300`。** 墙钟上界在有负载的机器上会红，而且红得和真
回归一模一样（这个仓库为这条已经付过三次学费，见 `load_sensitive` marker）。

改成用实测数做算术，在**结构测试**里断言：

```
makespan(一个模式) >= max(该模式非慢文件总秒数 / workers, 最长的单个非慢文件)
预测总时长 = Σ_模式 (makespan + startup)
assert 预测总时长 <= 预算
```

- `--dist loadfile` 切不开一个文件，所以「最长单文件」是硬下界。**只除以 workers 会
  给一个含九分钟单文件的层预测出三分钟。**
- 两个进程模式（native / torch shim）是**串行的两条 pytest 命令**，预算要买下两条。
- `startup`（解释器 + import jittor + collection）不随 workers 变小，要单列。

另外断言两条防退化的：**每一条推迟项都还指向一个门禁真的会跑的文件**（重命名会让它
静默失效），以及**快门禁仍然覆盖 ≥ N% 的文件**（防止「靠推迟整棵树来变快」）。

## 3.5 加 worker 之前先把线程数除开，否则会变慢

**实测，不是理论**：原生半边串行跑，逐用例耗时合计 1449 秒；同一批用例改成
`-n 4 --dist loadfile`、其它什么都不改，逐用例耗时合计变成 **6122 秒**，
单个文件最多涨 66 倍（`test_setitem.py` 17s → 1124s）。墙钟 1602 秒，
**比串行跑整棵树（1467 秒）还慢**，而它跑的用例更少。

原因：jittor 默认把 `OMP_NUM_THREADS` 设成「**它能用的**物理核数」——它读的是
affinity mask，`taskset -c 104-111` 下得到 8，这一步是对的。它不知道的是同一批
核上还有另外三个 worker 在做同样的事。于是 4 个 worker 各起 8 个 OpenMP 线程，
4 倍超订，而 OpenMP 屏障的代价随线程数增长、每线程的活却在变少。

**并行度只能在一处声明。** 加 `-n N` 的同时必须
`OMP_NUM_THREADS = max(1, len(os.sched_getaffinity(0)) // N)`。

**推广**：在问「加进程能不能更快」之前，先问「并行度是不是已经在别处用掉了」。
设备对拍那条电池组给出的是同一答案的另一面——每条用例在 CPU 与 GPU 两边各跑
一遍，CPU 那半已经吃满全部核，所以 4 个 worker 只快 6%。

## 3.6 快不是判据，结论可信才是

同一轮 4-worker 的设备对拍里，两个 worker 死了（`node down: Not properly
terminated`），一条用例被报成 FAILED 而原因不在它自己，会话以 xdist
`INTERNALERROR` 结束——**26 条里只拿到 23 个结论**。

**一个有时不给结论的验证器比一个慢的更糟。** 并行化门禁的验收标准因此是两条，
不是一条：（1）更快；（2）**两轮的 nodeid 与结论集合完全相同**。只看第一条会把
「跑崩了所以没跑」当成「跑完了而且更快」。

怎么机械地验第二条：`tools/gate_conclusion_diff.py record` / `compare`，见
`verifying-a-gate-actually-ran` 第九节。**别用眼睛比摘要行**——0.16 丢结论的那一轮，
数量看着也合理。那一节里还有一条同样重要：报「快了多少」之前先说清缓存是冷是热，
本行引用的 1405s「热缓存」数字复测不成立（同一批 nodeid 背靠背：冷 623s、热 25s）。

对全树并行还要加一条：**并行本身会制造新的红**。实测原生半边加 `-n 4` 之后多出
四条失败，都是并行下才成立的性质——进程级存活计数的绝对断言（`TestNode.test_lived`）、
跨进程构建锁的语义（`TestBuildLockIsShared`）。上并行之前要么修掉它们，要么标出来，
**不能让门禁自己制造红再教人忽略**。

## 4. xdist 用 `--dist loadfile`，不是 `--dist load`

跨文件状态泄漏是这棵树已知且已编目的性质，而且不少文件的用例依赖自己文件内的执行
顺序。按测试分发（`load`）会同时打破这两条，把一个有清单的问题换成一个查不动的问题。
按文件分发之后，每个 worker 看到的是「一部分文件、固定顺序」，和串行跑同形。

`--dist load` 只对**生成式**电池组正确（一条用例一个算子、彼此按构造独立），
那种情况单独一条 pytest 命令跑它。

## 5. 上了 xdist 之后，检查你的汇总还在不在

**这是最容易静默失效的一步。** xdist 下：

- `pytest_terminal_summary` 只在 controller 上跑；
- `pytest_collection_modifyitems` 和所有 fixture 只在 **worker** 上跑；
- `pytest_runtest_logreport` 两边都跑（worker 的 report 会回放到 controller）。

所以任何「在 fixture 里收集、在 terminal_summary 里打印」的统计，加上 `-n` 的那天
就变成空的，而且没有任何东西会说。实测踩到两个：跨用例状态泄漏调查（fixture 收集）
和每文件执行计数（collection 钩子填的集合）。

修法：worker 侧用 `pytest_sessionfinish` 自己落盘（文件名带 `config.workerinput["workerid"]`），
controller 侧从 `pytest_runtest_logreport` 重建它需要的集合。**改完要跑一次带 `-n` 的
回归，确认汇总段落还在打印。**

**这条坑咬过判据工具本身。** `tools/gate_conclusion_plugin.py` 用
`pytest_collection_modifyitems` 记 `collected`，于是每一次带 `-n` 的会话记下来的
`collected` 都是空的（串行 4、`-n 2` 是 0）。而 `compare` 里报「少给了一个结论」的
**两个分支都以 `collected` 为准**：`collected - conclusions` 恒为空，
「CONCLUSION LOST」那一支对每个 nodeid 都命中 `nodeid not in cand_collected` 而
`continue`。结果是丢多少结论都判成等价——实测四条用例、候选 deselect 掉一条，
`compare` 在同一段输出里先打 `passed 4 -> 3`、再打 `IDENTICAL`，退出码 0。

也就是说：**这个专门用来抓「丢结论」的工具，在最会丢结论的配置下抓不到任何东西**，
而 smoke 层恒定用 `-n 4` 跑。修法是在 controller 上实现 xdist 的
`pytest_xdist_node_collection_finished(ids)`（各 node 报的是自己 deselect 之后的 ids，
xdist 已校验各 node 一致，取并集），并且用 `optionalhook`，否则没装 xdist 时插件加载不了。
回归在 `tests/structure/test_gate_conclusion_record.py`。

推广一条：**判据工具也要有判据。** 一个「永远说 IDENTICAL」的差分器和一个不存在的
差分器等价，而前者更贵——它会让你相信一个没验过的结论。给它写一个「已知不同的两轮」
当反向用例，比给它写十个「相同的两轮」有用。

## 6. xfail 会被算成 skip

`report.skipped` 对 xfail 是 True（带 `wasxfail`）。如果你有「这个文件执行了几条」
的统计，xfail 会被算成没执行，而它的「skip 原因」是断言文本——可能碰巧命中环境
白名单被解释掉。实测：一条 xfail 的信息里出现了 `dataset` 字样，于是被当成
「这台机器没有 dataset」。判据要写成 `not report.skipped or hasattr(report, "wasxfail")`。
