---
name: verifying-a-gate-actually-ran
description: 在相信一次绿色的测试运行之前，先证明它跑的是你的代码、而且真的实例化出了用例。用于改动 tests/_helpers、noxfile 门禁清单、设备参数化、或在 git worktree / 多环境里验证任何改动。
---

# 一次绿色运行要满足两个前提

"跑绿了" 只说明**被执行的那些断言**通过了。它不说明：

1. 被执行的是**你改的那份代码**；
2. 你以为的那些用例**真的被生成并执行了**。

这两条在本仓库都失败过，而且失败时的表象和成功完全一样。下面是每次验证前必须做的检查。

## 一、你测的是哪一棵树

开发环境里 `jittor` 是 editable 安装：site-packages 的 `.pth` 指向**主工作树**。所以在
worktree 里裸跑 `python -c "import jittor"` 导入的是主树，不是你的改动。

- `pytest` 是对的：`tests/conftest.py` 把本 checkout 的 `python/` 放在 sys.path 最前，
  在 worktree 里跑 pytest 天然导入本 worktree。
- **手写的 `python -c`、`python 脚本.py`、`subprocess`、`nox` 一律不对**，必须显式设
  `PYTHONPATH=<你的工作树>/python`。
- `nox` 会新建 venv，里面的 jittor 来源要单独确认，不能假设。

每次开始验证前先跑这一条，它必须打印你自己的工作树路径：

```bash
PYTHONPATH=$WORKTREE/python JITTOR_HOME=$MYCACHE TMPDIR=$MYTMP \
  python -c "import jittor, os; print(os.path.dirname(jittor.__file__))"
```

C++ 改动还要再确认一次缓存目录：Jittor 的 cache 路径里带工作树名，
`Create cache dir: .../<工作树名>/jit` 打印的名字不是你的，就说明这次编译的不是你的源码。

## 二、用例真的被实例化了吗

pytest 对"0 个用例"和"全部通过"的输出**没有区别**：两者都退出 0。本仓库因此丢过整层证据——
227 个算子的反向正确性（gradcheck）在三套门禁里都实例化为 0 个用例，同时三套门禁全绿。

机制（`tests/_helpers/device_types.py`）：

- `instantiate_device_type_tests` 把一个模板类展开成 `TestFooCPU` / `TestFooCUDA`；
- **作者的设备钉**（`only_for=` / 逐方法的 `@onlyCPU`）和**运行方的设备选择**
  （`JITTOR_TEST_DEVICES`）是两个不同的过滤器；
- 曾经它们被合成一个：CUDA 门禁设 `JITTOR_TEST_DEVICES=cuda`，于是只生成
  `TestGradientsCUDA`，而它的每个方法都带 `@onlyCPU` 被过滤掉 → 空类 → 收集 0 项 → 通过。

现在的规则：**设备钉写在 `instantiate_device_type_tests(...)` 调用上，不要写在方法上**；
生成出零方法的类、模板本身没有测试方法、`only_for` 或 `JITTOR_TEST_DEVICES` 写了未知设备名，
都直接 raise；被运行方的选择排除掉的整个电池组会留下一条**带原因的 skip**，而不是消失。

### 三条必做的检查

**1. 数一数收集到多少。** 不要只看 "passed"，先看收集数：

```bash
JITTOR_TEST_DEVICES=cpu python -m pytest <目标> --collect-only -q | tail -3
JITTOR_TEST_DEVICES=cpu python -m pytest <目标> --collect-only -q | grep -c '<用例名片段>'
```

改了参数化逻辑，就在改前改后各数一次，把两个数字写进提交说明。

**2. 数一数多少是 skip。** `N passed, M skipped` 里 M 很大就要逐条问为什么。恒 skip 的
门禁条目等于没有条目（本仓库有过一个条目常年 `1 skipped` 被当作通过）。

**3. 新增门禁条目要先手动跑一遍全量。** 加进 `noxfile.py` 的 `*_TESTS` 之前，先用与门禁
相同的环境变量（`JITTOR_TEST_DEVICES`、`nvcc_path`、`REAL_TORCH_SITE`）单独跑一次完整文件，
把 passed/skipped/failed 记下来。门禁里第一次跑出一片红，成本比现在高得多。

## 三、进程模式：native 还是 torch

`tests/_helpers/process_modes.py` 的 `TORCH_MODE_PATHS` 列出必须在 Torch shim 模式下跑的路径
（含 `tests/ops/test_ops.py`）。`tests/conftest.py` 的 `pytest_ignore_collect` 会在 native 会话里
**整体忽略**这些路径。所以：

- `pytest tests`（宽选择）= native 会话，这些路径根本不被收集；
- 只有把路径**显式**写在命令行上，才会进入 Torch 模式并真的运行。

这意味着「`pytest tests` 全绿」不覆盖这些文件。要覆盖全树必须跑两个会话
（`tools/run_test_suite.py` 就是干这个的：native 会话 `tests` 加 `--ignore=`，torch 会话只跑这些路径）。

## 四、多 worktree 并行时，`.git` 里哪些是共用的

同一个仓库的多个 `git worktree` **不隔离** `.git` 里的大部分东西。分不清就会出现
"我 pop 出来的是别人的改动"。

| 共用（全仓库一份） | 每个 worktree 独立 |
| --- | --- |
| **stash 栈**、refs/branches/tags、reflog、object 库、config、hooks | index、工作区文件、`HEAD`、当前分支、`MERGE_HEAD` 等运行状态 |

所以：

- **不要用 `git stash`。** 别人在另一个 worktree 里 `git stash pop`，弹出的可能是你的改动。
  真要临时搁置几个文件，用补丁：

  ```bash
  git diff <文件...> > $MYTMP/wip.patch && git checkout -- <文件...>
  # 恢复
  git apply $MYTMP/wip.patch
  ```

- 一次只做一个任务，改完就测、测完就提交，不要让无关的 WIP 留在树里。
- 分支名也是共用的：同一个分支不能在两个 worktree 同时 checkout。

## 五、并发与缓存

- 每个并行进程独立 `JITTOR_HOME` 与 `TMPDIR`。共享缓存的并发运行会互相损坏，表象是
  **在无关算子上大面积报梯度不符**，看起来像真实回归。
- 不要 `kill -9` 正在编译的 Jittor 进程；必须杀就先删掉它的 `JITTOR_HOME` 再重跑。
- 磁盘满的表象与并发损坏一模一样（散布失败加段错误）。跑之前先 `df -h`。

## 六、白名单式门禁必然腐化——这条有实测数字

一个「门禁跑哪些文件」的手写清单，和测试树之间没有任何东西把它们绑在一起。它不会报错，
不会警告，缩水的时候看起来和通过一模一样。**它只会单调地越来越小。**

本仓库的两次计数，相隔一份审计的时间：

| | 测试文件总数 | 任一 workflow 可达 | 一个门禁都碰不到 |
| --- | --- | --- | --- |
| 审计当时 | 289 | 74（25.6%） | 215 |
| 2026-09-03 实测 | 332 | 98（29.5%） | 234 |

树长了 43 个文件，**白名单一个都没跟上**。新写的测试默认是死的。

反过来做——「整棵 `tests/` 减一份写明理由的排除清单」——之后，实测结果是
**排除清单可以是空的**：332 个文件里没有任何一个是真的不该跑的。也就是说，那份白名单
不是在保护什么东西，它只是没人维护。这是判断一份清单该不该存在的判据：

> 把清单反过来写成排除项，逐条问「排它的理由是什么」。答不上来的，说明它本来就不该在清单里。
> 如果一条都答不上来，说明这份清单从一开始就只是历史。

### 反过来写之后，要一起加的三件事

清单没了，就得有别的东西保证它不会悄悄缩回去：

1. **可达率有下界的断言**（`tests/structure/test_gate_scope.py`）：可达文件数不得低于
   总数减 N；每条排除必须有非空理由，且被排除的路径今天还存在。
2. **门禁定义只有一份**：`nox -s cpu`、`tools/run_test_suite.py`、结构测试都从
   `tests/_helpers/gate_scope.py` 求选择集。三份实现必然漂移，而漂移是静默的。
3. **每条目至少执行 1 个非 skip 用例**（`JITTOR_TEST_REQUIRE_EXECUTION=1`）：
   把文件收进门禁，不等于它在这台机器上真的跑了什么。恒 skip 的条目必须写明
   「这台机器上为什么没东西可跑」。

### 代价要一起报出来

反过来写会让门禁变长——本仓库实测：CPU 全树原生一遍 **53 分钟**
（8 核，机器负载 19；改之前的白名单是 22 个文件）。所以改的同时要把 CI 的
`timeout-minutes` 一并调够，并且说明它由后续的 smoke/full 分层降回来。
**不写清楚这一点，下一个人会以为门禁坏了，然后把范围改回去。**

## 七、把静默失败改成报错时，先找谁在依赖那个静默

**一个靠「另一处的返回值恰好是这个」工作的条件，会在那一处被修好时坏掉。**

实测的一次：`tools/run_test_suite.py` 的预热重试是为「冷缓存重建了 jit_utils，本进程无法
重载，请重跑」准备的，判据写成「没有 marker 且退出 0」。它一直**靠巧合工作**——那条路径
当时 `sys.exit(0)`。0.11 把它改成非零退出（为了让 CI 看得见），**同一个改动让这个重试在
第一次尝试就放弃**，一整轮全树跑因此零用例退出。

这和"新的报错照亮旧的错误"不是一回事，更隐蔽一层：**旧代码依赖的正是那个错误的行为**。

### 改之前做这一步

把静默失败改成报错（非零退出码、抛异常、断言）之前，grep 一遍谁在读那个信号：

```bash
# 退出码：谁在判 returncode / $? / check=
grep -rn "returncode\s*==\s*0\|returncode\s*!=\s*0\|check=True" --include="*.py" .
# 标志性输出：谁在按字符串判断成功
grep -rn "in completed.stdout\|in output\|in result.stdout" --include="*.py" tests/ tools/
```

逐个问：**它是在判"成功"，还是在判"那个特定的失败"？** 后者会随你的改动坏掉，
而且坏得很安静——重试变成不重试、循环变成直接返回、"没有输出"被当成"没有问题"。

修法是让条件**指名道姓**：认 `JIT_UTILS_UPDATED_EXIT_CODE` 这个常量本身，
而不是依赖它碰巧等于 0。

## 八、把散落的调用收编进 helper 时，调用点原来做的事是契约

收编（把 N 处内联的 `subprocess.run` 改成一个 helper）会系统性地丢掉一类东西：**每处调用
自己做的那点准备工作**。它看起来像样板，实际上是那处调用的契约。

实测的一次：一个探针在子进程里先塞一个假的 `torch` 模块再 `import jittor`，所以它必须
先 `env.pop` 掉四个 `JITTOR_TORCH_*` / `REAL_TORCH_SITE`。收编之后 helper 的 `env=`
是**叠加**在 `os.environ` 上的——**叠加删不掉东西**，四个变量原样回来，子进程报
"cannot install Jittor Torch compatibility over an existing Torch module graph"，
而这个用例的名字是"检查 import 环路"。

### 判据：把静默的损失变成第一次调用就响的错误

修一遍已知实例是不够的，因为**人工扫一遍只能找到你想得起来的那些**。真正的修法是让这类
调用**报错**：

```python
if inherit and extra and "PATH" in extra:
    raise AssertionError(
        "child_env() 拿到的像是一个完整环境（里面有 PATH）而 inherit=True。"
        "叠加到 os.environ 上删不掉任何变量，你做过的 env.pop 会被静默撤销。"
        "完整环境请传 inherit=False，或者只传你要改的那几个变量。")
```

判据很朴素：**"字典里有 PATH" ⇒ 调用方给的是完整环境** ⇒ 它多半删过东西 ⇒ 叠加语义会
悄悄撤销。加上这条断言的那一刻，同一棵树里又冒出四处同样的问题——其中一处 pop 的正是它
要测的那个变量（`test_openmp_threads` pop `OMP_NUM_THREADS`，然后断言子进程里它被自动
设成物理核数）。**这证明它不是一处疏忽，是收编动作系统性会丢的东西。**

同一个形状的第二个例子：子进程可能被信号杀死时必须隔离（`crash_isolated=True`），
否则 jittor 的 SIGCHLD 处理器会让 pytest 无声消失。也是"把契约写进 helper 的一个显式
选项，而不是指望每个调用方记得"。

### 收编前的检查清单

逐处 diff 内联版本，列出它做了但 helper 不会做的事，每一项要么进 helper 的显式选项，
要么在调用点留一行注释说明为什么不需要：

- `env.pop(...)` / `del env[...]`——**叠加语义删不掉东西**，这是最隐蔽的一类；
- `cwd`——决定 rootdir，进而决定哪份 conftest 生效；
- 超时——按什么假设定的？冷缓存下还成立吗？
- 编码（`encoding` / `errors`）——Jittor 的日志不是 ASCII，`LANG=C` 下按环境编码解会抛；
- 信号处理 / 崩溃隔离——子进程会不会被信号杀死；
- **打桩点**。收编重复代码时最容易悄悄失效的就是测试里的 `mock.patch.object(module.subprocess, "run")`：
  桩还在、桩住的东西已经不在调用路径上了，**测试照样绿**。改完 helper 之后 grep 一遍
  `mock.patch.*subprocess`、`mock.patch.*Popen`，确认每个桩仍然在新的调用路径上，
  并确认它要观察的行为没变。

做不到就不要收编——一份能跑但语义被削掉的 helper，比五处重复代码贵得多。

## 九、给门禁提速：结论集合是判据，快不是

一次门禁优化的失败形态**不是变慢，是少给一个结论**，而少给一个结论不会改变
「N passed」那一行的可信度。0.16 实测过一次：设备对拍加 `-n 4`，快 6%，
**26 条里丢了 3 个结论**，退出码是绿的、摘要行看起来正常。

> **一个有时不给结论的验证器比一个慢的更糟。**

所以口径是两条，不是一条：（1）更快；（2）**两轮对每一个 nodeid 给出同一个结论**。
「数量相同」不等于「集合相同」——0.16 那次丢结论时数量看着也合理。

### 怎么做：记下来，再逐条比

`tools/gate_conclusion_diff.py`（配 `tools/gate_conclusion_plugin.py`）就是这件事：

```bash
# 1. 改之前
python tools/gate_conclusion_diff.py record --out $RUNS/base.json --label before -- \
    tests/backends/parity/test_device_parity.py -q
# 2. 改之后（或者换配置：--env 会连同环境一起记进产物）
python tools/gate_conclusion_diff.py record --out $RUNS/cand.json --label after \
    --env JITTOR_REFERENCE_CACHE=1 -- \
    tests/backends/parity/test_device_parity.py -q
# 3. 判据：有任何一条不同就非零退出
python tools/gate_conclusion_diff.py compare $RUNS/base.json $RUNS/cand.json
```

产物里**分开记两件事**，差集就是判据：

- `collected`：这次会话决定要跑的 nodeid；
- `conclusions`：真的报出了结论的 nodeid（含 `passed/failed/error/skipped/xfailed`
  与 skip 原因）。

**两者之差正是丢结论的形状**：worker 死掉、崩溃带走会话、分发模式漏掉一条，都会留下
一个「收集了但没有结论」的 nodeid，而没有一个会让退出码显眼地变。

`record` **不要求跑绿**：结论是「这六条红」的基线一样能用，否则判据恰好在最需要它的
时候不可用。只有 `compare` 判成败。

三条容易漏的：

- **skip 原因变了也算结论变了。** 「从 passed 变成 skipped」和「skip 的理由换了一个」
  都是靠扩大排除清单假达标的形状（0.15 的红线）。
- **墙钟只报告，不做判据。** 有负载的机器上墙钟能差两倍（见 `gate-tier-budget`）。
- **两轮要选同一批测试**，变的是配置。选择集不同时 `compare` 会把两边的
  `pytest_arguments` 一起打出来，别把它当成通过。

### 报「快了多少」之前，先说清缓存是冷是热

这条让 0.16 的归因整个反了。它记录「热缓存 1405s ≈ 冷缓存串行 1444s，**所以这条电池组
不是编译瓶颈**」，0.22 的三个方向都是从这句推出来的。**2026-09-05 复测（同一批 26 个
nodeid、同一个 `JITTOR_HOME`、背靠背两轮）：冷 623s，热 25s——25 倍。它就是编译瓶颈。**

所以任何「快了 N%」都必须写明这一轮是冷是热，两轮之间对缓存做了什么。否则下一个人会
拿着一个不成立的归因去选方案。

**怎么造一个「算子冷、核心热」的对照组**（不用重编 10 分钟的 C++ 核心）：

```bash
CACHE=$(python -c "import jittor;print(jittor.compiler.cache_path)")   # 带 PYTHONPATH
mv "$CACHE/jit" "$CACHE/jit.aside" && mkdir -p "$CACHE/jit"   # 只丢算子 kernel
```

`jit/` 之外的东西（`jittor_core*.so`、`obj_files/`、`gen/`）都留着。**mv 而不是 rm**：
恢复是免费的，而且量完了还能对照。另外，Torch shim 模式与原生模式的 `cache_path`
**不是同一个目录**（0.07 的配置指纹里有 shim 数学开关），所以「我刚跑过一遍所以是热的」
在换了模式之后不成立——这也是 0.16 那个 1405s 最可能的来源。

### 如果你的提速手段是「缓存期望值」

缓存 oracle 是唯一一种能让门禁**在绿着的时候说谎**的优化：它回答的是产出那条缓存的
代码，不是被测的代码。要做就把三件事一起做（`tests/_helpers/reference_cache.py` 是
实现，`tests/backends/parity/test_reference_cache.py` 是它必须满足的性质）：

1. **键里带实现的内容哈希**（`python/jittor/**` 的 `.py`/`.cc`/`.cu`/`.h`）。用内容不用
   mtime：checkout 和 rebase 会重写 mtime 而不改任何数字。代价是任何一次源码改动都
   全部 miss——这是故意的。
2. **键里带物化之后的输入字节**，不是产生输入的种子。样本生成逻辑一改就 miss，
   而不是拿旧答案去比新输入。
3. **每条缓存复述自己的键**，读的时候不匹配就当没有；**写用临时文件加 `os.replace`**；
   **命中数要在摘要里打出来**（本仓库放在 `pytest_terminal_summary`）。一个没人打印的
   数字就是一个没人检查的数字。

还有一条不写在代码里的：`0` 维数组会被 `np.ascontiguousarray` 悄悄变成形状 `(1,)`。
全归约的 oracle 就是 0 维，于是它会被存成一元向量、读回来形状是错的，而与 0 维结果
比较时**广播掉了**——照样通过。存之前按 `ndim == 0` 分支处理。
