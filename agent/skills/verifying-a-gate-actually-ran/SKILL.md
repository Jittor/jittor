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
