# 测试体系

- 状态：已接受
- 上次复查：2026-09-11
- 基线：`2d716db31`
- Owner：测试基础设施维护者
- 复查触发：收集根目录、进程模式归属、marker、OpInfo 契约或后端门禁发生变化时

Jittor 用 pytest 作为仓库的测试运行器，同时保留兼容的 `unittest.TestCase` 测试。
套件位于顶层 `tests/`，**不属于已安装的包**。
[`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml)
是收集与 marker 配置的权威来源；
[`noxfile.py`](https://github.com/Jittor/jittor/blob/master/noxfile.py)
是可复现的命令面。

## 目标

- 用**独立参考**比对前向行为；
- 数值化验证解析梯度，并跨设备验证；
- 让结构、CPU 与硬件要求保持显式；
- 保留有价值的历史边界用例，同时把跨测试导入和临时发现替换为共享 helper 与注册表；
- 通过**具体的 skip 或严格的预期失败**让不支持的行为可见；
- 让收集阶段不产生编译、下载和硬件副作用。

## 目录布局

```text
tests/
├── _fixtures/            # 版本化的测试数据
├── _helpers/             # 显式的共享测试工具
├── opinfo/               # 算子元数据、样本、参考、skip 策略
├── ops/                  # 通用 OpInfo 前向与梯度批测
├── backends/             # CPU/CUDA/ROCm/NPU 与跨设备一致性
├── compiler/             # JIT/编译器行为与 kernel 陷阱
├── core/                 # 张量、dtype、图与自动微分契约
├── nn/                   # 神经网络算子与模块
├── optim/                # 优化器契约
├── distributed/          # MPI 与分布式行为
├── compat/               # 兼容接口与导入行为
├── integration/          # notebook 与跨组件工作流
├── structure/            # 仓库、打包与静态契约
├── models/               # 维护中的模型级测试
└── system/               # 进程/环境集成测试
```

测试模块可以经由为套件配置的 pytest Python 路径导入 `tests/_helpers` 和 `tests/opinfo`。
**不得把另一个测试模块当作隐式的 helper API 导入。** 共享工具若含有非平凡的比较或设备
逻辑，需要有自己的定向测试。

## 进程模式隔离

Torch 兼容的安装是**进程级**的，会改变公开方法、dtype 提升、归约默认值和惰性执行。
因此**原生测试与面向 Torch 的测试跑在不同的 pytest 进程里**。

`tests/_helpers/process_modes.py` 拥有 `TORCH_MODE_PATHS` 列表；`tests/conftest.py`
在广泛的原生收集中忽略这些路径，并在显式选中其中之一时激活 Torch 模式。共享的 OpInfo
与设备一致性套件使用面向 Torch 的签名，因此属于 Torch 进程。

`tools/run_test_suite.py` 是**完整套件的入口**：它以各自独立的状态、缓存和模式变量运行
原生与 Torch 两个会话，再汇总结果。直接执行 `python -m pytest tests` **有意只是原生
会话**，不得作为全套件覆盖来报告。

一个断言 `result_type`、Torch cast 别名、带类型的张量名或 Torch 特有默认值的测试属于
Torch 会话——即使该文件同时携带底层契约并且位于 `tests/core/` 下。

进程内的独立 PyTorch oracle 要求 `REAL_TORCH_SITE` 指向一个含 PyTorch 二进制 `_C`
扩展的 site-packages 目录，pytest 会在 Jittor 之前预加载该实现。**没有这个显式 oracle
时，测试会跳过可选的 PyTorch 对比**，即使能发现一个已部署的、由 Jittor 支撑的 `torch`
stub 也一样——**该 stub 绝不能被当作独立参考**。

## 算子证据的三层

### 1. 独立前向参考

每个 `OpInfo` 描述可调用对象、样本构造器、dtype、自动微分支持、容差，以及定向的
skip/xfail 策略。`tests/ops/test_ops.py` 把这个数据库在所请求的设备上展开，并与独立的
NumPy 或数学参考比对。

**参考实现不得调用被测的那个 Jittor 运算。** 样本构造器要覆盖有意义的形状、轴、广播、
可选参数和容易出错的 dtype，而不是**在没有语义多样性的前提下堆数量**。

### 2. 数值梯度

可导的 OpInfo 条目跑 float64 的 CPU `gradcheck`；支持二阶微分的运算还跑 `gradgradcheck`。
标注 `supports_autograd=False` 或 `supports_gradgrad=False` 是**契约决定，不是把失败的
测试弄绿的手段**，必须在定义中或问题总账里给出理由。

数值检查在 CPU 上证明导数公式。**它不证明加速器正确执行了同一个反向 kernel。**

### 3. 设备一致性

[`tests/backends/parity/test_device_parity.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/parity/test_device_parity.py)
在 CPU 和可用的加速器上跑相同的输入与余切，用全局与逐元素两种误差度量比较前向输出和
梯度。这一层捕获设备特有的编译失败、丢失的梯度贡献，以及静默的 kernel 偏差。

> **CPU 只有在被前向层与数值梯度层独立钉死之后，才是有效的一致性参考。** 后端环境
> 失败要与框架缺陷分开报告。

## 覆盖面：三层之外的那个问题

上面三层回答的是**「这个算子对不对」**。它们不回答**「我们到底问了哪些算子、
哪些 dtype」**——而缺陷正是从没被问到的地方冒出来的。

四个已经证实的形态：

**算子面远大于数据库。** 公开可调用面有 1294 个名字，OpInfo 覆盖 179 个算子。
差集不是"还没来得及"，它是缺陷的住所。

**"被覆盖"不是二值的。** `bitwise_not` 对 bool 恒返回 `True`（与输入无关），
全程有 OpInfo 条目。条目声明 `dtypes=_INT`（`integral_types()` 不含 bool），
样本生成器又写着 `dt = _int_dtype(dtype)` 把传入 dtype 强制转成整型——于是没有
任何测试**能**喂给它 bool。算子读起来是被覆盖的。另注意 `OpInfo.dtypes` 缺省为
`floating_types()`：忘记声明的条目静默只测浮点。

**按文本统计覆盖率是自欺。** 1294 个名字每一个都在 `tests/` 下某处出现过
（import、注释、无关标识符），那个口径给出约 96%。

**测量只盖住了其中一个面。** 上面这 1294 个名字全是原生面。Torch 兼容面是另一张
对象图——`torch` 是兼容 owner 的命名空间视图，`torch.Tensor` 不是 `jt.Var`，
`torch.nn` 是组合出来的——原生清单一个名字都覆盖不到它。它自己有 1295 个公开
可调用名字，与原生面一样大，而在这套体系建成之前完全不在测量范围内。

对应的四件工具（清单和基线按进程模式各一份）：

| 工具 | 回答 |
| --- | --- |
| [`tests/structure/public_api_manifest.json`](https://github.com/Jittor/jittor/blob/master/tests/structure/public_api_manifest.json) 与其门禁 | 原生的 1294 个名字**还在不在** |
| [`tests/structure/torch_api_manifest.json`](https://github.com/Jittor/jittor/blob/master/tests/structure/torch_api_manifest.json) 与其门禁 | Torch 面的 1295 个名字**还在不在** |
| [`tests/_helpers/api_coverage.py`](https://github.com/Jittor/jittor/blob/master/tests/_helpers/api_coverage.py)（`JITTOR_API_COVERAGE=1`） | 一次运行**真的调用**了哪些入口 |
| [`tools/api_coverage_ratchet.py`](https://github.com/Jittor/jittor/blob/master/tools/api_coverage_ratchet.py) 与两份基线 | 没被调用的那个集合**有没有变大** |
| [`tools/opinfo_dtype_gaps.py`](https://github.com/Jittor/jittor/blob/master/tools/opinfo_dtype_gaps.py) | 哪些 (算子, dtype) **跑得通却没声明** |

覆盖测量默认关闭：包装每个公开入口是诊断，不是常态。它用开/关对照验证不改变被测
系统——原生 `tests/ops/test_where_op.py` 开关两侧同为 18 passed，Torch 会话在排除
下述 14 个文件后开关两侧的失败 nodeid 集合逐条相同。

**这条对照必须在每一个面上各做一次，在一个面上通过不能外推。** 同一行「包装所有
callable」在原生面一直无害，因为原生面没有可调用的 module；到 Torch 面上它替换掉
`torch.random`，前端的命名空间归属检查随即判定图已被改动，收集期直接报错。诊断在
一张对象图上验证过，就只是在那一张对象图上验证过。

Torch 面还有一条这套包装绕不过去的界限，它被记录而不是被藏起来：前端对「发布出去的
就是 owner 模块持有的同一个对象」有显式契约，fidelity 注册表又以对象本身为键，因此
替换对象的包装器对这些用例是可见的——`JITTOR_API_COVERAGE=1` 会让其中 86 条转红。
试过两种躲避方式，都更差（改绑定义模块 101 条，改绑全部别名 97 条且失败点移到注册表），
这说明这是「以替换实现的记录器」的性质而不是待修的 bug。名单、测量与出路记在
`tests/_helpers/api_coverage.py` 的 `IDENTITY_CONTRACT_FILES`：Torch 覆盖运行排除这 14 个
文件，排除项写进基线文件本身，而真正的解法是一个不替换任何对象的记录器。

一次运行只测量它所属进程模式拥有的那一个面：`JITTOR_TORCH_SHIM` 决定进程模式，
也就决定读哪一份清单、写哪一份基线；更新一个面的基线不动另一份。包装在**收集
结束之后**安装而不是在 session 开始时：拿到一个面意味着 import 它，而在 Torch
模式下这个 import 本身就是若干测试所测量的前端激活——早一步做，`enable()` 到测试
执行时已成空操作，诊断就改变了它的观测对象。

无法包装的入口单独记账而不是丢弃——分母里少算一个会美化结果。类是入口，但包装它
会替换类型并破坏 `isinstance`；module 是 owner 而不是入口，而 Torch 面有可调用的
module（`torch.random` 是带 `__call__` 的 module 子类），包装它会替换已发布的命名
空间绑定。两者都进 unwrappable，而不是从分母里消失。

棘轮的比较需要一次完整覆盖运行，属于 nightly 级作业而不是 PR 门禁；PR 门禁上跑的
只是基线自身的形态检查和规则的负向验证。dtype 探测按 (算子, dtype) 逐个试编译，
同样属于 nightly 级诊断。

## 跨切面契约

有些语义不属于任何单个算子，因而不属于上面任何一层：驻留与迁移、别名与就地写、
惰性与实体化时机。它们没有归属就没有测试。

[`tests/core/test_var_residency_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/core/test_var_residency_contract.py)
是这一类的第一个：未计算的 Var 没有驻留；读取 device Var 的数据会把存储真的迁回
主机；`device_id` 跨 `cpu()` 保留源设备而 `device`/`location()` 跟随数据。它整体
在 CPU 上运行，CUDA 用例逐条 skip——**去掉加速器只会收窄它而不是清空它**。

这一点是有来历的：Var/Module 的设备方法契约此前只有四条用例，且全部要求两张
CUDA 卡，于是在单卡与 CPU 机器上整体跳过，报告里和四条通过长得一模一样。现在
"设备数量不够"由 `insufficient-devices` 单独计数并在汇总里点名，与"没有加速器"
区分开——前者是硬件在场却仍丢掉的覆盖，不是环境事实。

## 测试分类与 marker

Marker 在 [`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml)
中注册：

| Marker | 契约 |
| --- | --- |
| `structure` | 不执行设备代码；布局、打包与静态检查 |
| `cpu` | 维护中的 CPU 行为 |
| `cuda` | 需要 NVIDIA CUDA 环境 |
| `rocm` | 需要 AMD ROCm 环境 |
| `npu` | 需要昇腾 CANN 环境 |
| `mpi` | 需要 MPI 启动器或多进程 |
| `slow` | 排除在快速 PR 门禁之外 |
| `network` | 需要外部网络访问 |
| `manual` | 显式选择；**绝不进入自动的默认运行** |

**使用适用范围最窄的 marker。** 硬件测试要探测真实运算，**不得在 CPU 上静默通过**。
network 与 manual 测试要在模块 docstring 里说明其外部要求。

## Skip 与已知失败

- **skip** 表示前置条件不可用，或某个契约有意不被支持；它的 reason 要指明**确切的
  前置条件或限制**。
- **预期失败** 表示一个已复现的框架缺陷。pytest 使用严格 xfail，因此修复会产生 XPASS
  并**强制清理总账**。
- **不要**在测试体外面捕获任意异常再转成 skip。要在执行前**窄范围地**探测可选环境。
- 每个持久的预期失败都要列进
  [问题总账](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)，
  带 owner 和退出条件。

## 门禁诚实度：三种「看起来通过了」

一次运行可以在完全不失败的情况下什么都没证明。三种形态都实际发生过，
各自的守卫也都因此存在：

**只会 skip 的条目和通过长得一模一样。** 227 个算子的反向公式就是这样在三次全绿里
保持未验证。`JITTOR_TEST_REQUIRE_EXECUTION=1` 要求一个条目要么执行了什么，
要么带理由列进 `gate_scope.EXECUTES_NOTHING`。

**「硬件不够」不是「没有硬件」。** 一个要两张卡的测试在单卡机器上跳过时，理由里
同样写着 CUDA，于是落进 `accelerator` 桶、读起来像一句环境事实——而这台机器有
加速器，覆盖是白丢的。Var/Module 的**全部**设备方法契约正好坐在这个跳过后面：
四条用例在非多卡机器上从不执行，报告里和四条通过无法区分。现在
`insufficient-devices` 单独计数并在汇总里点名。

**中途死掉的会话产出一份「结束了」的日志。** 原生崩溃发生在测试里时不会让那条
测试失败——它带走整个解释器：pytest 打印完即将执行的 nodeid，进程就没了，
没有结果行、没有 traceback、没有汇总，后面的测试从未运行。日志里没有任何一处
写着 failed，扫描失败的人找不到，grep 失败的脚本也同意。这比 skip 更糟——
skip 至少在汇总里占一行。

维护的 CPU 门禁正是这样只跑了自己的一部分：CPU-only 构建
（`nvcc_path=""`，即 `tools/run_test_suite.py` 所设）上 `scatter_add` 段错误
让 Torch 会话在 48% 处消失，而在有 CUDA 的机器上同一会话跑得完——所以它一直
没被发现。

[`tests/_helpers/session_completion.py`](https://github.com/Jittor/jittor/blob/master/tests/_helpers/session_completion.py)
让会话自述完成：`pytest_sessionfinish` 在最后一条测试之后运行，无论通过、失败还是
被 pytest 中断，**唯独进程死亡时不运行**——这正是想要的区分。哨兵同时带上
collected/executed 计数，于是「完成了但悄悄收集得更少」也可见。
[`tools/check_session_completed.py`](https://github.com/Jittor/jittor/blob/master/tools/check_session_completed.py)
是消费者，证据优先级为哨兵 > 日志标记 > pytest 汇总行。**缺哨兵不当作「大概没事」**：
这个失败模式看起来本来就像什么都没发生。

一条共同的纪律：**新增的门禁必须被证明会红**。写完把它要防的缺陷造出来，
确认变红，再恢复。没有做过负向验证的守卫，和不存在的守卫在报告里是同一个样子——
本轮就修好过两个从未检查过任何东西的既存守卫（一个扫描布局迁移后已删除的目录，
一个读陈旧路径而从未触及运行时行为）。

## 四条被事故教会的纪律

下面四条都不是从原则推出来的，是某一次「绿着的报告什么也没证明」之后补上的。每条
注明是被哪件事教会的，问题总账
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)
里有完整记录。

### 1. 子进程一律走 `_helpers.child_process`，并且要指定设备、自证落点

`tests/conftest.py` 把**被测的这棵 checkout** 放在自己的 `sys.path` 上，但**不导出
`PYTHONPATH`**。所以测试里一个裸的 `subprocess.run([sys.executable, ...])` 交给子进程
的是解释器环境解析到的那个 `jittor`——在开发 checkout 里，往往是指向**另一棵树**的
可编辑安装。

**安静的失败最贵**：导入了别的树的子进程照样能跑，测试照样绿，而它对被测代码什么都
没证明。吵闹的那一版在改名时才露出来：`[0.08]` 把 core 的 `set_lock_path` 改成
`set_lock_fd`，子进程加载的是这个分支刚构建出来的 core、导入的却是主树的
`compiler.py`（旧名字），死在一个哪棵树都对不上的 `AttributeError` 上。

规则是机械的而不是启发式的：
[`tests/structure/test_child_process_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/structure/test_child_process_contract.py)
用 AST 扫描 `tests/` 下每一处启动，**任何直接命名解释器的地方都让它红**。
`tools/run_test_suite.py` 和 `tools/gate_conclusion_diff.py` 也在扫描范围内，因为它们
启动同样的子进程。故意要**去掉** `PYTHONPATH` 的场景（"一个刚装完的人 `import jittor`
会解析到哪里"）也走 helper，写成 `child_env(..., repo_paths=False)`——一个读起来像
疏忽的例外无法和疏忽区分开。

**光传对树还不够，设备也要传，而且要自证。** `tools/roundtrip_consistency_sweep.py`
的子进程一直没设 `use_cuda`，靠默认值——而**全新进程里 `use_cuda` 默认是 0，即便机器上
有八张卡**。于是 `--device cuda` 那一档比的是「CUDA 父进程」对「CPU 子进程」，把两个
设备的差异当成了序列化缺陷。它一直报 OK，是因为在 `-Ofast` 下这个模型在两个设备上
碰巧逐位相同；换成 `-O3` 之后 CPU 的结果动了 `1.49e-08`，运气用完了，缺陷才露出来。

现在设备作为参数传给子进程，子进程用 `x.location()` 断言自己确实落在被要求的那一端：

```python
where = x.location()
expected = "device" if device == "cuda" else "cpu"
assert where == expected, "child asked for %s, tensor is on %s" % (device, where)
```

**传标志容易写对，也容易被静默忽略**——回退、缺驱动、标志读得太晚——一旦被忽略，这个
检查就又变回在比两个设备而看上去像在比两个进程。

（教会它的：子进程契约门禁长期红着并列着四个违规者，提交 `51cdefae2`；往返扫描的设备
缺陷，提交 `fef56a211`。）

### 2. 会崩溃的用例跑在自己的子进程里

原生崩溃发生在测试里时**不会让那条测试失败**——它带走整个解释器：pytest 打印完即将
执行的 nodeid，进程就没了，没有结果行、没有 traceback、没有汇总，**后面的测试从未
运行**。日志里没有任何一处写着 failed。

所以一个**已知会以信号死亡**的用例必须跑在自己的进程里，并且用
`run_child_script(..., crash_isolated=True)` 声明这一点——没有它，信号致死会以一个裸
的负返回码到达 pytest。两个现成的样板：

- [`tests/backends/cuda/test_auto_flush_graph_split.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/cuda/test_auto_flush_graph_split.py)：
  `auto_flush_ops` 的每个取值跑在自己的进程里，因为失败形态是段错误。它断言 loss
  逐位相同**并且**比较梯度——一个「止住崩溃但反向读了错字节」的修法能过「跑起来没有」
  的检查，在这里过不去。
- [`tests/ops/test_index_bounds.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_index_bounds.py)：
  设备端 trap 会带走 CUDA context，所以它不能和后面的任何东西共用进程。

（教会它的：`KI-EXEC-001`，五个 bottleneck 块在 `auto_flush_ops` 的发布默认值 128 上
确定性段错误；以及维护中的 CPU 门禁——CPU-only 构建上 `scatter_add` 段错误让 Torch
会话在 **48% 处消失**，而在有 CUDA 的机器上同一会话跑得完，所以它一直没被发现。）

### 3. 会读「files this session proved nothing about」这份报告

每次运行结束时，`tests/_helpers/pytest_policy.py` 会打印一段：

```text
========== files this session proved nothing about ==========
tests/.../test_x.py  collected 0 tests
tests/.../test_y.py  12 skipped, 0 executed
Reported only. Set JITTOR_TEST_REQUIRE_EXECUTION=1 (the gates do) to make an
unexplained entry fail the run.
```

两类条目，含义不同：

- **`collected 0 tests`**：这个文件连用例都没收集到。通常是收集期异常或整文件被过滤。
- **`N skipped, 0 executed`**：收集到了，一条也没真的执行。

条目后面可能带两种注记：`-- expected here: <理由>` 表示它列在
`tests/_helpers/gate_scope.py` 的 `EXECUTES_NOTHING` 豁免表里；`-- explained: <skip 理由>`
表示 skip 理由被识别为环境事实（没有加速器、缺可选依赖）。**没有注记的条目就是这份
报告真正要你看的东西**，门禁开着 `JITTOR_TEST_REQUIRE_EXECUTION=1` 时它会让整轮变红。

为什么单独写一节：这套机制**存在而且工作正常**，失效的是「有人读它」。
`tests/structure/backends/acl/test_acl_python_registration.py` 收集了 40 条、执行了
**0** 条，整整两天；报告每一轮都照实打印了那一行，而一片红的套件里多一行红是看不见的。
所以——**看到套件是红的，先看这一段，再看失败列表**：失败的用例至少还在说话。

（教会它的：`KI-TEST-004`。同一天用同一个问题——「如果它要检查的东西坏了，这个门禁会
红吗？」——问出了另外两个：`KI-EXEC-002` 的剖析器空报告，和往返扫描的设备缺陷。）

### 4. 跨越会改变显存驻留的 flag 做数值对照，先关掉 cuDNN 自动调优

cuDNN 的卷积算法是**实测候选者**选出来的，结果按形状缓存。实测时显存里有什么，决定
哪个算法赢；而 `auto_flush_ops` 这类调度 flag 改变的正是驻留。于是**跨着这类 flag 比较
梯度，比的是两件事**：被测的改动，和被换掉的卷积算法（实测值 `3.8e-4` 相对差，
确定性，而且前向 loss 一位不差）。

所以这种对照的第一行是：

```python
import jittor as jt
jt.cudnn.set_benchmark(0)     # 强制走确定性的启发式
```

关掉之后，`KI-EXEC-001` 跨设置的梯度残差是 `2e-6`——不同批边界带来的重结合。

（教会它的：`KI-EXEC-001` 的第一次修复**被错误地否决**了。理由是它「把崩溃变成了个别
元素上 40–60% 的梯度错误」，而两半都是审阅者的错：对照跑在自动调优**开着**的情况下，
而自动调优依赖驻留、驻留正是被测 flag 改变的东西；「40–60%」则是拿最大绝对差除以梯度
的 **RMS** 而不是除以它所属那个元素的大小。关掉调优重做，残差是 `2e-6`。教训不是
「多测几次」，而是**跨越改变驻留的 flag 的对照必须先把调优器按住，相对误差要除以它相对
的那个元素**。详见 `KI-EXEC-003` 与[数值契约](../notes/numerics-contract.md)。）

## 三层：core / smoke / full

一次全量跑完是十几分钟起步,拿它当「我刚才改的东西有没有坏」的答案太贵,于是有三层,
回答的是三个不同的问题:

| 层 | 问题 | 选择方式 | 实测 |
| --- | --- | --- | --- |
| `core` | 改完一处,有没有立刻坏掉 | `tiers.CORE_FILES` 的**白名单**,每个基本面一个文件 | 44 s(双进程模式,串行) |
| `smoke` | 一个 PR 该等多久 | 整棵树**减去** `tiers.SLOW_FILES` | 约 390 s(4 worker) |
| `full` | 这棵树到底是什么状态 | 整棵树 | 十几分钟起 |

只有 `full` 是关于这棵树的陈述,另外两层是关于时间的陈述 —— 这也是为什么 `core`
用白名单而 `smoke` 用黑名单:`smoke` 漏掉一个文件是**覆盖的洞**,所以默认必须是包含;
`core` 漏掉一个文件只是**发现得晚一点**,后面两层照样会红,所以可以是选择。

`core` 刻意串行:它小到 worker 买不到什么,而且这样在没装 pytest-xdist 的检出上照样能跑。
它的预算由 `tests/structure/test_gate_tiers.py` **按算术**校验(把每个文件的实测秒数加起来
和预算比),不是去断言墙上时钟 —— 后者在忙机器上失败的样子和真回归一模一样。

```bash
python tools/run_test_suite.py --tier core    # 改完就跑这个
python tools/run_test_suite.py --tier smoke   # 提 PR 前
python tools/run_test_suite.py                # 全量,按需
```

往 `core` 里加文件,就是在花所有人的每一次编辑的时间;条目里要写清**实测秒数**和
**它守的是哪个基本面**,两样都由结构测试检查。

## 命令

```bash
# 三层,以及仅原生收集、单个模块
python tools/run_test_suite.py --tier core
python tools/run_test_suite.py
python -m pytest --collect-only -q tests
python -m pytest -v tests/ops/test_ops.py

# 选择某个运算或后端 marker
JITTOR_TEST_DEVICES=cpu python -m pytest tests/ops/test_ops.py -k exp
python -m pytest -m structure tests/structure

# 兼容层测试：走符号链接的那条路径，不要直接指 compat/
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
    python -m pytest python/jittor/compat/tests/torch

# 可复现门禁
python -m nox -s structure
python -m nox -s cpu
python -m nox -s optional
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s nccl
```

nox 会话会建立隔离的状态与缓存。**直接并发运行时也必须使用不同的 `JITTOR_HOME` 或
`cache_name`；新 JIT 运算或扩展的首次构建应当串行执行。**

兼容层那条命令为什么要绕一下：`compat/` 自己带 `__init__.py`，直接
`pytest compat/tests/torch` 会让 pytest 把它当**顶层**包 `compat` 导入，而
`compat/_aliases.py` 的 `from .._runtime import ...` 要求自己是 `jittor.compat`，
于是每一个用例都以
`ImportError: attempted relative import beyond top-level package` 报错——收集阶段
看起来是好的（1888 collected），错误全落在 setup，所以很容易误判成"收集不了"。
`python/jittor/compat` 是指向 `compat/` 的符号链接，从它进去包名就解析成
`jittor.compat.tests.torch`，相对导入成立。门禁里没有这个问题是因为
`_run_pytest_once` 会先 `pip install -e compat/`，把 `jittor-torch` 装成真包。

## JIT 缓存：冷启动的代价，和它什么时候白付了

一次冷缓存的运行比热缓存慢一个数量级，差别**只在缓存**。在 8 卡机上用独占的
`JITTOR_HOME` 与 `cache_name` 实测，同一条命令冷/热各一次：

| | 用时 | 结果 |
| --- | --- | --- |
| `tests/nn` 冷 | **37:03** | 360 passed, 71 skipped, 948 subtests |
| `tests/nn` 热 | **1:02** | 360 passed, 71 skipped, 948 subtests |

### 冷缓存的耗时榜不是"重型文件"榜

`--durations=40` 占冷跑总时长的 78.3%，前两项就占 34%。但把同样的用例在热缓存下
再看一遍，这张榜就散了：

| 用例 | 冷 | 热 |
| --- | --- | --- |
| `test_bmm.py::TestBMM::test_bmm_cuda` | 466.00s | **1.16s** |
| `test_depthwise_conv.py::TestDepthwiseConv::test_data` | 291.08s | 10.46s |
| `test_norm_unification.py::…::test_module_and_functional_agree` | 98.50s | 1.98s |
| `test_pool3d.py::…::test_cuda_backward_matches_cpu_backward` | 59.88s | 1.51s |
| `test_knn.py::TestKnnOp::test_knn` | 54.82s | 1.68s |

**`tests/nn` 里没有重型文件。** 冷跑的耗时榜是一张"谁先跑到、谁替全场付编译费"
的表，榜首取决于收集顺序而不是用例本身。拿冷缓存的 durations 去挑"要优化的测试"，
挑到的是收集顺序。

### 预热预热不掉什么：72% 的 kernel 的 key 是那张子图

冷跑一次 `tests/nn` 落地 604 个 JIT kernel：

| 类别 | 数量 | 占比 | 通用预热能否提前编出 |
| --- | --- | --- | --- |
| 融合 kernel（`__opkey…`） | 436 | 72% | **不能** |
| 库自带的 `code` kernel | 111 | 18% | 能 |
| 其它单算子 kernel | 57 | 9% | 能 |

融合 kernel 的 key 由**被融合进去的那串算子**拼成——每个算子的 dtype、秩、是否
跨步、广播形态，按融合顺序排列——而不是由算子名决定。所以"把常用算子组合编一遍
再退出"这个方向按实测只够得着约 28%，**剩下 72% 不跑出同一张图就不会命中**。
这是否定结论：能摊销冷代价的不是"预热更多算子"，而是**让本该命中的缓存真的命中**。

### 真正让"热缓存看起来是冷的"的那件事：locale 进了缓存目录名

缓存路径里有一段 `archXXXXXXXXXX`，它是 `cc -march=native -Q --help=target` 输出
文本的哈希——用编译器实际展开的指令集做 key，本意是对的。但这份输出里每个特性的
开关状态是**经 gettext 翻译**过的：

```text
LC_ALL=C           -m128bit-long-double  [enabled]   -> arch0fe147db07
LC_ALL=zh_CN.UTF-8 -m128bit-long-double  [启用]      -> arch6fe5b27f7d
```

同一台机器、同一个编译器、同一颗 CPU、同一组 flag，**两棵完全不相干的缓存树**：

```text
…/Linux-5.4.241-x7a/arch0fe147db07/da0673f5cfe0/demo/cfgd436a523
…/Linux-5.4.241-x7a/arch6fe5b27f7d/da0673f5cfe0/demo/cfgd436a523
```

这不是理论问题，本仓库正好两边都占：`noxfile.py` 的 `_session_env` 设了
`LC_ALL=C`，而仓库里**再没有第二处**设它。于是 `nox` 门禁用一棵树，所有手跑的
`pytest`（以及 `tools/run_test_suite.py`，它不设 locale）在非英文机器上用另一棵，
在同一台机器上**各自付一遍对方已经付过的冷编译**。`_shared_jittor_cache()` 共享
`JITTOR_HOME` 是生效的——它只是被同一个函数里 12 行之外的 `LC_ALL=C` 架空了。

实测代价（空 `JITTOR_HOME`，`JT_PROBE_CACHE=0`，只跑 `import jittor`）：

| | 第 1 次 `LC_ALL=C` | 第 2 次 `LC_ALL=zh_CN.UTF-8` | 第 3 次 同上 |
| --- | --- | --- | --- |
| 修之前 | 60s（冷） | **59s（又一次全冷）** | 3s |
| 修之后 | 60s（冷） | **2s（命中）** | 2s |

修之前落地两棵 `arch*` 树，修之后只有一棵。这只是 `import jittor` 的核心构建；
一个套件还要在第二棵树里把全部 JIT kernel 再编一遍。

这台机器上两棵树现在就并排躺着：

```text
arch0fe147db07   1.1G   7801 个 kernel
arch6fe5b27f7d   113M    196 个 kernel
```

`probe.json` 让这件事时隐时现而不是稳定可见：它按编译器缓存这个探测结果、不按
locale，所以**哪个 locale 先填上它，就由哪个决定全机器的缓存目录**，直到编译器的
size/mtime 变了为止——那一刻整台机器会悄悄搬到另一棵树上，重编一切。

修法两步。一是让探测本身在 C locale 下问（`jittor_utils.c_locale_environment()`），
而不是要求每个调用者去 export；它收敛到的正是 nox 门禁已经在用的那棵树，所以门禁
的热缓存不受影响，手跑的运行并进来。二是把 memo 的槽位名从 `target_arch:<cc>` 改成
`target_arch:c-locale:<cc>`——**只改探测不改槽位是不够的**：已经存过翻译文本的
`probe.json` 会继续作答，于是 `target_arch_key()` 和它自己的缓存对不上，那台机器
永远停在第一个问它的 locale 选中的目录上。改了槽位名，每个 home 重新探测一次就
全部落到同一个 key（本机实测：改之前进程解析到 `arch6fe5b27f7d`，改之后
`arch0fe147db07`，也就是那棵 1.1G、7801 个 kernel 的热树）。门禁：
[`tests/build/test_build_config_cache.py`](https://github.com/Jittor/jittor/blob/master/tests/build/test_build_config_cache.py)
`::test_the_instruction_set_key_does_not_depend_on_the_message_locale`，
先证明它会红（`{'C': 'arch0fe147db07', 'zh_CN.UTF-8': 'arch6fe5b27f7d'}`）再修。

审计过其余进缓存 key 的探测（`version:`、`cuda_archs`、`nvcc_archs:`、
python-config、mpicc）——它们都只取结构化的数字或 flag，**只有 `target_arch`
把翻译过的自由文本喂进了哈希**。

### 第二件：oneDNN 从源码编译，四路并行，而且没人认领

`mkl`（oneDNN v3）是**懒加载**的：`import jittor` 不碰它，
`tools/run_test_suite.py` 原来的预热探针（`(jt.array([1.,2.])*2).sum()`）也不碰它。
冷缓存下它要从源码 cmake 构建 ~1200 个 TU，而 `--parallel` 曾被写死成
`min(4, cpus)`。于是这段构建落在**收集顺序里第一个走到 CPU 卷积/matmul 的用例**
头上，表现为一个没有任何输出的长停顿。

隔离 A/B（同一份源码树，先后串行跑，机器空闲）：

| `cmake --build --parallel` | 用时 |
| --- | --- |
| 4（原值） | **443s** / **448s** |
| 32（现默认上限） | **87s** / **87s** |

约 **5 倍**，一次冷缓存省掉约 6 分钟。现在的取值按核数与内存两头收敛
（`JT_ONEDNN_BUILD_JOBS` 可直接覆盖），小 runner 上仍然退回到核数。

端到端确认（预热探针逐库计时，同一台机器）：

```text
mkl       available  105.3s      <- oneDNN 源码构建，并行度已放开
cutt      available   10.6s
cub/cudnn/cublas/curand/cufft/cusparse   各 0.0s
```

原来那次冷跑里，同一个 oneDNN 构建在 `--parallel 4` 下花了约 480s，还被记在
`test_bmm_cuda` 头上。

同时预热探针改为显式把后端库建出来（`probe_library(..., load=True)`）：
**这不减少总工作量**，它只是让这笔开销落在预热步骤里可归因、可加超时，
而不是伪装成某个 bmm 用例慢了 466 秒。通信库（`mpi`/`nccl`/`hccl`）**不在**
预热名单里——它们的 loader 会初始化通信子并等待其它 rank，这里没有其它 rank，
一个可能永久阻塞的预热比它要替换掉的停顿更糟。

### 缓存命中不进全局锁（它曾经要，那是并行的瓶颈）

`Op::jit_run`（`src/core/op.cc`）先查**进程内**的 `jit_ops` map，命中就直接跑；
没命中才走 `OpCompiler::do_compile`。`jit_ops` 只帮到"同一个进程里第二次用同一个
key"——磁盘缓存全热时，每个进程对每个不同的 JIT key 都要走一次 `do_compile`。

**过去 `do_compile` 第一行就是 `jittor::lock_guard lg`**，整段（包括纯命中）都在
全局 `jittor.lock` 里。锁里干的事在两种情况下差别巨大——冷是一整次 nvcc/g++
（数秒），热只是 key 比对加一次 `dlopen`（毫秒）——但命中路径照样要排队。

现在 `do_compile` 不再拿锁；锁下沉到 `jit_compiler::compile()`，**只有真要编译时
才拿**。命中走一条免锁快路径：算出这条编译命令现在会产生的 cache key，和产物旁边
记录的 `<product>.key` 比对，一致就直接 `dlopen`。

这条路安全的唯一依据是**发布顺序**：`cache_compile()` 先把成品 `rename()` 到位，
**之后**才写 `.key`，两次都是同目录内的 `rename()`，是原子的。所以"看得见匹配的
key"蕴含"它旁边的目录项已经指向那次构建的完整产物"——不存在 key 已发布而产物没
发布的中间状态。生成的源码是唯一只在内存里的输入，快路径**不写它**（写是构建动作），
而是直接哈希那个字符串，这与哈希"它将要写成的那个文件"完全等价。

`tests/build/test_jit_cache_concurrency.py` 就是守这几条的：它从外部采样一次慢速
重建，一旦出现"新 key 配旧产物/半个产物"就失败（把 `install_file` 改成就地拷贝可以
复现这个失败）；它还让 8 个进程向同一份缓存并发冷编译重叠的 kernel 集合，跑 3 轮，
要求每个进程的数值逐条一致。

实测（8 卡机，独占 `JITTOR_HOME` 与 `cache_name`，每个数字跑两次）：

| | 修之前 | 修之后 |
| --- | --- | --- |
| 同一批全热 kernel，无人持锁 | 0.226s / 0.225s | 0.239s / 0.223s |
| 同一批全热 kernel，另一进程持锁 30s | **30.231s / 30.231s** | **0.225s / 0.225s** |
| `tests/nn` 单进程热跑 | 59.7s / 61.1s / 61.2s | 60.9s / 59.9s |
| `tests/nn` 按文件分 8 份、8 进程共享同一热缓存 | **76.9s / 85.9s** | **26.4s / 30.1s** |

第二行是机制本身：修之前它整整等满 30 秒，修之后完全不受影响。第四行是结果：同样
8 个进程共享同一份热缓存，过去比单进程**慢** 1.3–1.4 倍，现在比单进程**快** 2 倍
以上。四组 `tests/nn` 运行都是 360 passed、0 failed，与串行完全一致。

（注意第三行：单进程不与任何人争锁，所以本来就不该变快，实测也没变——这条是防回归
用的，不是收益。）

测这种事有两个必须先排掉的干扰，否则数字测的是别的东西：`import jittor` 自己整段
跑在 `lock_scope()` 里，所以**在 import 之前**起的持锁进程会被 import 等掉；而第一次
`transpose()` 会经 Python 的 `compile_custom_ops` 懒构建 `cutt` 后端库，那是另一条
（仍然持锁的）每进程一次的路径，不是每个 JIT key 的开销。两者都要在开表之前做完。

**还没做的：** Python 侧 `@lock.lock_scope()` 的 `compile_custom_ops` /
`compile_module` / `download_url_to_local` 仍然是整段持锁，即使只是缓存命中。它是
每进程一次而不是每 key 一次，在上表第四行里已经不是主项，但多进程冷启动时它仍然会
串行化。

以前这一节推荐的出路是"给每个 worker 自己的 `JITTOR_HOME`、用 `cp -al` 硬链接播种
一份预热缓存"。**现在不需要了**，而且那条路本来就有代价：每个新 home 第一次
`import jittor` 仍会重建 `jit_utils` 并以退出码 3 要求重跑，那次重建还会让已播种的
kernel 失效一部分（实测每个 worker home 重编约 84/875 个）。共享一份热缓存现在既
更省磁盘也更快。

（同一份数据也说明一件事：这台 384 核机器在冷跑 `tests/nn` 时 `user/real` 只有
**1.5**，热跑是 **4.9**——瓶颈从来不是算力。）

### 读这些数字时要当心的两件事

**"热缓存"是要被证明的，不是默认成立的。** 上面那张 37:03 / 1:02 的表里，两次跑
的 passed/skipped/subtest 完全一致；一次运行慢了 35 分钟而报告长得一模一样。所以
一个"慢"的观测在归因到别的原因之前，先确认缓存真的是热的——最省事的确认方式是
**把同一条命令再跑一遍**，第二遍的时间才是这套用例本身的耗时。

**跑大目录时用 `-v`，不要用 `-q`。** `-q` 的失败清单**只在最后汇总时打印**：
一次跑了 4 小时、在 76% 处被 timeout 杀掉的 `pytest tests/ -q`，日志里一条失败也
没留下。`-v` 每跑完一条就写一行 `…::test_x PASSED/FAILED`，所以被杀掉的运行仍然
留着到那一刻为止的完整记录。实测同一个文件：`-v` 在汇总之前有 23 行结果，
`-q` 是 0 行。这条和「中途死掉的会话产出一份"结束了"的日志」是同一个毛病的两面，
配合 `tools/check_session_completed.py` 的哨兵一起用。

**共享缓存并发跑，要区分冷和热。** **热**的那一份现在可以共享着并行跑：缓存命中
不再进 `jittor.lock`（见上文"缓存命中不进全局锁"）。**冷**的那一份仍然串行：真正
要编译的进程还是要拿同一把全局锁，所以想靠"多开几个进程"来并行完成首次编译依然
无效——**新 JIT 运算或扩展的首次构建应当串行执行**，或者给它自己的 `JITTOR_HOME`
/ `cache_name`。进程内的并行编译器（`use_parallel_op_compiler`）仍是冷编译唯一的
并行路径。

## CI 支持矩阵

工作流状态是**显式声明**的，因为"某个 nox 会话能跑"并不等于 CI 拥有所需的硬件或依赖。
"手动"意味着维护者必须在已配置的机器上运行指定的 fail-closed 会话，之后才能声称该接口
已验证——**不得把它描述成自动检查**。

| 会话 | CI 状态 | Runner 与触发 |
| --- | --- | --- |
| `cuda` | 自动 | 推送与每周计划使用声明的 RTX 4090 / CUDA 12.2 runner。维护者可给 PR 打 `ci:cuda` 标签；打标签、重开和之后的 synchronize 事件都会跑同一门禁。 |
| `optional` | 手动 | 未声明依赖完备的 CUDA runner。请在下述预置环境中运行 `python -m nox -s optional`。 |
| `rocm` | 手动 | 未声明 AMD/ROCm runner。请在真实受支持的 AMD GPU 上运行 `python -m nox -s rocm`。 |
| `mpi` | 手动 | 未声明多进程 MPI runner。请在有可用启动器与编译器包装器的环境运行 `python -m nox -s mpi`。 |
| `nccl` | 手动 | 声明的 CUDA runner 只保证一块 RTX 4090，而该门禁需要两块可见 GPU。请在多卡主机上运行 `python -m nox -s nccl`。 |

**该矩阵描述的是当前调度，不是后端支持程度。** 手动结果必须记录被测提交、工具链、
设备拓扑、命令和 pytest 结果；**runner 不可用不是通过。**

维护中的 CUDA 会话跑完整的 CUDA 后端目录、dtype 覆盖、CPU/CUDA 设备一致性、Torch TF32
控制项和严格的 CUDA OpInfo 套件。

`nccl` 会话默认需要两块可见的 NVIDIA GPU。它先为每个 rank 串行预热一个隔离的 JIT 缓存，
再用 `jittor.distributed.launch` 验证真实的扁平 FSDP2 参数分片、NCCL all-gather 与
reduce-scatter，以及针对独立 NumPy 结果的分片优化器更新。设置 `JITTOR_NCCL_WORLD_SIZE`
并暴露至少那么多设备，可以跑多于两个 rank。

`optional` 会话是一个 fail-closed 的离线 CUDA 门禁，针对预置的 TorchMetrics、
mmcv-lite/MMEngine、PEFT、Safetensors、TensorDict 和已部署的 FlashAttention 适配器。
它在 pytest 之前逐个探测每个包、显式启用 Jittor Torch shim，并把 **PEFT 导入失败当作
错误而不是可选 skip**。当 `JITTOR_FLASH_ATTN_JITTOR_SRC` 指向官方 FlashAttention
checkout 时，该会话分两阶段：常规可选测试用已部署的 math 适配器运行，然后一个
**要求原生实现**的阶段跑融合 fp16 前向、稠密/变长/打包反向、dropout RNG 重放、GQA 和
float32 opt-in 测试。原生阶段默认 head 维度 32 与 fp16；FlashAttention 的能力环境变量
是在这个基础集上**扩展**而非替换。**原生阶段不能靠回退来满足。**

## 增加覆盖

当一个运算可以共享标准的样本、参考、dtype、梯度和设备一致性机制时，用 **OpInfo 定义**。
当契约涉及状态、变更、序列化、错误行为、模块生命周期、导入顺序、分布式协调，或某个
无法用 OpInfo 表达的具体回归时，用**定向测试**。

一个新运算通常需要：

1. 独立的前向样本与参考；
2. 适用的 dtype、形状、轴、空张量、广播和非连续用例；
3. 有测试支撑的梯度与二阶梯度声明；
4. 对**每一个宣称支持的加速器**做设备一致性；
5. 显式的错误契约测试；
6. 一条 OpInfo 报告条目或定向测试名，使**缺失的覆盖可被发现**。

## 验收

一次测试体系改动在满足下列条件时算完成：收集过程无设备副作用、新的 harness 逻辑有定向
自测、结构门禁通过、至少有一个强制 CPU 用例真正执行，并且每个硬件结果都区分了通过、
框架失败与环境不可用。

**数量本身不是验收标准**——证据必须真正检验所声称的语义。
