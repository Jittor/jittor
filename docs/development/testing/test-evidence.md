# 数值、梯度与覆盖证据

[本栏目目录](../index.md) · [English](test-evidence.en.md)

前提：准备源码 checkout，并了解[测试体系](test-system.md)中的进程模式。本页说明数值结果和覆盖计数分别能证明什么。

## 算子证据的三层

## 1. 独立前向参考

每个 `OpInfo` 描述可调用对象、样本构造器、dtype、自动微分支持、容差，以及定向的
skip/xfail 策略。`tests/ops/test_ops.py` 把这个数据库在所请求的设备上展开，并与独立的
NumPy 或数学参考比对。

**参考实现不得调用被测的那个 Jittor 运算。** 样本构造器要覆盖有意义的形状、轴、广播、
可选参数和容易出错的 dtype，而不是**在没有语义多样性的前提下堆数量**。

## 2. 数值梯度

可导的 OpInfo 条目跑 float64 的 CPU `gradcheck`；支持二阶微分的运算还跑 `gradgradcheck`。
标注 `supports_autograd=False` 或 `supports_gradgrad=False` 是**契约决定，不是把失败的
测试弄绿的手段**，必须在定义中或问题总账里给出理由。

数值检查在 CPU 上证明导数公式。**它不证明加速器正确执行了同一个反向 kernel。**

## 3. 设备一致性

[`tests/backends/parity/test_device_parity.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/backends/parity/test_device_parity.py)
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
| [`tests/structure/public_api_manifest.json`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/structure/public_api_manifest.json) 与其门禁 | 原生的 1294 个名字**还在不在** |
| [`tests/structure/torch_api_manifest.json`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/structure/torch_api_manifest.json) 与其门禁 | Torch 面的 1295 个名字**还在不在** |
| [`tests/_helpers/api_coverage.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/_helpers/api_coverage.py)（`JITTOR_API_COVERAGE=1`） | 一次运行**真的调用**了哪些入口 |
| [`tools/api_coverage_ratchet.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/api_coverage_ratchet.py) 与两份基线 | 没被调用的那个集合**有没有变大** |
| [`tools/probes/opinfo_dtype_gaps.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/probes/opinfo_dtype_gaps.py) | 哪些 (算子, dtype) **跑得通却没声明** |

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

[`tests/core/test_var_residency_contract.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/core/test_var_residency_contract.py)
是这一类的第一个：未计算的 Var 没有驻留；读取 device Var 的数据会把存储真的迁回
主机；`device_id` 跨 `cpu()` 保留源设备而 `device`/`location()` 跟随数据。它整体
在 CPU 上运行，CUDA 用例逐条 skip——**去掉加速器只会收窄它而不是清空它**。

这一点是有来历的：Var/Module 的设备方法契约此前只有四条用例，且全部要求两张
CUDA 卡，于是在单卡与 CPU 机器上整体跳过，报告里和四条通过长得一模一样。现在
"设备数量不够"由 `insufficient-devices` 单独计数并在汇总里点名，与"没有加速器"
区分开——前者是硬件在场却仍丢掉的覆盖，不是环境事实。

## 运行时内省：测试该读什么

测试与共享 pytest 策略通过 [`jt.introspection`](../../notes/runtime/runtime-introspection.md)
观察能力、策略与计数，而不是直接读原生 flag 或编译器属性。迁移时保留的是**测试真正在问
的那个问题**：

| 旧读法 | 受支持的替代 | 边界 |
| --- | --- | --- |
| `jt.compiler.has_cuda`、按 `jt.flags.use_cuda` 跳过加速器用例 | `jt.introspection.capabilities.backend("cuda")` | 先检查 failed，再按 disabled/absent 跳过 |
| 按后端名调 `core.get_device_count()` | `jt.introspection.capabilities.devices(name)` | 区分 FAILED/UNPROBED 与 0 |
| `compile_extern.cudnn_ops is not None` 之类的库检查 | `jt.introspection.capabilities.library("cudnn")` | UNPROBED 不等于不可用 |
| 当前的 `jt.flags.use_cuda`、`amp_reg`、`no_grad`、精度读取 | `jt.introspection.policy.runtime.<name>` | 保持生效策略的语义 |
| `jt.flags.cache_path` 与编译器配置读取 | `jt.introspection.policy.startup.<name>` | 不可变的启动配置 |
| `jt.flags.exec_called` | `jt.introspection.counters.exec_calls` | 不隐式执行图 |
| `stat_allocator_total_alloc_byte` 及对应的调用/释放计数 | `jt.introspection.counters.allocator.*` | 插桩模式与重置语义要紧 |
| `number_of_lived_vars/ops`、`number_of_hold_vars`、`liveness_info()` | `jt.introspection.counters.live_vars/live_ops/held_vars` | 不隐式 GC |
| 给 flag 赋值 | 可逆时用 `jt.runtime.scope(...)` | 写入不属于内省 |

**例外是显式登记的。** 那些本身就在验证原生 setter、注册内部或绑定生成的测试，保留对
它们所测实现的直接访问；这些文件连同理由登记在
`tests/_helpers/introspection_exceptions.json`。这份登记表目前没有门禁读取，新增例外时
同步更新它，别让它成为又一份单向的清单。

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
