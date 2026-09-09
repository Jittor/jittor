# 错误分级

Jittor 把错误分成**两档**，各有明确的入口：

| 入口 | 用途 | 结果 |
| --- | --- | --- |
| `USER_CHECK` / `USER_CHECKop` | 校验用户传入的参数 | 抛出**可捕获的** `RuntimeError` |
| `ASSERT` / `ASSERTop` | 保护内部不变量与后端/运行时状态 | 内部错误 |

**除非能证明某个公开参数直接抵达该检查，`ASSERT` 必须保持为非用户错误。**

CUDA 后端里报告计划失败、CUDA/NCCL/CUDNN 状态或测试 harness 失败的检查，一律归类为
内部不变量。它们由 `tests/structure/core/test_error_categories.py` 的结构门禁跟踪，
避免今后的错误边界迁移**不小心削弱后端诊断**。CUB 测试 kernel 的 CUDA 状态断言也属
此列——它们不是用户输入校验，因此保持 `ASSERT`。

当主机上没有对应的 CUDA、CUDNN、CUB 或 NCCL 设备时，仅限硬件的负向测试可以注册为静态
契约。**静态证据不声称硬件路径真的跑过。**

当前后端清单中没有其它可以安全转成用户输入断言的项；今后要把某个 `ASSERT` 改成
`USER_CHECK`，必须**先演示一条直达的公开参数路径**。

## 绑定层：`is_type` 已经覆盖了什么

`py_converter.h` 看起来像一大块未迁移的用户边界，但**大部分不是**。生成的包装器通过
对每个参数询问 `is_type<T>` 来解析重载，**之后**才调用 `from_py_object<T>`。因此非法
输入在转换**之前**就被解析器拒绝了，抛出的是可捕获的 `RuntimeError`，并且会点名算子、
列出收到的类型。

`from_py_object` 内部那些 `CHECK(is_type<...>)` 只在同一个生成绑定的这两半互相矛盾时
才触发——那是**框架自相矛盾**，所以它们保持为内部不变量。

例外是从绑定**返回出来**的值，因为没有任何解析器看过它：`GradCallback` 收到的是用户
`Function.grad` 返回的任何东西。梯度数量不对或返回了非 Var 是**调用者的错误**，属于
`USER_CHECK`。将来如果某个参数确实带着转换器拒绝的元素通过了 `is_type`，那时才会有
另一处检查越过这条边界。

## 析构函数与信号处理器

两者都**不能通过抛异常来报告**，而 `LOGf` 会抛。这条约束能否守住，取决于两件事，
而 grep `LOGf` 只是其中第一件。

**扫描范围。** 扫描必须覆盖每一个存放 C++ 的根目录——现在是 `python/jittor/src`、
`python/jittor/extern` **以及 `backends/`**。CUDA、ACL 和 ROCm 的 kernel 已经迁进了
最后那个目录，只点名前两个的扫描会**对它们保持沉默却仍报出一个健康的总数**。扫描还
必须覆盖 `.cu` 和 `.cuh`，不只是 `.cc` 和 `.h`。这就是
`test_destructor_and_handler_contract.py` 断言**每一个根目录都贡献了析构函数**、
而不是断言一个总数的原因。

**字面 grep 是必要条件，不是充分条件。** 一个调用了任何未标 `noexcept` 之物的析构函数
仍然可能抛出：`~VarHolder` 曾经经由 `release_both_liveness()` 抵达一个 `ASSERT`，
静态扫描什么也没看见，进程照样中止。因此检查一个析构函数意味着**把调用图往外读一跳**，
并且在析构函数确实调进运行时的地方，要有一个真的会跑的销毁测试
（`tests/backends/cuda/test_var_holder_teardown.py`）。

信号处理器一侧是 `segfault_sigaction` 以及它能触及的一切：`sig_write*`、
`print_trace_from_signal` 和 `sigquit_callback` 列表。**它们只允许 `write(2)` 和
`_exit`。** `handle_signal` 是仅限 Windows 的处理器；它不抛异常，但用了 `std::cerr`
和 `abort()`，两者都不是异步信号安全的，因此与这两档的划分分开跟踪。
