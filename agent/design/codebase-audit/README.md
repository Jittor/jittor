# Jittor 2.0 代码库设计审计

2026-09-02。范围是 2.0 分支的全部代码：C++ 核心运行时、Python API 层、Torch 兼容层、
构建与打包工具链、测试体系、后端实现与分布式、以及跨层的架构与代码组织。七个方向
分别审计，每条发现都带 `文件:行号` 证据，按严重度分级。本页是总览与优先级，细节在
七份分报告里。

方法：静态阅读为主，辅以只读的运行时验证（CPU-only、独立缓存目录）。未修改任何文件。
凡标注"运行时验证"的结论都实际复现过；凡是推断都写明是推断。

## 七份分报告

| 报告 | 范围 | 该层最根本的问题 |
| --- | --- | --- |
| [核心 C++ 运行时](01-core-runtime.md) | `python/jittor/src/` 46605 行 | Node 是所有子系统共用的可变涂鸦板；该层最微妙的部分（liveness 与融合策略）不在源码树里 |
| [Python API 层](02-python-api.md) | `python/jittor/` 除 compat/extern/src 外约 40k 行 | 没有对象模型只有一堆约定：无视图与存储、设备不是张量属性、参数身份是散落的标记属性 |
| [Torch 兼容层](03-compat-shim.md) | `python/jittor/compat/` 28k 行 | 它根本不是一层：`torch is jittor`，不存在独立的 Torch 对象模型 |
| [构建与工具链](04-build-tooling.md) | compiler、compile_extern、jittor_utils、noxfile、打包 | `import jittor` 启动的是一套完整构建系统，没有阶段划分没有事务 |
| [测试体系](05-tests.md) | `tests/` 289 文件 2513 用例 | 测试资产与门禁彻底解耦；投入方向倒挂 |
| [后端与分布式](06-backends.md) | `python/jittor/extern/` 28790 行 | 每个库封装都是抄来的调用序列，不是有生命周期与错误契约的适配层 |
| [架构与代码组织](07-architecture.md) | 横切：抽象、分层、边界、重复、API 面 | 把绑定、构建、兼容三层的关注点焊进了核心数据结构 |

## 五条贯穿全局的判断

**一、这个框架有一块看不见的核心。** `python/jittor/src/` 下有五个头文件没有对应的 `.cc`：
`fuser.h`、`node.h`、`opt/pass/{atomic_tuner,shared_reduce,parallel}_pass.h`。它们的实现装在
`python/jittor/utils/data.gz`（437 KB）里，`compiler.py:1402-1429` 在构建时解压成 `data.cc`、
编译成 `data.o`、然后删除源文件。已核实：解压后 569 行 1.5 MB，标识符全部替换成 `x10364`
这种形式，字符串十六进制转义，宏改名，并插入由 `src/utils/vdp` 消除的噪声行；导出符号 31 个，
含 `jittor::count_fuse`、`jittor::Node::free`、`own_forward_liveness`。也就是说**三套 liveness
引用计数和融合划分策略不可读、不可 diff、不可单测**。而这两样东西恰恰是多设备、执行计划
缓存、执行流水化三项重构都必须改动的地方。这是所有结论里最优先要处理的一条。

**二、最危险的失败模式是"不报错、结果错"，而且成规模存在。** 七份报告独立地收敛到同一
个模式。举证据最硬的几条：`_th_require_grad` 与 `_is_scalar` 共用同一个 flag 位
（`node.h:47-48`），于是每个 `requires_grad_(True)` 的参数在类型推导里被当成标量，**AMP 在
所有涉及参数的算子上静默失效**；`.item()` 对无符号 dtype 读满 8 字节而只写了 dsize 字节
（`var_holder.cc:284`），`jt.array(np.uint8([200])).item()` 返回随机大整数；融合图的边在 jit key 里
用两位十六进制编码算子号（`fused_op.cc:181`），超过 255 个后回绕，两个结构不同的融合段
命中同一个已编译 kernel；`torch.autocast` 是完全空操作，混合精度训练静默跑成 fp32；
DDP 没有任何 hook 与 bucket，标准的 `loss.backward(); opt.step()` 写法下梯度从不同步，
N 卡训出 N 个不同的模型；MPI 的 int64 被映射成 `MPI_DOUBLE_INT`，按元素数传会读越界。
Python API 层的审计者在 CPU 上实际复现了 20 条这类缺陷。共同点是它们都发生在**并行路径
从不交叉验证**的地方。

**三、本该是结构化数据的东西全是文本。** jit key 是 2 MB 无边界检查的字符缓冲，溢出靠
guard page 段错误在信号处理器里报告；KernelIR 用 `string type` 加 `map<string,string>` 表示
中间表示，`get_attr` 对拼错的键静默返回空串；算子身份用 `name()` 字符串比较 25 处，其中
包含"跳不跳 NaN 检查""流不流水"这类正确性判据；Python 绑定用正则扫描 C++ 头文件生成，
`split_args` 只数尖括号不数圆括号；后端移植的机制是把**整个源码树复制一份做文本替换**，
其中一处靠把 `run_pass<FloatAtomicFixPass>();` 替换成字面量 `WTF` 让该 pass 编译失败从而
关掉它。文本处理的共同后果是：改一个名字、加一个成员、换一种写法，行为就静默改变。

**四、抽象边界与目录边界不重合，而且方向是反的。** 最底层的 `Node` 反向 include Python
tracer；C++ 核心里有 16 处 Torch 兼容概念 `th_mode`；后端模块在 import 时改写编译器全局
22 处；核心算子按字符串名认识可选后端 5 处；优化器 tuner 里写死 `mkl_conv` 与 `cudnn_conv`。
依赖图里有三个真环（`jittor_utils` ⇄ `compiler`、`Executor` ⇄ `VarHolder`、`Node` ⇄ tracer）。
后果是没有任何一层可以单独理解、单独测试、单独替换：加一个后端要改核心，改一行核心要
重新验证每个后端的文本替换是否还成立。

**五、门禁保护的不是最容易出错的东西。** 227 个算子的导数公式正确性在所有门禁中实例化为
**零个用例**——`@onlyCPU` 标注的反向测试在 CUDA 门禁下被设备过滤全部跳过，而该文件根本
不在 CPU 门禁的白名单里。项目的核心目标（与 PyTorch 数值一致、速度不更慢）对应的生态
对拍测试不可从任何门禁到达。289 个测试文件里 215 个不在任何 CI 路径上。与此同时，跑得
最全的是 `tests/structure`（234 用例 8071 行），它断言的是模块路径、re-export 恒等和文件
行数预算——也就是说，改错一个 kernel 的算术不会被任何 PR 门禁挡住，而重命名一个文件会。

## 优先级

**第一档，先做能改变"能不能信"的**

1. 把 `data.gz` 里的五个翻译单元还原进源码树。在此之前任何涉及 liveness 或融合的改动都是
   盲改。（[核心](01-core-runtime.md)）
2. 修那批静默算错的缺陷。按入口常见程度排：flag 位重叠导致的 AMP 失效、转置标记陈旧
   （任何优化器 step 都触发）、`no_grad` 装饰器递归后永久泄漏、`.item()` 无符号读越界、
   融合边号回绕、`jt.Function` 实例复用。（[核心](01-core-runtime.md)、[Python](02-python-api.md)）
3. 把"看起来支持其实是空操作"的 API 全部改成明确报错，需要显式开关才降级为静默。
   兼容层里已列出 14 条，其中 autocast、`load_state_dict(strict=)`、`torch.load(weights_only=)`、
   DataLoader 的 `num_workers`、DDP 的梯度同步是最危险的五条。（[兼容层](03-compat-shim.md)）
4. 让算子级反向门禁真正跑起来（一行设备过滤的修正加一条门禁条目），并把生态对拍接进
   nightly。没有这两条，上面的修复无法防止复发。（[测试](05-tests.md)）

**第二档，消除结构性成本**

5. 视图与存储模型。这是 114 个 `foo_` 就地方法、转置隐藏标记、兼容层三条回写标记链的
   共同根因。（[Python](02-python-api.md)、[兼容层](03-compat-shim.md)）
6. 设备与后端注册表。设备成为 `Var` 的属性，后端成为运行期注册项而非构建期常量；
   Python 层 98 处 flag 判断、核心 5 处字符串名查询、三份 compiler 全局改写一并作废。
   （[后端](06-backends.md)、[架构](07-architecture.md)、另见 `../multi-backend-design.md`
   与 `../device-placement.md`）
7. 执行器拆成 Planner 与 Runner，调度结果按图结构缓存。这是 520 行的 `run_sync` 里
   同时住着 BFS、融合、拓扑排序、内存分配、迁移、编译、发射、profiling 的直接后果。
   （[核心](01-core-runtime.md)、[架构](07-architecture.md)）
8. 参数模型：把"什么是参数"从 26 个标记属性收敛为类型，五份遍历收敛为一份。
   （[Python](02-python-api.md)）

**第三档，基础设施**

9. 缓存路径加构建配置指纹；锁统一为一种类型；探测结果落盘。这三条决定后续每一项改动的
   验证周期。（[构建](04-build-tooling.md)）
10. 门禁分层：smoke 进 PR，full 进 nightly；默认跑整个 `tests/` 用 marker 做减法，而不是
    白名单做加法。（[测试](05-tests.md)）
11. 删掉两处 import 期的全局副作用（重置随机种子、替换 `PIL.Image.open`）与进程级关闭
    TLS 校验。三行删除，但它们让"复现实验"和"在同一进程里用别的库"多了三条隐藏规则。
    （[Python](02-python-api.md)、[构建](04-build-tooling.md)）

## 值得保留的

审计不是只列问题。有三样东西经得起检验，重构时应当原样保留：**统一惰性图加融合执行的
执行模型**（本轮实测中 Jittor 的 GPU kernel 总时间已优于 PyTorch，差距在调度而非计算）、
**JIT 代码生成与循环变换管线**（`src/opt/` 82 文件 7589 行，是真正的差异化资产）、
以及**按域拆分的 Python 包骨架**（名字级检查显示确实没有重复实现）。前两项正是这个框架
存在的理由，第五节列出的问题都不在它们本身，而在它们周围的接缝上。
