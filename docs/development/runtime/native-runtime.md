# 原生运行时、状态与提交

[本栏目目录](../index.md) · [English](native-runtime.en.md)

修改运行时状态、holder 生命周期或图提交时阅读本页。先读[源码架构](../source-architecture.md)了解模块归属。

本页描述实现契约，不是新的硬件验收报告。原文复查基线为 `e3c369acb`；本次拆分保留该证据边界。

## 根模块归属

`python/jittor/` 下直接存在的 Python 文件只有 `__init__.py` 与 `selftest.py`；算上
`__init__.pyi`，根目录共三个源文件。参数策略在 `_core.arg_policy`，命名空间组合与安装器
排序在 `_runtime.composition` 与 `_runtime.install_order`，计时 API 在 `tools.benchmarking`。
历史导入保持为显式的同对象别名。

`__init__.py` 在编译核心引导之后发布显式的 `jittor._core.api.__all__`。组合模块从
`_core.var`、`module`、`function`、`hooks`、`flags`、`diagnostics` 导入规范对象；
**实现不再共用一个巨大的 `core_api.py` 命名空间**。原生 save/load 与安全 pickle 算法属于
`serialization.native`，由同一层组合再导出。`jittor._core.module` 拥有原生 Module 实现。
公开的根导出保持对象标识，历史的根级 pickle 路径仍可加载。历史的
`jittor._runtime.core_api` 导入解析到与 `jittor._core.api` **同一个模块对象**，没有第二份
实现。

`_core/__init__.py` **不导出**名为 `var`、`flags`、`hooks` 的可调用对象或状态对象——这些
包属性必须继续解析到它们各自的模块；对象级导出属于 `_core.api` 和公开的 Jittor 根。
API 组合在导入其实现域之后安装退出钩子，保持既有的注册顺序。`__init__.pyi` 拥有公开根的
类型标注面。`_core.flags` 构造唯一的原生 Flags 对象、它的运行时上下文和运行时作用域
API；原生与 Torch 两种组合都保留**同一个对象**。

### 配置、运行时开关与只读内省

`_runtime.flag_policy` 为绑定生成和 Python API 两侧分类原生 flag。`_runtime.state` 提供
不可变的 `jt.config`、可写的 `jt.runtime` 开关和只读的 `jt.runtime.context` 诊断。运行时
写入调用**原始的原生 setter**，保留其副作用；`jt.runtime.scope(...)` 使用既有的可重入
flag 作用域实现。快照包含分离出来的 Python 值，包括映射值的拷贝。执行与分配器计数器
通过 runtime 和旧的 Flags 对象都是只读的。

`jt.introspection` 提供三层受支持的只读观察：**capabilities** 查询具名的后端/设备注册表
与既有的库证据；**policy** 视图转发启动配置与生效的运行时设置；**counters** 观察执行器、
分配器和原生图的存活服务。**不存在第二份 Runtime 状态、隐式的可选库加载、图冲刷或可写
的观察路径。**

启动配置包括编译器/工具路径、编译选项、缓存/源码路径、CUDA 架构和缓存锁策略。在后端
后处理与兼容组合完成之后，一次**单向的原生封印**会拒绝经由任何 `Flags` 实例的写入，
包括 `jt.flags`、`compiler.flags` 和 `core.Flags()`；编译器模块也拒绝对这些字段的迟到
公开赋值。`jt.config` 捕获一份分离的不可变快照，架构列表变成元组。**启动选项要在导入
Jittor 之前通过环境设置。** 新算子仍然接受局部的 `extra_flags` 和逐算子的
`compile_options`；加载一个扩展**不会重新打开启动配置**。分类文件参与绑定生成器的构建
指纹。

### 原生状态与生命周期

原生的 held-root 存储在 `src/runtime/holder_state.{h,cc}`。`RuntimeHolderState` 同时拥有
holder 列表和弱同步游标；执行器、自动微分、图检查和内存诊断共享它导出的核心访问器。
**注册或移除一个 holder 时它绝不运行图。** 弱同步先 peek 再检查目标截止点，并且只在该
检查之后前进。移除操作在存活性释放可能重入运行时之前先修复游标。该 owner 不可拷贝、
具有进程生命期，以支持迟到的扩展/静态 holder 析构；它**不拥有**所指向的 holder。这保持
的是既有的"变更需序列化"要求，**不是新的线程安全保证**。

`src/runtime/runtime.{h,cc}` 拥有进程生命期的 `NativeRuntime`，内含执行器、held-root、
遍历与设备状态。`runtime_executor()` 与 `runtime_holder_state()` 从核心、JIT 算子和后端库
解析到**同一个 owner**。执行器以空分配器指针启动；**构造过程不选择设备、不初始化后端。**
fork 初始化重置其既有设备状态，不重建继承来的 holder。`runtime_traversal_state()` 在核心
与 JIT 库之间共享戳记计数器和活跃 epoch 计数。`TraversalEpoch` 位于 `src/runtime/`；嵌套的
遍历标记在离开 epoch 前恢复，**包括异常展开时**。fork 保留计数器以避免与继承的节点戳记
冲突。

`TraversalQueueMarks` 在同一遍历 owner 中区分“已入队”和“存储别名环已展开”，避免把
这两个状态当成嵌套遍历而为每个节点分配恢复记录。真实嵌套仍恢复外层戳记；该 owner
只用于同步遍历，不缓存读者集合，也不省略写入前的别名和已保存激活检查。
一次读者遍历内，若多输出算子没有待计算的输出或待展开的共享关系，可复用这次空扫描的结果，避免
融合优化器的每个状态输入都重复检查数百个输出；该缓存不跨遍历或训练步保留。

`runtime/device_state.h` 存放 `use_cuda`、`device_id`、`sync_run`、缓存的设备数/当前设备、
设备切换钩子和 peer access 记账。`runtime/device.{h,cc}` 提供设备操作；旧的
`misc/cuda_flags.*` 已删除。`DEFINE_RUNTIME_FLAG` 通过导出的存储访问器注册 Python/环境
访问，**而不是全局变量或动态初始化的全局引用**。setter 的纠正、回滚和后端切换冲刷保持
原有顺序。仅 CPU 的算子路由保持常量 CPU，而 flag 绑定仍读取真实的运行时状态。ROCm 的
回调选择在头文件中显式给出，不再依赖旧的二进制转换器识别旧文件名。上面未列为运行时
所有的领域专属原生 flag 仍留在各自的模块里——**Python 侧的生命周期划分并不声称搬走了
所有 C++ 存储**。

`runtime/jit_policy` 拥有 CUDA kernel 的数学策略（`default`、`strict`、`backend`）。普通与
融合的 CUDA key 在编译前捕获策略，编译器按捕获到的值转换启动 flag。切换策略会先按**旧
策略**提交待定图。显式的逐算子 flag 仍是局部覆盖。Torch 预检选择的是这个运行时策略，
而不是改写启动 NVCC flag 或仅仅为了改变 kernel 数学而创建另一份核心构建配置。ACL 预检
仍会从其启动环境中移除历史的 CUDA 专属 strict flag。

原先导出的 `Executor exe` 数据符号已删除。树内 CUDA/ACL 消费者与内嵌 CUDA 模板改用
`executor.h` 的 `runtime_executor()`。**树外的 C++ 扩展必须更新该访问方式并针对新的
头文件/核心库重新构建；旧的预编译扩展二进制不兼容。** `tflag_count` 符号与
`misc/traversal_epoch.h` 路径同样删除；使用遍历内部的扩展必须改用 runtime 头文件并重建。
`use_cuda`、`device_id`、`sync_run` 数据符号也已删除——外部原生消费者必须 include
`runtime/device.h` 或 `runtime/device_state.h` 并重建；**Python 的 `jt.flags` 名字不变**。

### 领域包与构建工具

编译器、外部库设置、绑定生成和 CuPy 引导的实现都在 `build/` 下；历史的根模块导入是同
对象别名。`distributions/`、`init/`、`linalg/` 是公开的原生包；`selftest.py` 是安装后的
冒烟测试入口。**新增根级文件需要一次归属评审和相应的结构门禁更新。**

这三个领域的初始化文件显式再导出其实现对象。线性代数把复数例程、分解、求解、范数和缩并
分开，并共享数组助手与结果类型。分布把基础契约、约束、助手、离散/连续/松弛/多元族和
KL 散度分开。初始化把基础填充、fan/gain 规则、带缩放的初始化器和截断正态分开，其门面
保留既有的 Var 方法绑定。函数元数据指名物理归属者，而历史的公开 pickle 全局符号继续
经由门面解析。`python/jittor` 下所有 `.py` 文件现在都在 1,500 行以内。原先庞大的兼容 NN、
数值、张量和 FlashAttention 模块已变成有独立实现归属者的常规包。**这次源码分解并不等于
独立 Torch 架构迁移已经完成。**

仅在运行时需要的框架导入被推迟到调用处，以免导入循环面继续扩大。六个历史的复数线代
函数被惰性再导出为它们原来的对象，从而保留具体的 ComplexNumber 类型标注，又不让包引导
依赖 NN 门面。

贡献算法物理上位于 `contrib/{ccl,loss3d,math_util,einops}`。它们的历史包与子模块路径是
**按需解析的**弃用同对象别名，而不是急切导入全部四个域。`igamma.h` 资源随其归属者进入
`contrib/math_util/src`；源码 checkout 与安装后的资源查找使用**同一条模块相对路径**。
公开的 `contrib` 包同时拥有历史的组合助手；`jittor.compat.contrib` 现在是它的别名。别名
发布保留同名的包函数（如 `math_util.igamma`、`ccl.ccl_2d`），**而不是用子模块对象替换
它们**。einops 的解析与变换共用 `einops._errors` 中的同一个 `EinopsError` 类，由公开包
再导出，不产生"实现到门面"的导入循环。

独立的构建工具物理上在 `build/utils`，同时保留 `jittor_utils` 这个运行时命名空间，使工具
导入可以在 Jittor 核心引导**之前**运行。`jittor.build.utils` 的导入是同一批对象的别名。
这把物理归属与独立引导命名空间分开，既没有引入重复的工具实现，也没有对 Jittor 的急切
反向导入。源码与依赖扫描器检查的是真实的 `build/utils` 树，不只是那座独立导入桥。

编译器按职责拆分：`build/compiler.py` 保留启动状态与编排，`build/codegen.py` 拥有源码
生成，`build/compilation.py` 拥有编译与自定义扩展操作。**生成器指纹包含被抽出的归属者**，
因此一次改动不会静默复用过时的构建戳。

后端配置现在是由选定 provider 返回的**冻结** `BuildConfig`，显式服务放在 `BuildContext`。
**Provider 不修改编译器全局变量，也不向其源码列表追加内容**；引导过程集中发布兼容属性。
entry point 只为选中的后端加载，显式选择 CPU 会跳过 CUDA 发现。构建工具通过注入获得其
绑定/编译器服务，**不再导入 Jittor**。张量 checkpoint 算法在 `jittor.serialization`，
原生 save/load 与安全 pickle 在 `serialization.native`；历史工具路径在引导后查询运行时
注入的 loader。

## 原生支持层布局

`runtime/backend*` 中的原生 `BackendRegistry` 由 `NativeRuntime` 拥有。它发布带版本校验的
回调表，名字由其自有、回调具进程生命期。**注册 CPU 与 CUDA 描述符不会初始化驱动**：
纯 CPU 构建仍然"知道" CUDA，但报告设备数为零。`runtime/backends/` 实现原始池选择、设备
操作、拷贝、同步与流。公开的分配器代码保留 SFRL/NFEF/Temp/Stat 组合，并从注册表获取原始
池——**方向绝不反过来**。

数组创建、主机/设备迁移、device-copy 算子、fetch 与换出传输都调用该接口。
`allocation_device()` 从分配器推导物理设备；**它不会把一个 Var 保留的设备亲和性误当成
它当前的驻留位置**。双重暂存与延迟释放的存储报告其真实的池设备。有序的 peer 拷贝同时保留
源与目的的流依赖，fetch 通过其回调持有块。旧的 CUDA 流函数保留为已注册流钩子的适配器，
共享实现在 `src/runtime/backend_streams.cc`，SDK 操作在对应的后端驱动中。

`core.registered_backends()` 与 `core.backend_device_count(name)` 查询原生注册表。`jt.flags`
中四个历史加速器模式别名会发出 `DeprecationWarning`，但保留其 setter 行为。ACL 发布规范
名字 `acl`；旧拼写 `acl_legacy` 解析到**同一个**原生描述符和 Python kernel 表，不注册重复
实现。Python 侧的选择消费原生设备上下文；**不存在独立的 Python 分配器或写死的后端能力
原型**。

## 分布式归属

`jittor.distributed.process_group` 拥有 `ProcessGroup`、`Work`、通信器创建以及实时的原生
rank/world 查询。启动与 rendezvous 由 `distributed.launch` 和 `distributed.store` 拥有。
Torch 安装器委托给这些对象，只保留 Torch 拼写、参数适配和它自己的安装/引导事务。历史的
`_JittorProcessGroup` 与 `_JittorWork` 导入仍是别名，因此旧的 pickle 全局符号解析到同一批
规范类。**非法的原生 world-size 元数据现在报错**，不再被静默当成单进程运行时。

## 动态形状与提交边界

算子构造函数与 `Op::init()` 建立边、推断形状元数据并捕获 autograd 策略，**不提交工作，
也不调用回调**。负的维度记录数据相关输出的分配上界，由算子执行时替换为实际的非负形状。

- `schedule_pending_from_python()` 在一个完整的 Python holder 创建之后运行；
  `Executor::submit_pending()` 在那里解析动态结果所需的子图，保持现有的"Python 看到具体形状"
  契约。普通静态结果沿用配置好的惰性/即时提交策略。
- 原生 C++ 组合不需要中间 Python holder。`Executor::run_sync()` 在构造执行计划之前按依赖顺序
  解析内部的动态输入，再刷新下游的 `infer_shape()` 元数据——这是执行期操作，**绝不是构造
  期的递归**。显式提交后仍未解析的动态输出报错，而不是发布一个负的具体形状。
- `VarHolder::item()` 用 `sync(false, false)`：提交所请求的依赖图，不做扩散到兄弟分支的弱
  同步。CUDA 设备到主机的拷贝使用生产者流上的事件；原生 where、candidate 与 CUB where 的
  计数读回走同一个 `backend_copy()` 网关，分配器与 GIL 策略仍归该网关与执行器入口作用域。
- fetch 的构造只创建 fetch 节点。公开的 Python `fetch()` 在构造之后调用
  `core.submit_pending_fetches()`，执行就绪的输入并施加有界的挂起 fetch 反压；原生调用方可以
  用同一个显式边界，也可以正常提交自己的 fetch 图。关闭期的设备等待属于清理，不受此契约影响。

`jt.submit_pending(*vars, device_sync=False)` 是**显式提交选定挂起根**的边界：每个参数必须是
`Var`，它的挂起生产者图经已有的 `Var.submit_pending` 绑定提交；其它 holder 根保持挂起，正常
的惰性/自动 flush 策略不变。`device_sync=True` 时再同步这些根以便主机立即消费。返回唯一的根
或根的 tuple，保持对象标识；空输入与非 Var 参数抛清晰的 Python 错误。它面向确切知道自己
输出根的 Function 回调、fetch 桥与执行适配器；**不做隐式全局 flush、不调 GC，也不让无关的
挂起图变得可执行**。实现委托给已有的执行器部分提交路径，不拥有第二张图、队列、liveness
计数或调度策略，CPU 与加速器用同一个边界。

并行开发时守住这些归属：

- 新的动态算子实现上界输出元数据、纯的 `infer_shape()` 与执行期的最终形状发布；**不要**
  恢复构造期调用 `run_sync()` 的变通写法。
- 前端只在其语言对象与图边都存在之后才提交，不复制执行器的形状依赖遍历。
- 后端 provider 在运行时回调背后实现有序拷贝与事件等待；算子族不引入整设备的读回等待。
- 构造、`submit_pending`、形状解析与 fetch 反压的改动是**一次核心接口改动**，要一起协调；
  普通的算子或前端新增只消费这些接口。

定向回归：`tests/core/test_dynamic_shape_submission.py`、`tests/core/test_partial_graph_submit.py`、
`tests/ops/test_where_op.py`、`tests/data/test_fetcher.py`，以及
`tests/structure/core/test_var_holder_submission.py`。
