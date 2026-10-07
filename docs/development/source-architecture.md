# 源码架构与模块边界

- 状态：已接受
- 上次复查：2026-10-05（并入整改期的后端构建、动态形状、dtype 与前端运行时状态契约）
- 基线：`e3c369acb`
- Owner：Jittor 核心维护者
- 复查触发：公开模块搬动、新增实现域、或运行时资源路径变化时

本文定义 Jittor 内部 Python 源码的分解方式。仓库、打包与运行时资源的归属由更上层的
[仓库布局决定](repository-layout.md)定义。

## 原则

1. **一个物理归属者。** 一个公开领域就是一个常规包；它**不再有第二棵藏着真实实现的
   私有树**。
2. **导入跟随归属。** 实现的元数据与回溯信息报告**拥有该代码的那个模块**。再导出模块
   不递归改写 `__module__`。
3. **组合保持浅层。** 包的 `__init__.py` 负责组合与导出公开名字；大块实现住在有意义的
   子模块里。
4. **运行时路径是契约。** 编译器加载的 C++/CUDA 资源只能随着显式的编译器与打包迁移一起
   移动。
5. **兼容是分层的。** 框架原生能力、可复用的兼容机制、导入 shim 和下游集成，各有不同的
   归属者。
6. **每次搬动都保持行为。** 重构要保留公开名字、承诺过的可调用对象标识、支持范围内的
   pickle 行为、后端分派，以及定向回归覆盖。

## 当前的领域划分

```text
python/
├── jittor/
│   ├── __init__.py              # 根组合与运行时初始化
│   ├── __init__.pyi             # 公开根的类型标注面
│   ├── _core/                   # 原生 Python API 的实现域
│   │   ├── api.py               # 编译核心引导之后的显式组合
│   │   ├── var.py               # 张量构造、运算与 Var 绑定
│   │   ├── module.py            # Module 与参数/缓冲归属
│   │   ├── function.py          # 自定义自动微分上下文与梯度钩子
│   │   ├── hooks.py             # 可移除句柄与钩子支持
│   │   ├── flags.py             # 原生作用域与共享运行时状态
│   │   ├── arg_policy.py        # 显式的不支持/忽略参数策略
│   │   └── diagnostics.py       # 日志、性能分析、进程作用域与退出清理
│   ├── _runtime/
│   │   ├── core_api.py          # _core.api 的同对象历史别名
│   │   ├── composition.py       # 显式的命名空间发布
│   │   ├── install_order.py     # 有序的安装器注册与校验
│   │   └── state.py             # 注入的原生 Flags 视图，不做引导期导入
│   ├── serialization/
│   │   └── native.py            # 原生 save/load 与安全 pickle 实现
│   ├── build/                   # 编译器/引导的实现归属
│   │   ├── compiler.py          # 启动状态与构建编排
│   │   ├── codegen.py           # 原生绑定/注册的源码生成
│   │   ├── compilation.py       # 编译与自定义扩展操作
│   │   └── utils/               # 独立工具，以 jittor_utils 名义导入
│   ├── contrib/                 # 贡献算法与组合助手
│   │   ├── ccl/                 # 连通域标记
│   │   ├── loss3d/              # Chamfer 与推土机损失
│   │   ├── math_util/           # gamma 函数与共享原生资源
│   │   └── einops/              # 内置的张量表达式算法
│   ├── nn/                      # 神经网络公开 API
│   │   ├── modules/             # 有状态的 Module 实现
│   │   ├── functional/          # 无状态的张量函数
│   │   ├── backends/            # cuDNN 与只读的钩子调用适配
│   │   ├── utils/               # weight norm 等构造助手
│   │   └── attention.py
│   ├── autograd/                # 函数式自动微分
│   ├── fft/                     # 可导的原生 FFT 命名空间
│   ├── linalg/                  # 分解、求解、范数与缩并
│   ├── distributions/           # 概率分布族与共享约束
│   ├── init/                    # 初始化族与共享的 fan/gain 规则
│   ├── ops/                     # 张量、索引、归约与形状实现
│   ├── misc/                    # 已弃用的同对象张量 API 门面
│   ├── pool/                    # 已弃用的同对象池化 API 门面
│   ├── optim/                   # 优化器门面与算法模块
│   ├── sparse/                  # COO 稀疏张量与稀疏卷积
│   ├── compat/
│   │   ├── torch/               # 规范的 Torch 风格 API 兼容
│   │   ├── fsdp2/               # 分布式 FSDP2 兼容
│   │   ├── triton/              # Triton API 桥接与部署命令
│   │   ├── shim/                # Torch shim 运行时与部署命令
│   │   ├── module_patcher.py
│   │   └── external_backend.py
│   ├── selftest.py              # 安装后的冒烟测试
│   ├── tools/                   # 用户工具，含 benchmarking.py
│   ├── backends/                # 源码 checkout 的包路径桥接
│   └── distributed/             # 原生启动、rendezvous 与通信助手
└── jittor_utils/                # 通向 jittor/build/utils 的独立导入桥
```

## 包组合契约

### 神经网络 API

`jittor.nn` 是公开包。有状态的层在 `nn.modules`，无状态运算在 `nn.functional`，可选的
加速路径在 `jittor.backends.cuda.kernels`。`nn.backends` 保留的是**调用适配器，不是 CUDA
实现**。公开的再导出必须指向规范实现对象；只有当包装器确实在强制某条真实的 API 契约时
才成立。

依赖方向是：

```text
nn.modules -> nn.functional -> Jittor 张量/核心运算
nn.backends ----------------> 显式的后端/编译器接口
```

**functional 模块不得导入有状态的层实现。** 后端适配器必须保持可选，并在其工具链不可用
时以**可执行的能力错误**失败。

### 杂项与池化 API

`jittor.ops` 拥有张量、形状、索引与组合运算。历史的 `jittor.misc` 导入仍是**已弃用的
同对象门面**，包括旧的子模块路径。公开名字与 pickle 全局符号解析到规范实现；
**`misc` 下不存在第二棵可编辑的数学实现树。**

池化的数学与参数校验在常规的 `nn.functional.pooling` 包里，按平均池化、2D/3D 核心、
自适应、1D 和反池化分成不同归属者。`nn.modules.pooling` 只保存构造参数并调用这些无状态
实现；**functional 调用不会构造临时 Module**。CPU/CUDA 的生成源码保持单一来源，后端发射
构造器停在既有的注册边界上。

`jittor.pool` 及其历史子模块是**已弃用的再导出**。`nn.modules.pooling_legacy` 以原类名
保留三个历史类：`AvgPool2d` 和 `AvgPool3d` 为旧 pickle 保留其转发层状态，而
`AdaptiveAvgPool2d` 保留其固定窗口算法——**后者刻意不同于当前 NN 的重叠分箱算法；
一次布局搬动不会静默改变它的数值规则。** 自适应窗口的中间量是函数局部变量，不是持久的
Module 状态。

历史开关 `pool_use_code_op` 的唯一归属者是 `nn.functional.pooling._state`。`jt.pool`、
`jt.nn` 和 functional 池化包暴露的是对**同一个值**的实时读写，包括临时的属性覆盖/还原。
**后端执行绝不读取被复制过的门面值。** NN 在加载旧门面之前先构造它的 functional 与
module 归属者，因此这条兼容导入不会造成引导循环。

`jittor.sparse` 在不同子模块中分别拥有坐标格式稀疏张量和稀疏神经网络 kernel。历史名字
`jittor.nn.sparse` 是 `jittor.sparse.convolution` 的同对象别名；`jittor.nn` 与
`jittor.nn.functional` 再导出那些规范可调用对象。

`jittor.autograd` 拥有函数式自动微分。`jittor.fft` 拥有原生 Jittor 与 Torch 模式共享的
可导 FFT/shift/frequency 命名空间。拼接与索引在 `jittor.ops`，池化在 `jittor.nn`，
优化后的 softmax 在 `jittor.backends.cuda.kernels.nn`，权重归一化在 `jittor.nn.utils`。
**历史的根级拼写只是导入别名**，不保留物理源文件或包装实现。

### 根模块归属

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

### CUDA 资源布局

checkout 的 `backends/cuda/` 是 CUDA 的物理归属者。库算子在 `kernels/<library>`，库包装器
与头文件在 `libraries/<library>/{src,include}`，通用支持在 `src` 与 `include`。原生的
索引/where/candidate/转置实现在 `kernels/core`，诊断 kernel 在 `kernels/debug`。Python 侧
的 CUDA 实现与源码构造器在 `kernels/{nn,misc,math,pooling,sparse,ccl,loss3d}`。ACL 的 KV
实现在 `backends/acl/kernels`。

Python 导入在 checkout 与 wheel 中都使用规范的 `jittor.backends` 命名空间。一条仅限源码的
路径桥接加上显式的 package-directory 映射，避免在 `python/` 下出现第二棵可编辑实现树。
旧的 NN 模块拼写保持同对象别名。**`nn/` 里没有 CUDA kernel 模块，也没有 ACL KV 模块**；
`nn/backends` 只含它的初始化文件、cuDNN 适配器和钩子视图。

共享的索引与池化数学保持单一来源。注册生成器把后端索引片段与共享源码组合成一份**原子
发布、内容稳定**的 JIT 源码，并保留每一段的 `#line` 映射。纯 CPU 构建仍然编译那份纯主机
的循环调度助手。历史的 `type/cuda_atomic.h` include 转发到后端拥有的头文件，其中不再含有
CUDA 实现。

资源查找区分源码 checkout 与已安装包。**整棵树的源码转换机制已被移除。** 一个只含
`__pycache__` 的目录不能遮蔽真正的源码归属者。打包测试检查 sdist、wheel 和隔离安装中的
**每一个**后端文件。

共享 C++ 核心在顶层 `src`（打包为 `jittor/src`），MPI/NCCL/HCCL 包装器在 `backends/comm`。
`python/jittor/src` 与 `python/jittor/extern` **在 checkout 中均已不存在**。外部
FlashAttention 集成保留其独立的兼容包迁移。`fused_adamw` 没有可搬迁的 CUDA 算法——它既有
的 ACL 实现与共享错误入口**不构成 CUDA 支持**。

### 原生支持层布局

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

### 原生算子分派

`ops/op_register` 发布不可变的 `OpDef` 对象，带进程内稳定的 `OpId`。**已实例化的图钉住它
的定义**；替换注册表条目影响新图，不影响活跃的图。每个定义把按后端索引的 `Kernel` 回调与
一个 `Codegen` 接口（源码片段、准备、优化、源码元数据）组合起来。形状与梯度语义仍在图
算子上。替换以及注销/重注册会获得唯一的编译标识；普通与融合 key 包含其定义及融合子项的
标识。首次注册保持稳定的磁盘缓存 key。执行器、并行编译器、tracer 和 relay 使用这些已注册
接口，包括一个保留其上下文与 relay 缓存的专用融合实现。**旧的虚执行方法只是源码适配器，
不是执行器的分派路径。**

生成的核心与扩展注册用 `register_op_definition<T>` 绑定具体算子实现。CUDA 库注册仅限
加速器的 kernel，MKL 注册 CPU kernel，核心的仅 CPU 算子声明其后端掩码。
`core.backend_supported_ops(name)` 枚举这些已注册实现；**个别的形状/dtype 限制仍然适用**。
缺失的实现**抛出异常**，而不是落进一个空的虚 `run()`。

可选库在自己的翻译单元中发布带类型的语义能力。核心的替换点和 matmul/conv 调优器查询这些
能力，**不点名** CUB、cuRAND、cuTT、cuBLAS、cuDNN 或 MKL 实现。能力查找在使用时解析构造器，
因此**库加载之前的一次查找不会把"未命中"永久缓存**。`core.backend_supported_capabilities(name)`
暴露可用的语义族。后端选择先于源码片段生成，包括双源 CodeOp 的缓存查找。

**原生扩展必须重建**：`Op` 布局变了，`OpInfo` 现在是 `OpDef` 的兼容别名，源码元数据位于
`definition.codegen` 之下。已转换的 ACL/ROCm/Corex 后端保留其历史构建路径。**仅主机的语法
检查不是 CANN ABI 或设备验证**；那些机器必须先构建并执行改动后的后端，才能声称硬件支持。

`jittor_utils.compile_module` 把生成的包装器与参数打印器定义作为**一个翻译单元**编译。
此前两个编译器输入会互相覆盖同一个 depfile，漏掉扩展头文件并静默复用过时的 ABI 布局。
改动后的命令会自动使旧缓存项失效；之后的头文件变化会被正确跟踪，无需重命名扩展或删缓存。
**已加载的扩展模块仍然需要一个新进程。**

### Python kernel 分派

`_runtime.dispatch` 拥有 Python kernel 注册。每个条目声明它的算子、后端、接受的张量 dtype、
形状/梯度谓词和优先级。`select_kernel` 返回真正的实现；`try_dispatch` 与 `optional_kernel`
适配器使用同一套选择逻辑。**未命中只允许一个显式的同设备通用实现，不允许隐式挪到 CPU。**
谓词与实现的错误向上传播，**不会去试另一个 kernel**。历史的模式专属限制变成注册限定词，
而不是调用点上的独立后端守卫。

`core.dispatch_context(inputs)` 返回运行时目标与由输入决定的设备，**不实体化张量、不探测
驱动**。Python 递归收集位置与关键字容器中的 Var，并记录它们全部的 dtype。原生查询检查混合
设备输入，保留有界的待定标量重定向规则。**存储驻留位置本身不是执行策略**：待定和主机暂存
的输入仍然可以以加速器为目标。按设备索引的 FFT 与注意力缓存使用这个上下文，含非默认设备
编号。

CUDA/历史库适配器、矩阵/卷积/RNN 选择、归一化/推理、索引/扫描等原生领域都使用这张表。
非 ACL 的已转换 CUDA 实现在旧守卫允许处保留其显式声明的 ROCm/Corex 注册——**这不构成对
那些设备的认证**。旧的 `_runtime.registry` 原型及其 bytearray 分配器已删除。根级的
flatten/clamp/outer 现在在同一张表里注册可移植实现。

`_runtime.backend_libraries` 拥有已加载模块、由其派生的 `.ops`、资源、loader 回调和可用性
策略。**未命中的查询不会被永久缓存**；显式加载会传播错误。MKL 的禁用在缓存模块查找与加载
**之前**检查，重新启用可以复用已加载模块。`compile_extern.*` 与根级库属性仍是动态只读查询，
不是可变快照。既有的引导顺序保持不变；完全惰性的核心导入是另一项独立任务。

`nn.backends.hooks` 是这张表的只读兼容视图。ACL provider 直接发布实现，该视图**绝不存储
独立回调**。内部测试用 `override_kernel` 做带作用域的替换或缺席，并在退出时恢复先前注册。
**直接给历史钩子/库属性赋值会被拒绝。** ACL 源码转换器仍是一项独立迁移；Python kernel
发布不再替换原生公开 API。

### ACL kernel 注册

ACL 的构建入口是 `jittor.backends.acl`。SDK 支持翻译单元在 `backends/acl/src`，面向 SDK 的
头文件在 `include/{aclops,aclnn}`，原生算子翻译单元在 `kernels/native`。provider 使用一份
显式的 45 文件构建清单（42 核心 + 3 注册），保持原有顺序，**不会误用通配把独立后端和
workspace 运行时源码卷进来**。源码构造器经由后端 include 根包含 `aclops/aclops.h`；真正的
SDK `acl/acl.h` 引用保持不变。

MPI、NCCL、HCCL 资源在 `backends/comm/{mpi,nccl,hccl}` 下，各有对应的 `inc`、`src`、`ops`
目录，NCCL 另外拥有它的 no-MPI 头文件。编译器查找在 checkout 与 wheel 中都用
`backend_root(..., "comm")`；已删除的 `python/jittor/extern` 路径**既不是 include 根也不是
包输入**。

`backends/acl/kernels/install.py` 在既有的 Python 分派表中发布模块级的张量、神经网络和
归一化实现。配套的 SDK 源码构造器在 `backends/acl/kernels/ops`；旧的 Python 模块路径是同
对象别名。原生函数与 Module 类保留各自的标识、校验、参数管理和通用数学。**不存在
`change_function` 或 `warp` 式的安装器。** 不支持的 Python 变体向其同设备的通用归属者
返回 `None`；执行错误向上传播。

原生注册表在定义发布时组合 ACL 的 `OpImplementation` 值，**包括初始化之后才加载的扩展**。
安装组合器时也会**事务性地**处理已有定义。定义保持不可变，已有的图保留其钉住的值。一个
稳定的启动版本保持跨进程的 JIT key；动态的实现替换仍会获得唯一标识。**一个迟到的扩展
不能仅仅靠避开初始化期扫描就获得 ACL 支持。**

`Kernel.compile` 取代了全局编译钩子。ACL 分别注册融合、映射、原语和显式不支持这几条路径；
融合 relay 的选择仍然跟随调优。HCCL 这类 SDK 原生扩展显式声明它们的编译器。**不支持的
回退条目不会表现为已实现的算子或能力**，其它后端的构造器也不会被删除。

`jt.code(..., backend="acl")` 把加速器源码标识为 ACL SDK 代码。**它不改变当前设备。**
构造器校验该标记，缓存 key 包含它，普通与多输出梯度继承它，错误的加速器目标在执行前就被
拒绝。`cpu_src` 保持独立。依赖 `// aclop` 或其它偶然出现的 `acl` 子串的第三方代码**必须
补上该标记**——基于注释的识别被刻意移除了。

带类型的原生 Getitem/Setitem 条目覆盖基本的正步长切片、整数索引、新轴、省略号、空选择和
广播赋值。一份纯粹的、经过检查的地址计划会合并连续的后缀；设备拷贝排在既有的 ACL 计算流
上。**没有张量数据经由 CPU 暂存。** 只有完全相同的共享映射才是 no-op；不安全的重叠、
高级/字符串索引、负步长和原生归约赋值**由该入口显式声明不支持**。既有的 Python ACL
构造器保留其独立变体。标量广播拷贝是保守的，**不构成性能主张**。原生整数视图保留链式
回写所需的生产者；基本索引的梯度让赋值转换显式化，同时不改变 indexed-add 的累加。基本的
索引赋值现在遵循原生 `VarView` 记录，此前 Python 侧的父链回写已移除。**这并不意味着每一种
高级索引或跨后端视图变体都有硬件覆盖。**

ACL 后处理只发布它的原生实现。它对锁页主机内存、编译器并发和归约的要求属于**后端描述符**，
由分配器、编译器和归约的归属者消费；**公开 flag 不会被覆盖**。BackendOps ABI 4 拒绝更旧的
描述符，扩展必须重建。历史的整树 SDK 转换（`process_acl`、`process_jittor_source`）已经
消失；**真实的 CANN/NPU 验证仍然是必需的**。

### 分布式归属

`jittor.distributed.process_group` 拥有 `ProcessGroup`、`Work`、通信器创建以及实时的原生
rank/world 查询。启动与 rendezvous 由 `distributed.launch` 和 `distributed.store` 拥有。
Torch 安装器委托给这些对象，只保留 Torch 拼写、参数适配和它自己的安装/引导事务。历史的
`_JittorProcessGroup` 与 `_JittorWork` 导入仍是别名，因此旧的 pickle 全局符号解析到同一批
规范类。**非法的原生 world-size 元数据现在报错**，不再被静默当成单进程运行时。

### 后端回退策略

`NativeRuntime` 拥有 `backend_fallback`，经由 `jt.flags` 与 `jt.runtime` 暴露。默认是
`warn`；`error` 拒绝自动的跨后端计算，`allow` 允许且不警告。**非法赋值保持原策略不变。**
执行器在迁移输入之前检查仅 CPU 的执行；数组暂存、fetch 和显式设备传输**不是计算性回退**，
在所请求设备上的通用 kernel **也不是**跨后端回退。

历史的 ACL 执行器在执行前对完整的融合组或独立运算做预检。**只有显式不支持的运算/变体才能
请求 CPU 回退。** SDK、形状和 kernel 执行失败会清理并传播其原始异常——**它们不是路由信号**。
一次被允许的回退即使 CPU 执行失败，也会恢复先前的执行模式、算子 flag 和融合上下文。族内的
SDK 资源清理仍有单独跟踪的工作；**仅主机的测试不构成 NPU 硬件支持**。

`core.backend_fallback_count()` 统计跨后端**决策**（含被拒绝的尝试），而不是已完成的 CPU
计算。硬件门禁使用 `error`。`_runtime.fallback.forbid_backend_fallbacks()` 还会在正常返回后
检查计数增量，从而发现那些异常被调用方吞掉的尝试。它保留主异常，并且**不隐式同步**：
调用方必须在作用域内执行并同步该工作。NPU 的 pytest fixture 与独立的生态 runner 使用这个
接口，**而不是去解析日志措辞**。

C++ 的 `src/misc/` 目录已不存在，支持代码按其实际角色分组。**这是源码布局变化，不是助手
算法的改变，也不表示后端注册表迁移已完成。**

| 归属 | 支持代码 |
| --- | --- |
| `src/debug/` | CPU/CUDA 的 NaN 检查与诊断 |
| `src/runtime/` | 设备流、float32 精度策略、遍历索引、RingBuffer、集合通信 dtype 与 rendezvous 助手 |
| `src/type/` | Nano 类型与标量数学、原子操作、intrinsic 和生成 kernel 使用的数值极限助手 |
| `src/utils/` | 通用字符串、哈希、容器、共享指针与清理助手 |
| `src/third_party/` | 内置的 miniz |

生成的 include 与后端源码转换都使用这些路径。**此前 include `misc/...` 的源码扩展必须先
更新 include 再重建。** 基名保持不变，因此既有的 ROCm/Corex 转换规则保留其分派标识。搬动
这些支持文件**并不等于**完成了另外的 `init`/profiler/锁或 Python 绑定的布局迁移。

在复用被转换过的源码缓存时，原始源码树中不存在的原生文件会在编译前被移出 `src/` 和
`extern/`，**保存**在缓存目录的 `<backend>_source_stale_*` 下而不是删除。这防止源码搬动之后
新旧翻译单元被一起编译。非原生的缓存产物不受影响。

### 兼容 API

独立激活在安装之前先构造它的 TorchNamespace。安装器把 API 与上下文发布到那个目标上；
一次事务性的"后端到归属者"绑定把 leaf、retained-gradient 和优化器记账路由到**同一份状态**。
历史的原生属性是该状态的别名，被回滚的安装步骤在重试时**针对同一份状态重放**。

TorchNamespace 拥有它自己的公开写入与删除。缺失的读取仍可使用它的原生归属者；删除会在
局部遮蔽这条回退。事务回滚恢复**确切的**局部绑定与删除状态。InstallContext 把安装目标与其
原生后端分开，**绝不通过命名空间回退继承安装标记**。发布过程让根的自别名在注册表与导入
映射中保持一致。独立激活拥有真正的 Tensor/Parameter 子类和 Module/NN 适配器，复用原生的
Var/Op 图与数学，**而不把这些 API 安装到原生类上**。状态查找使用私有的弱模块绑定并带事务
回滚；独立激活**不在原生模块上发布** leaf/retained/optimizer 状态别名，也不通过归属者查找
让一个已卸载的前端活着。

规范的 Torch 风格实现是 `jittor.compat.torch`。历史的属性/模块拼写 `jittor.torch_compat`
在显式导入时加载其可选的别名 provider，**它不是第二份源文件**。同理，规范的 Triton 实现是
`jittor.compat.triton`，`jittor.triton_shim` 作为对象标识别名保留。

NN、数值与张量 API 的兼容安装器以及 FlashAttention 适配器都是按实现族拆分的常规包。没有
安装状态捕获的公开可调用对象可以保留模块级标识；**有状态的安装路径保留显式上下文和其原有
注册顺序**。原生 Parameter 是真正的 Var 子类；原生 Module 按属性名注册参数与缓冲，
**不读写 Torch 角色标记**。Torch 专属的赋值规则住在独立 Module 适配器或显式的历史安装器里。
物理的 `compat/` 树现在就是独立的 `jittor-torch` 发行物，**核心打包排除它**。显式的历史模式
仍然改造原生类，**不得与独立激活混为一谈**。

基本索引使用原生 `VarView` 跟踪，而不是平行的 Python `_torch_index_parent`/切片链。尚无原生
视图记录的 Torch 专属切片形式会用 `_set_view_of` 显式挂上一个。分离的 `.data` API 保留其
独立的归属者/路径记账；赋值使用一个分离的右值节点，**这样停止 data 别名不会冻结它那个可
训练的归属者**。`requires_grad` 的只写强引用表已被移除。leaf 注册、retained-gradient 跟踪
和优化器注册被保留，因为它们有真实的消费者。独立的 Tensor holder 使用弱标识索引，而 leaf
标识由原生图查询决定。**一次无关的反向绝不会抹掉一个存活的独立 holder 的注册**；
retain_grad 持续到该 holder 的生命期结束。它的 flag、梯度、data 视图的归属者/路径、设备
提示和 RMSNorm 缓存都属于 TensorObjectState。历史的不可弱引用 Var 保留其独立的清理路径。

`jittor.compat.shim` 拥有可选的顶层 `torch` 接口的运行时与部署代码，供直接 import torch 的
应用使用。`jittor.torch_shim` 仅作为同对象的历史别名保留。shim 把 Torch 风格语义委托给
`jittor.compat.torch`；**别名与已部署的包都不拥有第二份实现**。

普通的 Jittor 启动使用 `_runtime.import_aliases` 处理原生别名、`_runtime.compat_bootstrap`
处理可选激活。**它不导入任何 `jittor.compat` 模块**；只有在请求兼容时，缺失的可选包才会
产生安装错误。Torch 安装器在显式的 Torch 模式预检之后运行，或经由已部署的 `torch` entry
point，或在历史别名 `jittor.torch_compat` 被导入时运行。**这防止类级的 Torch 改造在无关
进程中改变原生 Jittor API。**

vLLM 的实现及其专属测试位于主仓库的 `adapters/jittor_adapters/vllm` 与
`adapters/tests/vllm`，由既有的 `jittor-torch-adapters` 发行物分发，**不由 Jittor 或
jittor-torch 分发**。可选的 Torch 阶段选择名为 `jittor_vllm` 的 `jittor.module_patches`
entry point，指向 `jittor_adapters.vllm:register`。它的 `register(callback) -> None` 注册器
装配导入前的扩展设置，并通过共享的事务机制注册导入后的层补丁。**未安装时报告 `unavailable`
而不破坏 Torch。** 公开的 Jittor 原语仍然是数学实现本身。该抽离**不包含**单独维护的昇腾
platform 与 worker 源码，**也不声称完整的 NPU serving 或硬件验证**。

归属顺序是：

1. 原生 Jittor 语义与普遍有用的运算；
2. `jittor.compat` 中可复用的机制；
3. 可选的导入/部署 shim；
4. 核心发行物之外的项目专属集成。

行为层面的判定规则见
[Torch 兼容原则](../compatibility/principles.md)。安装与延迟钩子写了哪些进程状态、哪些
可以回滚，见 [Torch 安装事务](torch-install-transactions.md)。

### Torch 前端的运行时状态与 API 归属

独立前端拥有自己的 Tensor、Module 与安装状态，**不建第二张执行图**；能委托给原生数学实现
的 API，以原生实现为准。

- **实现有可导入的标识。** 安装器发布的是已有的模块级函数和类，真正的实现放在它所属的
  族模块里；改嵌套函数的 `__qualname__` 不算。每次调用的算法回调可以闭包住输入，但安装
  **不得**在每次尝试时创建新的公开实现。类型工厂只配置类并绑定模块拥有的行为：
  `nn_frontend.py` 的 `NNFrontendOwner` 拥有层类缓存，`LayerInitializer` 与 `nn_adoption.py`
  负责原生构造与参数认领，不改外部/共享模块；分布对象的构造、方法与属性用
  `distribution_adapters.py` 里的描述符，每个前端各有一份状态。
- **调用时解析 owner。** `get_install_context(native_backend)` 在调用时解析显式绑定的前端
  owner：它是只读操作，**不激活 Torch，也不创建安装**；owner 缺失或不一致时拒绝。正在
  安装的代码直接用自己手里的 `InstallContext`。被捕获的原生实现先放进 context 拥有的状态
  再发布包装器，后来的查找因此不会递归调到包装器自己。CUDA facade 的可变状态
  （`installers/cuda/api.py` 的 `CudaRuntimeState`：逻辑流、NVTX、显存查询缓存与精度档位）
  也属于 `InstallContext.state`；API 标识稳定，可变状态归当前前端。
- **保真度如实登记。** 最终发布的对象用 `register_fidelity` 登记：`approximate` 表示仍有
  限制，`unimplemented` 也覆盖已有的注解或 no-op 占位，不只是会抛异常的函数。登记按 API
  拼写为准，别名可以共享对象但声明不同的支持行为；`api_manifest.py` 记录家族安装之后的
  最终组合路径，`fidelity_report()` 返回排好序的不可变记录，`fidelity_table()` 渲染覆盖表。
  **登记与命名空间封闭都不证明完整的 Torch 语义**：CUDA 逻辑流是串行化的，事件计时用同步
  后的主机时间戳，显存峰值在查询时采样，这些都是登记在案的限制。
- **根命名空间发布后封闭。** 独立根命名空间只在 bootstrap 期间允许原生回退；发布之后未
  声明的名字抛 `AttributeError`，删除一个局部名字也不会露出原生实现。`native_api.py` 为
  共享的数学运算声明稳定的 `NativeOperation` 对象，它们解析一张只读、归事务所有的委托表，
  并进入前端的结果类型/autograd 作用域。原生专属的 `flags`、`core`、`runtime` 不是公开的
  Torch 属性。FFT/linalg facade 复制公开导出（不含被导入的实现模块），保留自己的命名空间
  标识。

运行时服务与激活：

- `jt.runtime.service_state(namespace, factory=...)` 拥有扩展状态，**不往 `jittor` 上加私有
  属性**。不带 factory 的读取不创建状态；成功的 factory 每个 Runtime 只运行一次，失败的
  可以重试，同一服务的递归构造被拒绝；服务自己决定其字段的同步与事务语义。
- shim 激活使用 `jittor.torch.activation` 服务。**安装锁同时覆盖激活状态检查、状态转换和
  安装本身**：并发调用者等待，同一线程的重入激活被拒绝；回滚冲突必须释放锁并留下可查询
  的失败状态（`jittor.compat.shim.activation_status()`）。
- 张量记账使用 `jittor.torch.tensor_states`，一张 Runtime 拥有的弱 owner 表。
  `get_tensor_state()` 解析显式的前端绑定，`latest_optimizer()` 解析弱优化器引用。历史的
  leaf/retained/optimizer 根别名被认领一次后移除，安装失败时回滚恢复它们原来的归属。
- vmap 的 getitem 下沉提示是上下文局部、按 owner 划分的：用
  `TransformGetItemToIndex(owner)` 与 `getitem_transform_active(owner)`，不用模块属性。
  同一作用域的嵌套与异常退出都恢复进入时的状态，另一个执行上下文不继承活动的改动。
- `compat.torch.install` 在创建安装 context 之前就拒绝原生模块目标；`activate` 与导入期
  组合总是选择独立命名空间。旧的 false 模式参数和环境变量是显式错误。

### dtype 边界

`torch.dtype` 是**不可变对象，不是 Python 字符串**。`torch.long` 与 `torch.int64` 是同一个
对象；`str`、`repr` 与格式化都得到 `torch.int64`；与 `"int64"` 或 `"torch.int64"` 比较为假。
pickle 返回规范对象，并接受旧的字符串子类 pickle 状态。dtype 对象不可调用：转换用
`tensor.to(dtype=...)`，原生 Jittor 代码用 `jt.float32(...)`。

边界上是两种不同的操作，外加两处注册：

| 操作 | 归属 | 契约 |
| --- | --- | --- |
| 为元数据、分派、源码生成或持久化读取 dtype 名 | `jittor._core.dtypes.dtype_name(value)` | 返回裸的规范名；**不授予计算支持** |
| 在 Python/NumPy 分配之前消费 dtype | `jittor._core.dtypes.dtype_for_compute(value)` | 走原生带检查的 NanoString 转换器，拒绝不支持的占位符 |
| 注册前端 dtype 类 | `jittor._core.dtypes.register_dtype_type(type)` | 在 Python 位置 dtype 参数与 C++ 实参边界注册同一个确切的类 |
| 原生 dtype 实参消费对象 | `src/bindings/pyjt/py_dtype.{h,cc}` 与 `py_converter.h` | 除原生与 NumPy 拼写之外，只接受显式注册的前端类型；带 `name` 或 `type` 属性的任意对象**不被接受** |

原生注册表由核心与它的 JIT 扩展共享。带检查的转换器保留 Python 异常：占位符在算子构造
之前抛 `NotImplementedError`。规范的裸字符串继续走原生快路径；带 `torch.` 前缀的字符串与
注册过的占位符名字经由注册的 dtype 对象解析。`ones`、`zeros`、`empty` 也认得出最后一个
位置参数里的注册 dtype——把它当成形状分量会静默退回默认的 float32。

前端定义了 34 个不同的 dtype 对象，其中 14 个有原生分配/计算表示：bool、四种有符号整数、
四种无符号整数、float16、bfloat16、float32、float64 与 complex64。其余 20 个只供导入期的
字典与元数据使用，**不能**传给工厂函数、cast、随机数生成或原生算子的 dtype 参数。各后端
实际支持的集合比这个框架级集合更窄。

原生 Python 代码**不得**把 `str(tensor.dtype)` 当 kernel 类型名，也不得拿前端 dtype 对象
和字符串比较，这两件事都用 `dtype_name`。把 dtype 对象直接交给原生算子，或在交给 NumPy
之前用 `dtype_for_compute`。保留 NumPy 输入的 dtype；float64 直接构造，而不是先建
float32 再在精度丢掉之后 cast。BF16 的 NumPy 输入先显式暂存为 float32，再走原生 BF16
cast。新增序列化状态名或 dtype 比较时同样用 `dtype_name`；不得往原生规范化器里重新引入
兼容层导入，也不得为了让过期调用点工作而恢复字符串继承。

## 后端构建配置

### 配置值与 provider 钩子

`jittor_utils.build_config.BuildConfig`（实现在 `python/jittor/build/utils/build_config.py`）
是一个**冻结的值**：编译器路径与 flag、缓存/源码根、后端能力、额外的核心源码、环境覆盖
和后端资源。源码清单是 tuple；资源与环境映射复制输入并且只读；加载进来的模块与驱动句柄
算作资源，不是编译器状态的副本。`evolve(...)` 返回一份新配置，不修改输入。

后端 provider 实现 `configure(context) -> BuildConfig`。`BuildContext` 提供显式的编译、
动态加载与库发布服务，**不提供源码改写服务**：需要不同代码的后端注册自己的实现、贡献自己
的翻译单元，没有 provider 会拿到共享源码树去改写。`config.jittor_path` 指向被编译的
checkout，由 bootstrap 设定一次，之后不会被改指到派生副本。provider 不得 import
`jittor.compiler`、给编译器属性赋值、往它的源码列表追加，也不得修改进程环境；bootstrap
应用返回的环境变化，并一次性发布兼容用的编译器属性。`compiler.build_config` 是最终发布的
值，`compiler.make_backend_context()` 把同一份配置交给 `install_extern(context)` 与
`post_process(context)`。

`jittor_utils.backend_discovery` 用 Protocol 描述被选中的 SDK provider（Python 3.7 从条件
依赖 `typing_extensions` 取 `Protocol`）：

| 入口 | 输入 | 结果与归属 |
| --- | --- | --- |
| `configure` | 不可变的 `BuildContext` | 一份新的 `BuildConfig`；不改编译器全局，也不改输入配置 |
| `install_extern` | 带库发布服务的 context | 布尔值：该 provider 是否处理了外部库的安装 |
| 可选的 `post_process` | 核心构造完成之后的 context | 绑定运行时算子；协调者忽略返回值 |

可选钩子单独建模为 `BackendPostProcess`。发现过程只导入被选中的 entry point，并在
`configure` 编译任何东西之前校验所有必需钩子和每个提供了的可选钩子；缺钩子或签名不对时
`TypeError` 指名 provider 与方法。缺 SDK 与运行时失败仍是被选 provider 的显式失败，
**不是隐式的 CPU 回退**。ACL 把额外源码文件放在返回值里，并把编译出的注册初始化器与加载的
运行时库作为资源保留；Corex 在发布前把 OpenMP 从通用与 kernel flag 中去掉；ROCm 把自己的
HIP 运行时翻译单元声明为后端源码，SDK 根作为资源，库经注入的注册回调发布。三者都不再改写
共享源码。

Python 构建协议**不替代** `src/runtime/backend.h` 带版本的 `BackendOps` 表。那张表拥有
设备、内存、流、事件回调与执行能力；原生注册表校验 ABI 与必需回调并复制该表，代码与分配器
池必须活得比它的 Runtime 长，流/事件句柄属于其声明的设备。当前是 **ABI 3**：复制描述符或
读取其策略尾部之前先检查版本与结构体大小，按旧版本构建的扩展必须重建。ACL 对锁页主机内存、
编译器并发和归约的要求就写在这张描述符里（见上文"ACL kernel 注册"），并按实际的加速器
目标选用：ACL 构建运行 CPU 作用域时用普通的 CPU 策略。分配才是驻留的权威；前端放置是另一个
图约束，见[设备与放置](../notes/device-placement.md)。

### 发现与选择

- provider 注册在 `jittor.backends` 包的 entry-point 组里（`acl`、`rocm`、`corex`）。有已安装
  元数据时用元数据；源码 checkout 对这三个内置 provider 有匹配的惰性回退条目。
- `JT_BACKEND=cpu|cuda|acl|rocm|corex` 显式选择 provider，`npu` 是 `acl` 的别名、`hip` 是
  `rocm` 的别名，其它名字选择第三方 entry point。没有显式选择时，由 SDK 环境变量或惯常的 SDK
  编译器路径识别可选后端；**配置了多个 SDK 时要求显式选择**，而不是看哪个模块先被导入。
- 未选中的 provider 不被导入，它们的 `check()` 不运行，发现过程既不调用其编译器也不初始化
  其设备。被选 provider 的配置失败向上传播，不会静默换成另一个后端。
- 显式选择 CPU 时，在任何 CUDA 安装检查、可执行文件查找、驱动查询或下载之前就返回，即使
  设了冲突的 `nvcc_path`。
- 历史的 `has_cuda` 构建字段描述"是否编译了 `HAS_CUDA` 加速器支持"，**不证明有 NVIDIA
  驱动**：配置成功的 ACL/ROCm/Corex provider 也会设它，`is_cuda` 单独标识 NVIDIA 编译。
  Corex 自己提供该宏并移除继承的 `IS_CUDA`。
- 普通 CPU/CUDA 的缓存指纹保留原有字段与值；显式的后端选择与配置的可选 SDK 路径只在存在
  时才加入指纹字段。

### 工具层与资源

- `jittor_utils.compile_module` 接收 `ModuleBuildServices`（绑定生成回调、命令格式化器、编译器
  路径与缓存/源码根）。编译器 bootstrap 安装默认服务，独立使用者可以显式传入；两者都没有时
  它在写任何构建产物之前报错，提示先 import jittor——**它从不为了找服务而 import Jittor**。
  不存在把 checkout 复制进每后端缓存再改写的整树源码转换器。
- 张量序列化实现在 `jittor.serialization`，不在构建工具包里。历史的
  `jittor_utils.load_pytorch`、`load_pytorch_old`、`save_pytorch` 查询 runtime bootstrap 注入
  的加载器：访问得到的函数与类是规范对象，模块对象本身不是别名；历史 pickle 的 `GLOBAL`
  名字在 bootstrap 之后照常解析。单独导入旧工具模块无害，但在 `import jittor` 之前访问其运行
  时 API 会显式报错。
- CUDA 实现只有一个物理归属 `backends/cuda`：`kernels/` 放算子 kernel 与 Python 实现，
  `libraries/<name>/{include,src}` 放库支持代码，`include/` 与 `src/` 放公共支持资源；Python
  模块名是 `jittor.backends.cuda.kernels.*` 与 `jittor.backends.acl.kernels.*`。源码包
  `python/jittor/backends` 只是路径桥，不复制后端实现；setuptools 把各后端包映射进 wheel 的
  `jittor/backends/`，sdist 保留顶层布局与包映射。
- `jittor_utils.backend_resources.backend_root(jittor_path, name)` 在不导入 Jittor 的前提下
  解析源码、已安装或转换布局下的资源根：源码 checkout 优先顶层后端；已安装候选要求真实的
  包标记，因此只剩 `__pycache__` 的残留目录不能把编译重定向走。没有任何东西再产生转换布局，
  识别它只是为了让已安装的旧树继续可解析。
- 纯主机的索引调度器保留在 CPU 构建中；加速器代码生成翻译单元与 NaN 检查 CUDA 源码只进入
  加速器构建。

### 验证边界

离线测试覆盖不可变输入、entry-point 选择、未选 provider 的隔离、注入的编译服务，以及伪造
的 ACL/ROCm/Corex 配置；`tests/structure/test_core_source_is_not_ported.py` 拒绝
`BuildContext` 上的改写服务、任何"镜像整树并改写原生源码"的产品函数，以及 bootstrap 之外对
`jittor_path` 的重新绑定。`tests/build/test_compile_module_dependencies.py`、
`tests/serialization/test_native_serialization_ownership.py` 与
`tests/bindings/test_load_pytorch_strides.py` 含 JIT 依赖的检查，与其它原生验证一样串行跑在
受支持的缓存策略下。**这些都不证明 CANN/ROCm/Corex 的 ABI 或设备正确性**：在目标机器上配置
SDK、设 `JT_BACKEND`，在支持该断言的套件里禁用 CPU 回退，先测普通计算、扩展加载与失败传播，
再声称硬件支持。各后端上机要跑什么见
[`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md)。

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

## 导入与初始化规则

Torch 的 dtype 对象及其原生/NumPy 消费点遵循上文的"dtype 边界"一节。

- 模块导入**不得**编译 kernel、下载资源、修改源码 checkout，或静默安装外部包。
- **注册必须幂等。** 重新导入一个兼容模块不得把同一个可调用对象包两次，也不得创建第二个
  模块对象。
- 可选依赖的检查发生在**运算边界**，除非导入期发现本身就是那个 API。
- 宽泛的异常处理器可以标注并**重新抛出**失败；**不得**把一个只安装了一半的兼容接口转成
  表面上的成功。
- 昂贵的导入不进入仅做收集的结构测试。
- 同一文件里的顶层定义**不得**被后面的定义静默替换。跨文件的相同实现同样被扫描；保留的
  重复需要属于一个窄的、经过评审的类别，例如独立的部署入口、后端代码生成模板、模型局部的
  架构块，或历史的序列化读取器。

## 运行时资源

以下几棵树被编译器或打包代码按**物理路径**消费，因此改动需要特别评审：

- `src/`（安装为 `jittor/src/`）
- `backends/acl/{include,kernels/native,src}/`
- `backends/comm/`
- `python/jittor/contrib/math_util/src/`
- `compat/shim/cpp_extension/`（可选的 `jittor-torch` 发行物）

**只有当源码 checkout、sdist、wheel、冷 JIT 构建和安装后冒烟测试全部一致时，一次搬动才算
完成。** 目录美观本身不是搬动这些资源的理由。

## 重构流程

每次模块搬动：

1. 清点定义、赋值、导入、注册和消费者；
2. 显式定义规范目的地以及任何兼容别名；
3. 搬动一个**连贯的领域切片**，不夹带无关的行为改动；
4. 尽可能**机械地**比对搬动前后的定义集合与公开导出；
5. 测试导入标识、公开调用、动态分派、适用时的序列化，以及相关的 CPU/加速器行为；
6. 删除过渡源码路径，并把它加进结构门禁；
7. 在**同一次改动中**更新长期文档与活跃链接。

**搬动之后不得保留两份可编辑的实现。** 兼容必须委托给规范对象，并且有退出条件。

## 验收

一次源码布局改动在满足下列条件时可以接受：

- 导入与公开名字保持其有文档记载的行为；
- 不残留任何历史实现树或根级兼容文件；
- wheel 含有全部必需的运行时资源，并排除仅供仓库使用的测试/工具；
- `bash tools/check_repo_layout.sh` 通过；
- 定向测试、结构测试和每一个受影响的后端门禁都通过；
- 任何**刻意的不兼容**都记入发布说明。
