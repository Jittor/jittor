# 算子注册、分派与后端回退

[本栏目目录](../index.md) · [English](operator-dispatch.en.md)

修改算子注册、kernel 选择或后端回退时阅读本页。先读[源码架构](../source-architecture.md)了解模块归属。

本页描述实现契约，不是新的硬件验收报告。原文复查基线为 `e3c369acb`；本次拆分保留该证据边界。

## CUDA 资源布局

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

## 原生算子分派

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

## Python kernel 分派

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

## ACL kernel 注册

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
由分配器、编译器和归约的归属者消费；**公开 flag 不会被覆盖**。BackendOps ABI 3 拒绝更旧的
描述符，扩展必须重建。历史的整树 SDK 转换（`process_acl`、`process_jittor_source`）已经
消失；**真实的 CANN/NPU 验证仍然是必需的**。

## 后端回退策略

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
