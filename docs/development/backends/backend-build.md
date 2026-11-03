# 后端构建与运行时资源

[本栏目目录](../index.md) · [English](backend-build.en.md)

修改后端 provider、构建配置或发行物资源时阅读本页。先读[源码架构](../source-architecture.md)了解模块归属。

本页描述实现契约，不是新的硬件验收报告。原文复查基线为 `e3c369acb`；本次拆分保留该证据边界。

## 配置值与 provider 钩子

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
编译器并发和归约的要求就写在这张描述符里（见[ACL kernel 注册](../runtime/operator-dispatch.md#acl-kernel-注册)），并按实际的加速器
目标选用：ACL 构建运行 CPU 作用域时用普通的 CPU 策略。分配才是驻留的权威；前端放置是另一个
图约束，见[设备与放置](../../notes/device-placement.md)。

## 发现与选择

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

## 工具层与资源

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

## 验证边界

离线测试覆盖不可变输入、entry-point 选择、未选 provider 的隔离、注入的编译服务，以及伪造
的 ACL/ROCm/Corex 配置；`tests/structure/test_core_source_is_not_ported.py` 拒绝
`BuildContext` 上的改写服务、任何"镜像整树并改写原生源码"的产品函数，以及 bootstrap 之外对
`jittor_path` 的重新绑定。`tests/build/test_compile_module_dependencies.py`、
`tests/serialization/test_native_serialization_ownership.py` 与
`tests/bindings/test_load_pytorch_strides.py` 含 JIT 依赖的检查，与其它原生验证一样串行跑在
受支持的缓存策略下。**这些都不证明 CANN/ROCm/Corex 的 ABI 或设备正确性**：在目标机器上配置
SDK、设 `JT_BACKEND`，在支持该断言的套件里禁用 CPU 回退，先测普通计算、扩展加载与失败传播，
再声称硬件支持。各后端上机要跑什么见
[历史记录](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/deferred-hardware.md)。

## 运行时资源

以下几棵树被编译器或打包代码按**物理路径**消费，因此改动需要特别评审：

- `src/`（安装为 `jittor/src/`）
- `backends/acl/{include,kernels/native,src}/`
- `backends/comm/`
- `python/jittor/contrib/math_util/src/`
- `compat/shim/cpp_extension/`（可选的 `jittor-compat` 发行物）

**只有当源码 checkout、sdist、wheel、冷 JIT 构建和安装后冒烟测试全部一致时，一次搬动才算
完成。** 目录美观本身不是搬动这些资源的理由。
