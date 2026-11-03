# Torch 前端的归属与 dtype 边界

[本栏目目录](../index.md) · [English](frontend-boundaries.en.md)

修改前端安装、API 对象标识或 dtype 转换时阅读本页。先读[源码架构](../source-architecture.md)了解模块归属。

本页描述实现契约，不是新的硬件验收报告。原文复查基线为 `e3c369acb`；本次拆分保留该证据边界。

前端与统一图的设计见 [Torch 架构](torch-architecture.md)；回滚细节见[安装事务](install-transactions.md)。

## 兼容 API

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
物理的 `compat/` 树现在就是独立的 `jittor-compat` 发行物，**核心打包排除它**。显式的历史模式
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

vLLM 实现和专属测试位于独立的 [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm)，不由 Jittor 或 jittor-compat 分发。可选 Torch 阶段发现 `jittor.module_patches` 中的 `jittor_vllm` 入口，指向 `jittor_vllm:register`。注册器通过共享事务机制装配导入前的扩展设置及导入后的层补丁。未安装时报告 `unavailable`，不破坏 Torch。公开 Jittor 原语负责数学实现，独立仓库负责硬件验收。主仓根 README 提供生态组织入口，不保留适配器目录。

归属顺序是：

1. 原生 Jittor 语义与普遍有用的运算；
2. `jittor.compat` 中可复用的机制；
3. 可选的导入/部署 shim；
4. 核心发行物之外的项目专属集成。

行为层面的判定规则见
[Torch 兼容原则](principles.md)。安装与延迟钩子写了哪些进程状态、哪些
可以回滚，见 [Torch 安装事务](install-transactions.md)。

## Torch 前端的运行时状态与 API 归属

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

## dtype 边界

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
