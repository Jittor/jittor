# PyTorch 调用如何快速下沉为 IR：关键技术点

[本栏目目录](../index.md) · [English](torch-lowering.en.md)

新增或调整 Torch API 时，沿本页检查参数适配、安装顺序与返回类型。先理解[分层架构](torch-architecture.md)，验收要求见[兼容原则](principles.md)。

- 状态：现行实现说明
- 上次复查：2026-11-13
- Owner：Torch 兼容维护者
- 复查触发：新增 torch API、安装器编排或下沉机制变化

本文是[分层架构](torch-architecture.md)第二层的展开：**PyTorch 代码怎样以最小代价变成
统一计算图上的节点**。目标不是"能跑"，而是"覆盖广、下沉快、差异可见"。

## 1. 一次调用的完整路径

以 `torch.cat([a, b], 1)` 为例：

```text
torch.cat                   TorchNamespace 上的前端函数
  → 参数归一                axis→dim、NestedTensor、空张量、0-d 补维
  → jt.concat(...)          原生算子，在图里建一个 Node
  → 返回时构造前端类型      C++ 侧按 PyContextVar 决定返回哪个 Tensor 子类
```

自定义算子走另一条支路：`torch.ops.<ns>.<op>(...)` → `_RegisteredOp.__call__` →
按 dispatch key 选 kernel → 普通 kernel 直接调用，注册过 autograd 的包进
`_LibraryAutograd` 挂到原生 tape。

下面各节是让这条路径"快"且"稳"的关键技术点。

## 2. 安装器编排：步骤表 + marker + 事务

下沉规则不是散落在 import 副作用里，而是一张显式步骤表
（`compat/torch/__init__.py`）：

- `_REQUIRED_STEPS`：二十余步，顺序为 `core → tensor.base → tensor.methods →
  nn → optim → autograd → cuda → distributed → serialization → utilities →
  data → distributions → compiler → numerical → ...`，末尾是各族
  `*.module-keys` 收尾；
- `_OPTIONAL_STEPS`：`torchmetrics` / `transformers` / `tensordict` /
  `safetensors` / `flash-attn` / `vllm`，失败只告警。

每个步骤是 `InstallStep = (name, installer)`（`compat/torch/contracts.py`），执行
前用 `inspect.signature` 校验签名（`validate_installer`）。`InstallContext`
（`compat/torch/context.py`）负责执行：

- `run_required`：marker 已是 `"complete"` 就跳过——**安装幂等**；失败经
  `swallowed()` 登记后抛 `InstallStepError`；
- `run_optional`：捕获可预期异常，marker 记 `"failed"`，只告警一次；
- `mark_complete`：在命名空间上写 `_torch_compat_install_complete`。

失败不会留下半成品：`install()` 捕获异常后回滚 `InstallTransaction`
（`compat/transaction.py`），恢复属性、marker、模块发布与委托表，并释放进程锁。
成功路径把子事务 `adopt` 给父事务后 `commit()`。步骤表末尾再执行
`native_api.install(context)` 与 `api_manifest.register_public_apis(context)`，
把显式委托与保真度台账补齐。

## 3. ContextVar：类型/精度/放置的零成本切换

"同一个 Var，两套前端类型"的接缝在 C++：`src/bindings/pyjt/py_tensor_frontend.cc`
用 `PyContextVar` 保存当前前端类型，`_set_tensor_frontend_type` 校验它必须是
`PyjtVarHolder` 的子类，缺省回退到原生 `Var`。

Python 侧把它包成一个作用域（`compat/torch/frontend.py` 的 `tensor_frontend()`），
一次进入同时设置四件事：

```python
token = backend.core._set_tensor_frontend_type(tensor_type)   # 返回哪种类型
precision_token = backend.core._set_float32_precision(...)     # float32 精度策略
placement_token = backend.core._set_tensor_placement(...)      # 构造期设备
with backend.autograd.policy_scope(backend.autograd.EXPLICIT_REQUIRES_GRAD):
    yield
```

`finally` 里逐个 `reset`。这个设计带来三点：

1. **不做全局类替换**，原生 `import jittor` 的默认行为不受影响；
2. **可重入、上下文安全**，嵌套作用域按栈恢复；
3. **异常安全**，失败路径也会还原策略。

注意 placement 是"构造期覆盖"，与运行时执行 flag 无关——它决定新张量落在哪个
设备，不改变执行器状态。

## 4. 声明式委托：一张白名单覆盖百余算子

对"拼写与原生一致"的数学 API，不逐个写包装，而是维护白名单
（`compat/torch/native_api.py`）：

- `_NATIVE_NAMES`：`matmul`/`einsum`/`gather`/`scatter`/`cummax`/`unique` 等
  百余个算子；
- `_NATIVE_MODULE_NAMES`：`fft.*`、`linalg.*`。

每个名字对应一个稳定的 `NativeOperation` 对象（有 `__name__`/`__module__`，
支持 `__reduce__` 序列化）。调用时从安装上下文取实现并在 `tensor_frontend()`
作用域内执行：

```python
implementation = context.state["native_torch_operations"][self._operation_key]
with tensor_frontend(context.state["Var"], device=device):
    return implementation(*args, **kwargs)
```

委托表由 `api_delegates.bind_delegates()` 写入 `context.state`，是**不可变**
`MappingProxyType`，并且用 `transaction.record(...)` 记录旧值——回滚时随账本
还原。**白名单而不是属性透传**，才能逐条登记保真度、逐条写测试。

## 5. 方法表：一族 API 一次安装

`torch.Tensor` 的方法同样表驱动：

| 表 | 位置 | 内容 |
| --- | --- | --- |
| `_BINARY_APIS` | `installers/tensor/method_api.py` | `__add__`/`__mul__`/`__truediv__` 等 dunder 二元运算 |
| `_CAST_APIS` | 同上 | `byte`/`short`/`long`/`half`/`float`/`bool` 等类型转换 |
| `_UNARY_INPLACE_APIS` | 同上 | `neg_`/`exp_`/`sqrt_`/`sigmoid_` 等原地一元运算 |
| `_SHAPE_REDUCTION_APIS` | `installers/tensor/shape_api.py` | `mean`/`prod`/`any`/`all`/`max`/`min`/`argmax`/`var` 等归约 |

消费点是 `installers/tensor/methods.py` 的 `_install_tensor_methods()`：先缓存原生
实现（`_native_operators`），再按表 `setattr` 覆盖前端 `Tensor` 子类，并把原生
句柄存进 `ctx.state["tensor_native_api"]`（`MappingProxyType`）。**新增一族方法
= 加一行表项**，而不是再写一个包装函数。

## 6. 语义适配只发生在必要处

能直接委托就不包装；包装只为真实的契约差异。`torch.cat`
（`installers/tensor/__init__.py`）是典型：

- `axis=` 别名归一到 `dim`；
- NestedTensor 与 `__torch_function__` 协议分派；
- 空张量跳过、0-d 张量补维，再落 `jt.concat`；
- uint8 输入不会被原生降成 int8。

两个反复出现的坑有专门处理：

- `_array_keep_dtype()`：`jt.array` 会把 int64 降到 int32，所以显式按 dtype
  构造，保住宽类型；
- 身份契约：与其包一层 `_axis_to_dim`（会让 `torch.amax is Var.amax` 断言失败），
  不如让稳定对象自带 `axis=` 并标记 `_torch_accepts_axis = True`。

## 7. nn.Module：子类化 + 参数认领 + forward 路由

`compat/torch/nn_frontend.py` 的 `NNFrontendOwner` 生成前端类型：

- `Module` 是原生 `jittor.nn.Module` 的**动态子类**；
- `Parameter` 是 `frontend.make_parameter_type()` 生成的 Var 子类；
- `adapt_class(native)` 按需为每个原生层生成子类并缓存；
- `copy_module()` 把 `jittor.nn` 模块树复制成 `torch.nn` 命名空间。

构造时的认领在 `compat/torch/nn_adoption.py`：`LayerInitializer` 调完原生
`__init__` 后执行 `adopt_owned_children()`，用 `ChildAdoption` 把原生子模块换成
前端类型、用 `ParameterRewrite` 把提升出来的参数换成前端 `Parameter`，`external`
集合里的对象永不改写。

调用路由的关键是**替换 `Module._dispatch_call` 而不是 `__call__`**
（`installers/nn/module_methods.py`）：这样 `__call__` → hook → `_dispatch_call`
的路径保留。`_prefer_forward()` 按 MRO 判断子类自己的 `forward()` 是否比最近的
`execute()` 更派生；`_execute()` 再兜底到原生实现。首次发布参数为 backward leaf、
FSDP2 真路径与流水线逻辑都挂在 `_call` 里。

## 8. autograd：桥到原生 tape

`compat/torch/autograd.py` 的 `Function` 继承 `jt.Function`，把 torch 的三件套
接到原生图上：

- `execute` → 调用户的 `forward`；
- `__call__ = _call_record_inputs`：记录输入形状、`needs_input_grad` 与输出；
- `grad = function_grad`：materialize 缺失梯度（浮点输出补 `jt.zeros`），并用
  `_sum_grad_to` 把梯度规约回输入形状。

`grad()` 最终走 `jt.core.grad_optional(...)`；`backward()` 走 Var 的 `_backward`
并填充优化器的 `pg["grads"]`。`torch.library.register_autograd` 注册的反向由
`_LibraryAutograd(jt.Function)` 承接：前向在 `execute` 里跑并把浮点输出
`start_grad()`，反向调注册的 `_backward`。

精度与开关不改变下沉路径，只改变原生图的行为：autocast 写 `jt.flags.amp_reg`
（影响原生 dtype 推断），`no_grad`/`enable_grad` 走原生作用域，`GradScaler` 只在
优化器梯度层缩放与跳步。

**诚实边界**：`saved_tensors_hooks`/`save_on_cpu`/`Node` 是 metadata-only 占位，
登记为 `Fidelity.UNIMPLEMENTED`，不做张量打包或 offload。

## 9. 自定义算子：dispatch key 与 schema

`compat/torch/library.py` 实现 `torch.library` / `torch._ops` 的可用子集：

- `_RegisteredOp.select_impl()` 按 `_CUDA_DISPATCH_ORDER` / `_CPU_DISPATCH_ORDER`
  的顺序挑 kernel，`_argument_residency()` 从第一个 Var 参数推断 CPU/CUDA，
  autocast key 按当前策略插到最前；
- **只注册了 `Meta`/`FakeTensor` 的算子显式报错**——meta kernel 只返回形状正确的
  空张量，用它服务真实调用会静默产生假数值；
- `custom_op` / `impl`（= `register_kernel`）/ `register_fake` /
  `register_autograd` 分别写前向、fake 与反向；`infer_schema()` 从 Python 注解
  推导 op schema。

注册了 backward 的算子在原生输出 tape 构造之前就把浮点/复数结果标为需要梯度（即使 kernel
内部 detach 了结果），整数/bool 结果不标；`Library` 的注册表与算子元数据写入参与当前的安装
事务（`_transactional_registry = True`），失败时随事务回滚，见
[Torch 安装事务](install-transactions.md)。

下游库的同名算子可以被集成覆盖：`compat/integrations.py` 的
`custom_op_overrides()` 提供替换表，`custom_op` 命中时记录
`_overridden_by_integration`，而不是悄悄换掉实现。

## 10. 失败可见与可回滚

下沉层的四条可见性机制：

| 机制 | 位置 | 作用 |
| --- | --- | --- |
| 保真度台账 | `compat/torch/fidelity.py` | `EXACT`/`APPROXIMATE`/`UNIMPLEMENTED` 逐条登记，`fidelity_table()` 可导出 |
| 显式失败 | `compat/stub_policy.py` | `unimplemented()` 直接抛错（只有 `JITTOR_TORCH_ALLOW_STUB` 才降级为警告）；`degraded()` 只告警 |
| 吞异常登记 | `compat/diagnostics.py` | `EXPECTED` 白名单（不含 `NameError`/`AssertionError` 等），`swallowed()` 有界记录，可用 `torch.compat_swallowed()` 回读 |
| 安装事务 | `compat/transaction.py` | `record`/`record_object_diffs`/`rollback`/`commit`，失败安装不留半成品 |

这四者合起来回答"这个 API 到底支持到什么程度"——**不需要读源码猜**。

## 11. 另一条路径：源码级翻译器

`compat/pytorch_converter.py` 提供**离线**的 PyTorch→Jittor 源码改写：

- `pjmap`：PyTorch 名字 → `jittor` 模块/名字，带参数改名（`links`）、强制补参
  （`extras`）、丢弃参数（`delete`），另有 `unsupport_ops` 黑名单与
  `pjmap_append()` 扩展点；
- `convert(code)`：`ast.parse` → `Converter` 遍历改写（import 改写、
  `load_state_dict → load_parameters`、`size(...) → shape[...]`、
  `forward → execute`）→ 反解析成源码；未映射的构造直接生成 `raise`；
- `converter_server.py` 是可选的 HTTP 前端。

它与运行时 shim 的分工：**能 `import torch` 跑起来就用 shim**（覆盖全、进同一张
图、带台账）；需要产出静态 Jittor 代码、或环境里不能装 shim 时用翻译器。翻译器
**不建立 IR**，也不参与 `TorchNamespace`。

## 12. 为什么"快"

- **一次安装、零 per-call 解释**：委托与方法是对象属性，类型选择是一次
  `PyContextVar` 设置，不是逐调用分支；
- **表驱动**：一族 API 一次 `setattr`，新增成本是一行表项；
- **单点适配**：差异只存在于包装处，其余全部直通原生算子；
- **优化复用**：下沉完成即进入原生融合与 JIT 代码生成，前端不需要再实现一遍
  优化（见[分层架构](torch-architecture.md)第 3 节）。

## 13. 新增一个 torch API 的清单

1. **先定 owner**：探测 `hasattr(jt, name)` / `hasattr(jt.Var, name)`。契约相同 →
   薄转发并在 import 期捕获原生对象；契约不同 → 由 compat 拥有包装并写清差异；
   没有原生 owner → compat 是最终 owner。判据见
   [torch-api-cohort-promotion](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/skills/torch-api-cohort-promotion/SKILL.md)。
2. **登记**：进 `compat/torch/api_manifest.py` 的 `API_PATHS`（`APPROXIMATE`）
   或 `UNIMPLEMENTED_PATHS`。
3. **保真度**：用 `register_fidelity()` 写清级别与限制明细。
4. **三件套验收**：模块级身份（`torch.foo is <owner>.foo`、`__module__`、
   `__name__`）+ `fidelity_of("torch.foo")` 元数据 + NumPy CPU 定点对拍；
   设备相关的再参数化到 CUDA 侧。
5. **结构门禁**：`tests/structure/test_torch_api_surface.py` 与已检入的
   `torch_api_manifest.json`；有意增删时用 `--regenerate` 在同一提交内同步。
6. **收尾**：`JITTOR_TORCH_SHIM=1 python -m pytest tests/structure -q`，判据是与
   改前**逐条同集合**，而不是"没有新增失败"。

## 14. 常见坑

- 把只注册了 `Meta`/`FakeTensor` 的算子当作可用实现；
- 覆盖原生方法前不缓存原实现，导致不可回滚；
- 用 `_axis_to_dim` 之类包装破坏对象身份契约；
- 忘记 `_array_keep_dtype`，让 int64 静默降位；
- 用宽泛 `except` 吞掉安装失败，把一个半装好的兼容层变成表面成功；
- 把 CPU 回退当成设备支持——下沉后必须在所声明的真实 device 上执行。

## 15. 训练输入与状态更新的边界

`DataLoader(num_workers>0)` 先在调用线程构造首个 batch，确定返回对象是否允许
安全预取。CPU / CUDA 构建的 Linux 上，默认或 `spawn` context 下的连续、无梯度 CPU Tensor batch
使用独立 CPU 进程；worker 禁用加速器与分布式初始化，顺序取样，异常传回主进程。
共享内存承载 NumPy buffer，接收线程只接收和预取内存页，训练线程通过原生 DLPack
构造 Tensor。内存池保留映射，最后一个 NumPy 或 Tensor 别名释放后，worker 才能复用
相应槽位。被调用方保留 batch 时，其存储仍然有效。`persistent_workers=True`
时，完整消费一轮后保留进程、worker RNG 与共享内存租约；下一轮用新 sampler，直接
由 worker 取首个 batch。中途放弃的一轮不复用未消费的结果。异常、超时或主动
关闭迭代器回收 worker；未开启持久 worker 时，每轮结束也会回收。

纯 Python / NumPy batch 继续使用线程预取；CUDA、有梯度、子类或其它不适用的
Tensor batch 保持串行。首次 batch 的线程归属、完整 worker RNG / context 语义与
固定内存尚未完全对齐 Torch，保真度仍为 approximate。
这条 CPU 数据通路不是计算向 CPU 的 fallback；模型与训练计算仍必须在声明设备验证。

worker 整理 NumPy 样本时，符合导入条件的新 `stack` 数组直接通过 DLPack 建立
Tensor，避免在 IPC 拷贝之前再复制一份。`from_numpy` 对可写、连续、原生字节序的
常见数值数组共享 CPU 存储并保留 NumPy owner；其它布局和类型保留可移植构造路径。
无梯度索引写入与其它原地写入一致，保留共享分配，NumPy 与 Tensor 双向观察更新。
`Tensor.new_*` 工厂创建独立叶子，默认不请求梯度；显式 `requires_grad=True` 不受
`no_grad` 工厂作用域影响，也不继承源张量的梯度策略。

Torch 普通 `BatchNorm` 默认使用 rank 本地统计量；显式 `SyncBatchNorm` 保持同步。
原生 Jittor `BatchNorm` 的默认设置不变。DDP 在初始化时广播参数，并按
`broadcast_buffers` 在初始化和每次前向前广播 buffer。写回保留 Var 与底层存储身份，
使别名和设备图读取同一块更新后的存储。

CPU worker 编码连续、无梯度的数值 Tensor 时，使用原生 data 描述符保留持有分配的
NumPy 视图，再同步复制进共享内存，省去整批中间副本；共享内存仍有独立的 lease 生命周期。
bool、bfloat16 与不支持的布局保留原编码通路。

CUDA FP32 四维 affine BatchNorm 的训练统计快速路径也服务于冻结参数和 `no_grad`
调用；它们继续按真实 batch 更新 running stats，不切换为 eval 统计。
无梯度、连续 NCHW 输入在 cuDNN 可用且 epsilon 至少为 `1e-5` 时使用其训练统计路径；
临时 running moments 先清零，再将 cuDNN 的无偏方差转换为公共更新函数需要的有偏方差。
其它布局、后端和需要梯度的调用保留各自的计算路径。

`torch.compile(module)` 在训练态 `no_grad` 调用中使用完整状态回放，保留 BatchNorm
等 buffer 的逐步更新；eval 调用使用推理图回放。训练步和设备图在覆盖状态存储之前，
还会提交调用者持有的待计算副本，保持无梯度快照的原始值。

冻结 FP32 特征的 CUDA SiLU 使用一次除法完成输出，深度卷积以 FP32 FMA 累加，
连续四维 Tensor 的 H/W 均值使用固定的四累加器归约树。所声明的梯度路径、其它
dtype 和不支持的布局继续使用对应实现；这些路径不启用 TF32。Float64 与
complex128 均值反向的比例也保留双精度，避免先计算 FP32 倒数再扩大到双精度。
`erf/erfinv` 的 Float64 反向常数同样直接以双精度构造，并跟随输入的设备放置；
一阶与二阶导数都保持调用者请求的精度。

交叉熵的自定义前向与均值分母使用独立的行权重存储边界，避免并行 CPU 融合只留下
某个工作线程的部分和；新增存储为 O(N)，不复制完整 logits。CPU Float32 L1 范数
内部使用 double 累加，公开结果与梯度保留 Float32。

原生 CUDA 二维池化将每块线程数限制为 256，并声明 launch bounds，前后向均检查
启动错误，避免 NHWC 内核的寄存器需求超过块资源上限。cuDNN RNN 的权重布局查询
在执行器首次运行前也会显式选择 CUDA 临时分配器并绑定当前流。

CUDA 梯度同步通过原生 `nccl_all_reduce_many` 将多个独立张量放进同一 NCCL group。
group 的打开、提交和流等待属于算子执行，因此设备图回放也保留该顺序；消费者在
group 提交和 join 后读取结果。各梯度保持自己的形状、dtype 与存储，FP32 均值使用
NCCL 的原生 average，其它梯度类型保留求和再缩放的通路。CPU 与其它后端保留原来的
逐梯度同步路径。分组减少发射和等待次数，本身不表示通信与计算已经重叠。

被捕获的单个 CUDA DDP 实例在 FP32 梯度达到 `bucket_cap_mb`（默认 25 MiB）时，
按首次普通 backward 中梯度计算节点的构建顺序估计就绪顺序，跳过 tape 与存储包装，
再由 rank 0 广播参数编号并缓存为各卡一致的打包顺序。这样共享参数的晚到梯度不会
仅因注册顺序堵住所有桶。捕获时验证缓存顺序；没有匹配的预热顺序时仍使用各卡一致的
反向参数顺序。`nccl_all_reduce_deferred` 在通信流提交归约并保留输入／
输出分配，后续计算可以继续；`nccl_wait_many` 等全部桶提交后才执行 join，返回的
别名才可被读取或更新。低层 deferred 结果必须经过此 join，不能直接消费。
普通别名 clone 在源就绪后优先发布，避免反向完成后才创建的 detach 节点拖延整个桶。
CUDA buffer 的批量广播使用 `nccl_broadcast_many`，由同一执行器线程打开、提交和
等待 group，再保留原存储写回。小张量按 dtype 打包，普通桶不超过 64 MiB，超过
上限的单个大张量直接传输；每次仍广播当前 buffer 内容。捕获时保留图内写入，
避免只在 tracing 时执行物理拷贝。不能跨执行器线程用 Python scope 配对 NCCL group。
这些路径保留真实通信与完整 FP32 计算，是否重叠仍需用同设备／context 的时间区间验证。

CUDA 上零 weight decay 的 Adam 可以复用 AdamW 的融合更新，因为此时两者公式相同。
更新仍读取当前学习率、betas 与每参数计数；显式 `fused=False` / `foreach=False`
可以禁用默认融合，带耦合 weight decay 的 Adam 保持自身计算路径。默认 Adam 的公开
step Tensor 仍在 CPU，显式 fused 计数保持参数设备位置。

捕获中的 CUDA FP32 `linear` 将前向 bias 加法并入 cuBLASLt GEMM；反向仍计算输入、
权重和 bias 的完整梯度。`addmm` 在 `alpha=beta=1`、连续二维矩阵与向量 bias 的训练
路径上直接使用 `[cin, cout]` 权重布局，权重梯度按相同布局写回，并用 BGRADA 求 bias
梯度，省去权重及其梯度的转置。其它缩放、形状、dtype 和设备保留原来的矩阵运算路径。

## 16. 相关文档

- [分层架构](torch-architecture.md)：统一计算图、下沉与优化的整体设计；
- [Torch 兼容 API](../../compatibility/torch.md)：支持范围与验证覆盖；
- [Torch shim](../../compatibility/torch-shim.md)：部署、激活模式与扩展构建；
- [流水式惰性执行](../../notes/runtime/pipelined-execution.md)：下沉后的图何时真正执行。
