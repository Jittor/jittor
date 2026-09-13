# PyTorch 调用如何快速下沉为 IR：关键技术点

- 状态：现行实现说明
- 上次复查：2026-09-11
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
   [`agent/skills/torch-api-cohort-promotion/SKILL.md`](../../agent/skills/torch-api-cohort-promotion/SKILL.md)。
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

## 15. 相关文档

- [分层架构](torch-architecture.md)：统一计算图、下沉与优化的整体设计；
- [Torch 兼容 API](torch.md)：支持范围与验证覆盖；
- [Torch shim](torch-shim.md)：部署、激活模式与扩展构建；
- [流水式惰性执行](../notes/pipelined-execution.md)：下沉后的图何时真正执行。
