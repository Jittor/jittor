# Torch 兼容的分层架构：一套计算图，多个前端

- 状态：现行实现说明
- 上次复查：2026-09-11
- Owner：Torch 兼容维护者
- 复查触发：新增框架前端、图或代码生成路径变化、回退策略调整

`jittor.compat.torch` 让按 PyTorch 写的代码不改一行跑在 Jittor 上。支撑它的不是
第二套张量库，而是一条单向流水线：

```text
框架 API（torch / torch.nn / torch.library / autograd / optim / cuda ...）
        │  ① 下沉：拼写与语义适配
        ▼
统一计算图（原生 Var/Op 图，唯一 IR）
        │  ② 图融合 + JIT 代码生成
        ▼
后端 kernel（CPU / CUDA / ACL / ROCm / Corex）
```

| 阶段 | 要回答的问题 | 代码归属 |
| --- | --- | --- |
| 构建 IR | 图与节点是什么、算子在哪里注册 | `src/core/`、`src/ops/op_register.h` |
| 框架转 IR | `torch.*` 调用如何变成图上的节点 | `compat/torch/` |
| 算子优化 | 图如何融合、编译并落到设备 | `src/core/fuser*`、`src/codegen/`、`backends/` |

`jittor-torch` 发行物的描述把边界写得很直接：**"Independent Torch frontend using
Jittor's native Var/Op graph"**（`compat/pyproject.toml`）。兼容层不建立第二张图，
也不复制数学实现；它只负责把框架语义翻译到这张图上。

## 1. 第一层：统一计算图（构建 IR）

### 1.1 图、节点与惰性执行

计算图的实现在 C++ 核心里：节点与变量在 `src/core/node.h`，算子在
`src/core/op.h`，图容器在 `src/core/graph.h`，变量的存储与生命周期在
`src/core/var_holder.h`。执行是惰性的——Python 侧先构造图，图在读取、同步或显式
提交时才被调度执行（见[流水式惰性执行](../notes/pipelined-execution.md)）。

### 1.2 算子注册表就是 IR 的算子契约

`src/ops/op_register.h` 里的 `OpDef` 是一个算子在图上携带的全部信息：

- `name`、`constructors`：算子的名字与构造入口；
- `Codegen codegen`：JIT 源码生成接口（源码片段、元数据、优化钩子）；
- `std::map<BackendId, OpImplementation> implementations`：**每个后端一行的 kernel 表**；
- `compile_identity` 与 `OpId`：进程内稳定的身份，供缓存键与"已实例化的图钉住
  自己的定义"使用。

`NativeOpRegistry` 负责注册与查询；执行器、并行编译器、tracer 都只通过这张表找
实现。**一个算子支持哪些设备，由注册表的行决定，不由 Python 分支决定。**

### 1.3 "唯一"是如何保证的

原生 Jittor 与 Torch 前端共用同一份 `VarHolder` 数据和同一张图，差别只在 Python
侧返回哪种类型。接缝在 C++：`src/bindings/pyjt/py_tensor_frontend.{h,cc}` 用
`PyContextVar` 记录"当前应构造哪个前端类型"，缺省回退到原生 `Var`。Python 侧在
`compat/torch/frontend.py` 的 `tensor_frontend()` 作用域里进入这个选择：

```python
token = backend.core._set_tensor_frontend_type(tensor_type)     # 结果类型
precision_token = backend.core._set_float32_precision(...)       # 精度策略
placement_token = backend.core._set_tensor_placement(...)        # 构造期设备
with backend.autograd.policy_scope(backend.autograd.EXPLICIT_REQUIRES_GRAD):
    yield
```

因此 `torch.Tensor is not jittor.Var`（见 [Torch shim](torch-shim.md)），但它们背后
是同一份 payload、同一张图、同一批 kernel。

## 2. 第二层：框架转 IR（PyTorch → 原生图）

### 2.1 激活边界：显式、进程级、可回滚

兼容不是"导入即生效"：`import jittor` 保持原生行为，只有
`jittor.compat.shim.activate()`、`JITTOR_TORCH_SHIM=1`、部署后的 `import torch`
或历史别名 `jittor.torch_compat` 才建立 Torch 命名空间。安装按
`compat/torch/__init__.py` 的步骤表执行（`core → tensor → nn → optim → autograd
→ cuda → distributed → ...`），失败经由 `compat/transaction.py` 的事务账本整体
回滚；未实现的路径由 `compat/stub_policy.py` 显式报错，而不是静默 no-op。

发布面是 `TorchNamespace`（`compat/torch/namespace.py`）：一个模块形状的视图，
在全部显式绑定完成前允许向原生 owner 借读，最后 `_seal()` 封闭，避免原生属性
泄漏进 `torch.*` 命名空间。

### 2.2 类型下沉：前端类型是原生类型的子类

`compat/torch/frontend.py` 的 `make_tensor_type()` 动态生成 `torch.Tensor` 子类，
`compat/torch/nn_frontend.py` 为 `nn.Module` / `nn.Parameter` 生成原生类的子类。
类型本身不携带计算逻辑；构造与运算经上面的 ContextVar 落回原生图。
`nn_adoption.py` 在每层模块构造后把原生子模块"认领"为前端类型，使 `isinstance`
与 `state_dict` 语义一致。

### 2.3 语义适配：把 torch 拼写翻译成图节点

`compat/torch/installers/` 按 `torch.*` 家族拆分（tensor、nn、numerical、cuda、
distributed、compiler ...）。每个安装器做的是签名归一、dtype/device/axis 规则与
边界语义，然后把调用交给原生算子。例如 `torch.cat` 处理 `axis=` 别名、
NestedTensor、空张量与 0-d 补维后落到原生 `jt.concat`，`torch.stack` 同理。

对"拼写与原生一致"的数学 API，用显式委托表而不是逐个写包装：
`compat/torch/native_api.py` 的 `_NATIVE_NAMES`（百余个算子，含
`matmul`/`einsum`/`gather`/`scatter` 等）与 `_NATIVE_MODULE_NAMES`
（`fft.*`、`linalg.*`）把每个入口注册成稳定的 `NativeOperation` 对象，调用时在
`tensor_frontend()` 作用域内执行原生实现。**委托是声明式、可枚举的**，不存在
"任意属性透传"。

### 2.4 调度表：`torch.library` 与 dispatch key

`compat/torch/library.py` 实现 `torch.library` / `torch._ops` 的可用子集：
`_RegisteredOp` 按 `_CUDA_DISPATCH_ORDER` / `_CPU_DISPATCH_ORDER` 的顺序（含
autocast key）为调用选择 kernel；只注册了 `Meta` / `FakeTensor` 的算子**显式
报错**，因为 meta kernel 只返回形状正确的空张量，会静默给出假数值。
`register_autograd` 的反向通过 `_LibraryAutograd(jt.Function)` 挂到原生 tape 上，
使自定义算子的梯度与原生图一致。

### 2.5 可审计性：保真度台账

`compat/torch/fidelity.py` 为每个已发布 API 登记保真度：

| 级别 | 含义 |
| --- | --- |
| `EXACT` | 与 torch 契约一致（逐位或到文档声明的误差） |
| `APPROXIMATE` | 语义一致，但 dtype、边界或设备有限制 |
| `UNIMPLEMENTED` | 显式不支持，调用即报错 |

配合 `diagnostics.py` 的 `swallowed()`（被吞掉的异常逐条登记）与
`transaction.py`（安装可完整回滚），"支持了什么、不支持什么"是可查询的，而不是
靠读源码猜。

### 2.6 其他框架

当前只有 PyTorch 一个框架前端。`compat/triton/` 是 kernel 级桥（保留
`@triton.jit` 的 launch 语法，host 工具为真实现），不是框架前端；TensorFlow 没有
兼容层，NumPy 只作为内部工具与测试 oracle。**但流水线的形状可复用**：新增一个
框架 = 新增一个命名空间 + 一张下沉表 + 一份保真度登记，图与优化层不动。

## 3. 第三层：具体算子优化（在图上，一次优化所有前端）

优化全部发生在统一图上，因此对原生与 Torch 前端同时生效。

### 3.1 图级：融合

`src/core/fuser.{h,cc}` 统计可融合的算子链，`src/core/fused_op.{h,cc}` 把多个算子
合成一个融合算子，`src/core/exec_plan.h` 保存执行计划（`var_fused` / `fuse_ops` /
range）。`Var.stop_fuse()` 是显式边界，供需要独立执行语义的场景（调试、特定
后端）关闭融合。

### 3.2 编译期：pass 与调优

融合后的算子经 `src/codegen/op_compiler.cc` 生成 JIT 源码，
`src/codegen/opt/pass_manager.cc` 按序应用编译期 pass：向量化
（`vectorize_pass`）、展开（`unroll_pass`）、循环合并/重排/切分
（`merge_loop_pass`、`reorder_loop_pass`、`split_loop_pass`）、并行化
（`parallel_pass`、`cpu_parallel_pass`）、归约优化（`blocked_reduction_pass`、
`warp_reduce_pass`、`shared_reduce_pass`、`reduce_accumulator_pass`）、重步长
（`restride_pass`）与中间量消除（`remove_intermediate_pass`）等。形状与参数搜索
由 `jit_searcher` 与 `tuner/{conv,matmul,reduce,broadcast,reorder}_tuner` 承担。

> 这里没有**图级常量折叠**；`const_var_pass` 处理的是 codegen 期的常量变量。

### 3.3 后端落地：一行一个 kernel

`OpDef::implementations` 按 `BackendId` 选择实现，物理归属在
`backends/{cpu,cuda,acl,rocm,corex}/`。同一张图在 CUDA 与昇腾上都能执行，靠的是
同一算子在不同后端各注册一行，而不是前端分叉。

### 3.4 前端快路径：算子级替换，必须可观测

兼容层可以做算子级快路径，但要留下证据。典型例子是
`F.scaled_dot_product_attention`（`compat/torch/installers/nn/attention.py`）：
先尝试昇腾的 flash attention 实现，再尝试 flash-attn 外部后端；每次命中或未命中
都按原因记入 `sdpa_flash_stats`（`mask`、`rank`、`not_cuda`、`dtype` ...），未命中
时回落到原生实现。**快路径不改变语义，也不允许把"跑在 CPU 上"说成"跑在设备
上"。**

### 3.5 明确不做的事

- 不在兼容层重写图：没有前端私有的 IR、pass 或常量折叠；
- 不做静默回退：跨后端回退由原生 `backend_fallback` 策略控制，默认 `warn`，
  硬件门禁用 `error`；
- `torch.compile` / `torch.fx` 不做真实图分析：`compile()` 对 `fullgraph=True`、
  自定义 `backend=` 显式 `unimplemented`，`script` / `trace` 是 eager 透传并
  `degraded()` 告警。

## 4. 为什么这样分层

1. **数学只有一份。** Torch 前端与原生 Jittor 调用同一批原生算子；后端新增支持
   只在一个地方生效。
2. **优化只有一处。** 融合、JIT pass、调优器不需要为每个前端重写。
3. **差异集中且可审计。** Torch 与原生契约不同的地方（`.data`、view、别名、
   dtype 提升）集中在 `compat/torch/`，每条都有保真度登记与定向测试。
4. **设备与精度策略归原生 owner。** 前端作用域只决定"构造什么类型、放在哪、用
   什么精度策略"，执行与分配器状态由运行时持有。

## 5. 新增一个框架前端需要什么

按现有形状，接入一个新框架（"PyTorch 等框架"的扩展点）需要：

1. 一个显式激活入口与独立命名空间（不污染原生 `jittor`）；
2. 一张声明式的下沉表：框架 API → 原生算子，无法映射的路径显式报错；
3. 一份保真度台账与安装事务；
4. 类型适配层（前端类型是原生类型的子类，共享 Var/Op 图）；
5. 与原生测试分进程运行的回归门禁。

**图、融合与代码生成不需要任何改动**——这正是统一 IR 的收益。

## 6. 边界与代价

- **别名语义差异。** Torch 的切片、`.data`、`.numpy()` 共享存储，Jittor 的 Var
  不是；差异在"写进临时张量"的方向上是静默的，因此兼容层用显式规则与写穿数组
  处理，而不是假装共享。
- **编译过的扩展。** 任何自带链接真实 libtorch 的 `.so` 都无法被 Python 兼容层
  加载；`compat/shim/cpp_extension/` 提供一套零依赖的 libtorch ABI shim，让源码
  扩展按 Jittor 编译（见 [Torch shim](torch-shim.md)）。
- **图级 API。** `torch.fx` / `torch.compile` 的真实图分析不可用，属于架构性
  边界，不是待办事项。
- **性能。** 快路径必须有回退、可观测、可复现；性能结论按
  [性能基准](../performance/benchmarking.md)的口径给出。

## 7. 相关文档

- [调用下沉](torch-lowering.md)：第二层的展开——PyTorch 调用变成 IR 节点的关键技术点；
- [Torch 兼容 API](torch.md)：支持范围、语义差异与验证覆盖；
- [Torch shim](torch-shim.md)：部署、激活模式与扩展构建；
- [源码架构与模块边界](../development/source-architecture.md)：`compat/` 在仓库中的归属；
- [流水式惰性执行](../notes/pipelined-execution.md)：图什么时候真正执行。
