# Layered architecture of Torch compatibility: one computation graph, multiple frontends

- Status: current implementation description
- Last reviewed: 2026-09-11
- Owner: Torch compatibility maintainers
- Review trigger: a new framework frontend, a change to the graph or code-generation
  path, or an adjustment to the fallback policy

`jittor.compat.torch` lets code written for PyTorch run on Jittor without changing
a single line. What supports it is not a second tensor library but a one-way
pipeline:

```text
框架 API（torch / torch.nn / torch.library / autograd / optim / cuda ...）
        │  ① 下沉：拼写与语义适配
        ▼
统一计算图（原生 Var/Op 图，唯一 IR）
        │  ② 图融合 + JIT 代码生成
        ▼
后端 kernel（CPU / CUDA / ACL / ROCm / Corex）
```

| Stage | Question it answers | Code ownership |
| --- | --- | --- |
| Build the IR | What the graph and nodes are, where operators are registered | `src/core/`, `src/ops/op_register.h` |
| Framework to IR | How `torch.*` calls become nodes on the graph | `compat/torch/` |
| Operator optimization | How the graph is fused, compiled, and lowered to devices | `src/core/fuser*`, `src/codegen/`, `backends/` |

The `jittor-torch` distribution description states the boundary directly:
**"Independent Torch frontend using Jittor's native Var/Op graph"**
(`compat/pyproject.toml`). The compatibility layer builds no second graph and
copies no mathematical implementation; it is only responsible for translating
framework semantics onto this graph.

## 1. Layer one: the unified computation graph (building the IR)

### 1.1 Graph, nodes, and lazy execution

The computation graph is implemented in the C++ core: nodes and variables in
`src/core/node.h`, operators in `src/core/op.h`, the graph container in
`src/core/graph.h`, and variable storage and lifetime in `src/core/var_holder.h`.
Execution is lazy—the Python side builds the graph first, and the graph is
scheduled and executed only when read, synchronized, or explicitly committed
(see [pipelined lazy execution](../notes/pipelined-execution.en.md)).

### 1.2 The operator registry is the IR's operator contract

`OpDef` in `src/ops/op_register.h` is all the information an operator carries on
the graph:

- `name`, `constructors`: the operator's name and construction entry points;
- `Codegen codegen`: the JIT source-generation interface (source fragments,
  metadata, optimization hooks);
- `std::map<BackendId, OpImplementation> implementations`: **a kernel table with
  one row per backend**;
- `compile_identity` and `OpId`: identities stable within the process, used for
  cache keys and for "an instantiated graph pins its own definition".

`NativeOpRegistry` is responsible for registration and lookup; the executor, the
parallel compiler, and the tracer all find implementations only through this
table. **Which devices an operator supports is decided by the registry's rows,
not by Python branches.**

### 1.3 How "only one" is guaranteed

Native Jittor and the Torch frontend share the same `VarHolder` data and the same
graph; the only difference is which type the Python side returns. The seam is in
C++: `src/bindings/pyjt/py_tensor_frontend.{h,cc}` uses a `PyContextVar` to
record "which frontend type should be constructed now", falling back to native
`Var` by default. On the Python side, the `tensor_frontend()` scope in
`compat/torch/frontend.py` enters this selection:

```python
token = backend.core._set_tensor_frontend_type(tensor_type)     # 结果类型
precision_token = backend.core._set_float32_precision(...)       # 精度策略
placement_token = backend.core._set_tensor_placement(...)        # 构造期设备
with backend.autograd.policy_scope(backend.autograd.EXPLICIT_REQUIRES_GRAD):
    yield
```

So `torch.Tensor is not jittor.Var` (see [Torch shim](torch-shim.en.md)), yet
they are backed by the same payload, the same graph, and the same set of kernels.

## 2. Layer two: framework to IR (PyTorch → native graph)

### 2.1 Activation boundary: explicit, process-wide, rollbackable

Compatibility is not "effective on import": `import jittor` keeps native
behavior, and only `jittor.compat.shim.activate()`, `JITTOR_TORCH_SHIM=1`, the
deployed `import torch`, or the historical alias `jittor.torch_compat`
establish the Torch namespace. Installation runs according to the step table in
`compat/torch/__init__.py` (`core → tensor → nn → optim → autograd → cuda →
distributed → ...`), and failures roll back as a whole through the transaction
ledger in `compat/transaction.py`; unimplemented paths raise explicitly through
`compat/stub_policy.py` rather than silently no-op.

The published surface is `TorchNamespace` (`compat/torch/namespace.py`): a
module-shaped view that may borrow reads from native owners until all explicit
bindings are complete, then `_seal()` closes it, preventing native attributes
from leaking into the `torch.*` namespace.

### 2.2 Type lowering: frontend types are subclasses of native types

`make_tensor_type()` in `compat/torch/frontend.py` dynamically generates a
`torch.Tensor` subclass, and `compat/torch/nn_frontend.py` generates subclasses
of the native classes for `nn.Module` / `nn.Parameter`. The types themselves
carry no computation logic; construction and operations fall back to the native
graph through the ContextVar above. `nn_adoption.py` "adopts" native submodules
as frontend types after each layer's construction, making `isinstance` and
`state_dict` semantics consistent.

### 2.3 Semantic adaptation: translating torch spelling into graph nodes

`compat/torch/installers/` is split by `torch.*` family (tensor, nn, numerical,
cuda, distributed, compiler ...). Each installer performs signature
normalization, dtype/device/axis rules, and boundary semantics, then hands the
call to a native operator. For example, `torch.cat` handles the `axis=` alias,
NestedTensor, empty tensors, and 0-d dimension padding before lowering to native
`jt.concat`; `torch.stack` works the same way.

For mathematical APIs whose spelling matches the native one, an explicit
delegation table is used instead of writing wrappers one by one: `_NATIVE_NAMES`
in `compat/torch/native_api.py` (over a hundred operators, including
`matmul`/`einsum`/`gather`/`scatter` and others) and `_NATIVE_MODULE_NAMES`
(`fft.*`, `linalg.*`) register each entry as a stable `NativeOperation` object,
and calls execute the native implementation inside the `tensor_frontend()`
scope. **Delegation is declarative and enumerable**; there is no "arbitrary
attribute passthrough".

### 2.4 Dispatch table: `torch.library` and dispatch keys

`compat/torch/library.py` implements the available subset of `torch.library` /
`torch._ops`: `_RegisteredOp` selects a kernel for a call in the order
`_CUDA_DISPATCH_ORDER` / `_CPU_DISPATCH_ORDER` (including the autocast key);
operators with only `Meta` / `FakeTensor` registered **raise explicitly**,
because a meta kernel returns only correctly shaped empty tensors and would
silently produce fake values. `register_autograd`'s backward is attached to the
native tape through `_LibraryAutograd(jt.Function)`, keeping custom-operator
gradients consistent with the native graph.

### 2.5 Auditability: the fidelity ledger

`compat/torch/fidelity.py` records fidelity for every published API:

| Level | Meaning |
| --- | --- |
| `EXACT` | Consistent with the torch contract (bitwise or to the documented error) |
| `APPROXIMATE` | Semantically consistent, but with dtype, boundary, or device restrictions |
| `UNIMPLEMENTED` | Explicitly unsupported; calling it raises |

Together with `swallowed()` in `diagnostics.py` (each swallowed exception is
recorded) and `transaction.py` (installation can roll back completely), "what is
supported and what is not" is queryable rather than guessed by reading the
source.

### 2.6 Other frameworks

Currently PyTorch is the only framework frontend. `compat/triton/` is a
kernel-level bridge (it preserves the launch syntax of `@triton.jit`, and host
utilities are real implementations), not a framework frontend; TensorFlow has no
compatibility layer, and NumPy serves only as an internal utility and a test
oracle. **But the pipeline's shape is reusable**: adding a framework = adding a
namespace + a lowering table + a set of fidelity registrations, with the graph
and optimization layers untouched.

## 3. Layer three: concrete operator optimization (on the graph, optimizing all frontends at once)

All optimization happens on the unified graph, so it takes effect for both the
native and Torch frontends.

### 3.1 Graph level: fusion

`src/core/fuser.{h,cc}` identifies fusible operator chains,
`src/core/fused_op.{h,cc}` combines multiple operators into one fused operator,
and `src/core/exec_plan.h` stores the execution plan (`var_fused` / `fuse_ops` /
range). `Var.stop_fuse()` is an explicit boundary that disables fusion for
scenarios needing standalone execution semantics (debugging, specific backends).

### 3.2 Compile time: passes and tuning

Fused operators generate JIT source through `src/codegen/op_compiler.cc`, and
`src/codegen/opt/pass_manager.cc` applies compile-time passes in order:
vectorization (`vectorize_pass`), unrolling (`unroll_pass`), loop
merging/reordering/splitting (`merge_loop_pass`, `reorder_loop_pass`,
`split_loop_pass`), parallelization (`parallel_pass`, `cpu_parallel_pass`),
reduction optimization (`blocked_reduction_pass`, `warp_reduce_pass`,
`shared_reduce_pass`, `reduce_accumulator_pass`), restriding (`restride_pass`),
and intermediate elimination (`remove_intermediate_pass`), among others. Shape
and parameter search is handled by `jit_searcher` and
`tuner/{conv,matmul,reduce,broadcast,reorder}_tuner`.

> There is no **graph-level constant folding** here; `const_var_pass` handles
> constant variables at codegen time.

### 3.3 Backend lowering: one kernel per row

`OpDef::implementations` selects an implementation by `BackendId`, with physical
ownership under `backends/{cpu,cuda,acl,rocm,corex}/`. The same graph can execute
on both CUDA and Ascend because the same operator registers one row per backend,
not because the frontend forks.

### 3.4 Frontend fast paths: operator-level replacement must be observable

The compatibility layer may provide operator-level fast paths, but it must leave
evidence. A typical example is `F.scaled_dot_product_attention`
(`compat/torch/installers/nn/attention.py`): it first tries Ascend's flash
attention implementation, then the flash-attn external backend; each hit or miss
is recorded in `sdpa_flash_stats` with its reason (`mask`, `rank`, `not_cuda`,
`dtype` ...), and a miss falls back to the native implementation. **A fast path
does not change semantics, and it is not allowed to describe "running on CPU" as
"running on the device".**

### 3.5 Things explicitly not done

- Do not rewrite the graph in the compatibility layer: there is no
  frontend-private IR, pass, or constant folding;
- Do not do silent fallback: cross-backend fallback is controlled by the native
  `backend_fallback` policy, defaulting to `warn`, with hardware gates using
  `error`;
- `torch.compile` / `torch.fx` do no real graph analysis: `compile()` raises
  `unimplemented` explicitly for `fullgraph=True` and a custom `backend=`, and
  `script` / `trace` are eager passthroughs with a `degraded()` warning.

## 4. Why layer it this way

1. **There is only one copy of the mathematics.** The Torch frontend and native
   Jittor call the same set of native operators; new backend support takes effect
   in only one place.
2. **Optimization happens in only one place.** Fusion, JIT passes, and tuners do
   not need to be rewritten for each frontend.
3. **Differences are concentrated and auditable.** Places where the Torch
   contract differs from the native one (`.data`, views, aliases, dtype
   promotion) are concentrated in `compat/torch/`, each with a fidelity
   registration and targeted tests.
4. **Device and precision policy belong to the native owner.** The frontend scope
   only decides "what type to construct, where to place it, and which precision
   policy to use"; execution and allocator state are held by the runtime.

## 5. What adding a framework frontend requires

Following the existing shape, integrating a new framework (the extension point
for "PyTorch and other frameworks") requires:

1. An explicit activation entry and an independent namespace (not polluting
   native `jittor`);
2. A declarative lowering table: framework API → native operators, with
   unmappable paths raising explicitly;
3. A fidelity ledger and installation transactions;
4. A type adaptation layer (frontend types are subclasses of native types,
   sharing the Var/Op graph);
5. A regression gate that runs in a separate process from native tests.

**The graph, fusion, and code generation need no changes at all**—that is
precisely the payoff of a unified IR.

## 6. Boundaries and costs

- **Alias semantic differences.** Torch's slices, `.data`, and `.numpy()` share
  storage; Jittor's Var does not. The difference is silent in the "write into a
  temporary tensor" direction, so the compatibility layer handles it with
  explicit rules and a write-through array rather than pretending storage is
  shared.
- **Compiled extensions.** Any `.so` that links against real libtorch cannot be
  loaded by a Python compatibility layer; `compat/shim/cpp_extension/` provides a
  zero-dependency libtorch ABI shim so that source extensions compile against
  Jittor (see [Torch shim](torch-shim.en.md)).
- **Graph-level APIs.** Real graph analysis for `torch.fx` / `torch.compile` is
  unavailable; it is an architectural boundary, not a to-do item.
- **Performance.** Fast paths must have fallbacks, be observable, and be
  reproducible; performance conclusions are stated according to
  [performance benchmarking](../performance/benchmarking.en.md).

## 7. Related documents

- [Call lowering](torch-lowering.en.md): an expansion of layer two—the key
  technical points that turn PyTorch calls into IR nodes;
- [Torch compatibility API](torch.en.md): support scope, semantic differences,
  and validation coverage;
- [Torch shim](torch-shim.en.md): deployment, activation modes, and extension
  building;
- [Source architecture and module boundaries](../development/source-architecture.en.md):
  where `compat/` belongs in the repository;
- [Pipelined lazy execution](../notes/pipelined-execution.en.md): when the graph
  actually executes.
