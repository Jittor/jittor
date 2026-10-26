# How PyTorch calls lower quickly to IR: key technical points

- Status: current implementation description
- Last reviewed: 2026-09-11
- Owner: Torch compatibility maintainers
- Review trigger: a new torch API, a change to installer orchestration or the
  lowering mechanism

This document expands layer two of the [layered architecture](torch-architecture.en.md):
**how PyTorch code becomes nodes on the unified computation graph at minimum
cost**. The goal is not "it runs" but "broad coverage, fast lowering, visible
differences".

## 1. The full path of one call

Take `torch.cat([a, b], 1)` as an example:

```text
torch.cat                   TorchNamespace 上的前端函数
  → 参数归一                axis→dim、NestedTensor、空张量、0-d 补维
  → jt.concat(...)          原生算子，在图里建一个 Node
  → 返回时构造前端类型      C++ 侧按 PyContextVar 决定返回哪个 Tensor 子类
```

Custom operators take another branch: `torch.ops.<ns>.<op>(...)` →
`_RegisteredOp.__call__` → select a kernel by dispatch key → ordinary kernels are
called directly, while those with registered autograd are wrapped in
`_LibraryAutograd` attached to the native tape.

The following sections cover the key technical points that make this path "fast"
and "stable".

## 2. Installer orchestration: step table + markers + transactions

Lowering rules are not scattered across import side effects but live in an
explicit step table (`compat/torch/__init__.py`):

- `_REQUIRED_STEPS`: more than twenty steps, in the order `core → tensor.base →
  tensor.methods → nn → optim → autograd → cuda → distributed → serialization →
  utilities → data → distributions → compiler → numerical → ...`, ending with
  each family's `*.module-keys` finalization;
- `_OPTIONAL_STEPS`: `torchmetrics` / `transformers` / `tensordict` /
  `safetensors` / `flash-attn` / `vllm`, where failures only warn.

Each step is `InstallStep = (name, installer)` (`compat/torch/contracts.py`);
before execution the signature is validated with `inspect.signature`
(`validate_installer`). `InstallContext` (`compat/torch/context.py`) is
responsible for execution:

- `run_required`: skips when the marker is already `"complete"`—**installation is
  idempotent**; failures are recorded through `swallowed()` and then
  `InstallStepError` is raised;
- `run_optional`: catches expected exceptions, sets the marker to `"failed"`,
  and warns only once;
- `mark_complete`: writes `_torch_compat_install_complete` on the namespace.

A failure leaves no half-finished state: after catching an exception, `install()`
rolls back the `InstallTransaction` (`compat/transaction.py`), restoring
attributes, markers, module publication, and the delegation table, and releases
the process lock. The success path `adopt`s child transactions into the parent
and then `commit()`s. At the end of the step table, `native_api.install(context)`
and `api_manifest.register_public_apis(context)` run to complete the explicit
delegation and the fidelity ledger.

## 3. ContextVar: zero-cost switching of type/precision/placement

The seam for "one Var, two frontend types" is in C++:
`src/bindings/pyjt/py_tensor_frontend.cc` stores the current frontend type with a
`PyContextVar`, and `_set_tensor_frontend_type` validates that it is a subclass
of `PyjtVarHolder`, falling back to native `Var` by default.

The Python side wraps it in a scope (`tensor_frontend()` in
`compat/torch/frontend.py`), setting four things at once on entry:

```python
token = backend.core._set_tensor_frontend_type(tensor_type)   # 返回哪种类型
precision_token = backend.core._set_float32_precision(...)     # float32 精度策略
placement_token = backend.core._set_tensor_placement(...)      # 构造期设备
with backend.autograd.policy_scope(backend.autograd.EXPLICIT_REQUIRES_GRAD):
    yield
```

Each is `reset` individually in `finally`. This design brings three things:

1. **No global class replacement**, so the default behavior of native
   `import jittor` is unaffected;
2. **Reentrant and context-safe**, with nested scopes restored by stack;
3. **Exception-safe**, with policies restored even on failure paths.

Note that placement is a "construction-time override" unrelated to runtime
execution flags—it decides which device new tensors land on, without changing
executor state.

## 4. Declarative delegation: one whitelist covers over a hundred operators

For mathematical APIs whose spelling matches the native one, wrappers are not
written one by one; instead a whitelist is maintained
(`compat/torch/native_api.py`):

- `_NATIVE_NAMES`: `matmul`/`einsum`/`gather`/`scatter`/`cummax`/`unique` and
  over a hundred other operators;
- `_NATIVE_MODULE_NAMES`: `fft.*`, `linalg.*`.

Each name corresponds to a stable `NativeOperation` object (with
`__name__`/`__module__`, supporting `__reduce__` serialization). At call time the
implementation is taken from the installation context and executed inside the
`tensor_frontend()` scope:

```python
implementation = context.state["native_torch_operations"][self._operation_key]
with tensor_frontend(context.state["Var"], device=device):
    return implementation(*args, **kwargs)
```

The delegation table is written into `context.state` by
`api_delegates.bind_delegates()`; it is an **immutable** `MappingProxyType`, and
old values are recorded with `transaction.record(...)`—rollback restores them
along with the ledger. **A whitelist rather than attribute passthrough** is what
makes it possible to register fidelity and write tests item by item.

## 5. Method tables: install a whole family of APIs at once

`torch.Tensor` methods are likewise table-driven:

| Table | Location | Contents |
| --- | --- | --- |
| `_BINARY_APIS` | `installers/tensor/method_api.py` | dunder binary operations such as `__add__`/`__mul__`/`__truediv__` |
| `_CAST_APIS` | same | type conversions such as `byte`/`short`/`long`/`half`/`float`/`bool` |
| `_UNARY_INPLACE_APIS` | same | in-place unary operations such as `neg_`/`exp_`/`sqrt_`/`sigmoid_` |
| `_SHAPE_REDUCTION_APIS` | `installers/tensor/shape_api.py` | reductions such as `mean`/`prod`/`any`/`all`/`max`/`min`/`argmax`/`var` |

The consumer is `_install_tensor_methods()` in `installers/tensor/methods.py`: it
first caches the native implementations (`_native_operators`), then overrides the
frontend `Tensor` subclass by table with `setattr`, and stores the native handles
in `ctx.state["tensor_native_api"]` (`MappingProxyType`). **Adding a family of
methods = adding one table entry**, rather than writing another wrapper
function.

## 6. Semantic adaptation happens only where necessary

Delegate directly when possible; wrap only for real contract differences.
`torch.cat` (`installers/tensor/__init__.py`) is typical:

- the `axis=` alias is normalized to `dim`;
- NestedTensor and the `__torch_function__` protocol are dispatched;
- empty tensors are skipped, 0-d tensors get a dimension, and then `jt.concat` is
  called;
- uint8 inputs are not lowered to int8 by the native operator.

Two recurring pitfalls have dedicated handling:

- `_array_keep_dtype()`: `jt.array` lowers int64 to int32, so tensors are
  constructed explicitly by dtype to preserve wide types;
- identity contract: rather than wrapping with something like `_axis_to_dim`
  (which would make the `torch.amax is Var.amax` assertion fail), the stable
  object itself accepts `axis=` and marks `_torch_accepts_axis = True`.

## 7. nn.Module: subclassing + parameter adoption + forward routing

`NNFrontendOwner` in `compat/torch/nn_frontend.py` generates the frontend types:

- `Module` is a **dynamic subclass** of native `jittor.nn.Module`;
- `Parameter` is a Var subclass generated by `frontend.make_parameter_type()`;
- `adapt_class(native)` generates and caches a subclass for each native layer as
  needed;
- `copy_module()` copies the `jittor.nn` module tree into the `torch.nn`
  namespace.

Construction-time adoption lives in `compat/torch/nn_adoption.py`:
`LayerInitializer` runs `adopt_owned_children()` after calling the native
`__init__`, replacing native submodules with frontend types via `ChildAdoption`
and replacing promoted parameters with frontend `Parameter` via
`ParameterRewrite`; objects in the `external` set are never rewritten.

The key to call routing is **replacing `Module._dispatch_call` rather than
`__call__`** (`installers/nn/module_methods.py`): this preserves the `__call__` →
hook → `_dispatch_call` path. `_prefer_forward()` uses the MRO to decide whether
the subclass's own `forward()` is more derived than the nearest `execute()`;
`_execute()` then falls back to the native implementation. First-publication
parameters as backward leaves, the real FSDP2 path, and pipeline logic all hang
off `_call`.

## 8. autograd: bridging to the native tape

`Function` in `compat/torch/autograd.py` inherits `jt.Function`, connecting
torch's trio to the native graph:

- `execute` → calls the user's `forward`;
- `__call__ = _call_record_inputs`: records input shapes, `needs_input_grad`, and
  outputs;
- `grad = function_grad`: materializes missing gradients (filling `jt.zeros` for
  floating outputs) and uses `_sum_grad_to` to reduce gradients back to input
  shapes.

`grad()` ultimately goes through `jt.core.grad_optional(...)`; `backward()` goes
through the Var's `_backward` and fills the optimizer's `pg["grads"]`. Backward
registered through `torch.library.register_autograd` is handled by
`_LibraryAutograd(jt.Function)`: the forward runs in `execute` and calls
`start_grad()` on floating outputs, and the backward calls the registered
`_backward`.

Precision and switches do not change the lowering path, only the behavior of the
native graph: autocast writes `jt.flags.amp_reg` (affecting native dtype
inference), `no_grad`/`enable_grad` go through native scopes, and `GradScaler`
only scales and skips steps at the optimizer gradient layer.

**Honest boundary**: `saved_tensors_hooks`/`save_on_cpu`/`Node` are
metadata-only placeholders, registered as `Fidelity.UNIMPLEMENTED`; no tensor
packing or offload is done.

## 9. Custom operators: dispatch keys and schema

`compat/torch/library.py` implements the available subset of `torch.library` /
`torch._ops`:

- `_RegisteredOp.select_impl()` picks a kernel in the order
  `_CUDA_DISPATCH_ORDER` / `_CPU_DISPATCH_ORDER`, `_argument_residency()` infers
  CPU/CUDA from the first Var argument, and the autocast key is inserted at the
  front according to the current policy;
- **operators with only `Meta`/`FakeTensor` registered raise explicitly**—a meta
  kernel returns only correctly shaped empty tensors, and using it to serve real
  calls would silently produce fake values;
- `custom_op` / `impl` (= `register_kernel`) / `register_fake` /
  `register_autograd` register the forward, fake, and backward respectively;
  `infer_schema()` derives the op schema from Python annotations.

Operators with a registered backward mark floating/complex results as requiring
gradients before the native output tape is built (even if the kernel detached the
result internally); integer/bool results are not marked. `Library`'s registry and
operator metadata writes participate in the current installation transaction
(`_transactional_registry = True`) and roll back with it on failure; see
[Torch installation transactions](../development/torch-install-transactions.en.md).

Downstream libraries' operators with the same name can be overridden by
integrations: `custom_op_overrides()` in `compat/integrations.py` provides the
replacement table, and when `custom_op` matches, it records
`_overridden_by_integration` rather than quietly swapping the implementation.

## 10. Visible and rollbackable failure

The lowering layer has four visibility mechanisms:

| Mechanism | Location | Purpose |
| --- | --- | --- |
| Fidelity ledger | `compat/torch/fidelity.py` | Records `EXACT`/`APPROXIMATE`/`UNIMPLEMENTED` item by item; `fidelity_table()` can export it |
| Explicit failure | `compat/stub_policy.py` | `unimplemented()` raises directly (only `JITTOR_TORCH_ALLOW_STUB` downgrades it to a warning); `degraded()` only warns |
| Swallowed-exception logging | `compat/diagnostics.py` | `EXPECTED` allowlist (excluding `NameError`/`AssertionError` and the like); `swallowed()` records with bounds, readable back via `torch.compat_swallowed()` |
| Installation transaction | `compat/transaction.py` | `record`/`record_object_diffs`/`rollback`/`commit`; a failed install leaves no half-finished state |

Together these four answer "to what extent is this API actually supported"—**no
need to read the source and guess**.

## 11. Another path: the source-level translator

`compat/pytorch_converter.py` provides **offline** PyTorch→Jittor source
rewriting:

- `pjmap`: PyTorch name → `jittor` module/name, with parameter renaming
  (`links`), forced extra arguments (`extras`), dropped arguments (`delete`),
  plus an `unsupport_ops` blacklist and the `pjmap_append()` extension point;
- `convert(code)`: `ast.parse` → `Converter` traverses and rewrites (import
  rewriting, `load_state_dict → load_parameters`, `size(...) → shape[...]`,
  `forward → execute`) → unparse into source; unmapped constructs generate a
  `raise` directly;
- `converter_server.py` is an optional HTTP frontend.

Its division of labor with the runtime shim: **if you can `import torch` and run,
use the shim** (broad coverage, same graph, with a ledger); use the translator
when you need to produce static Jittor code or the environment cannot install the
shim. The translator **builds no IR** and does not participate in
`TorchNamespace`.

## 12. Why it is "fast"

- **Install once, zero per-call interpretation**: delegation and methods are
  object attributes, and type selection is a single `PyContextVar` assignment,
  not a per-call branch;
- **Table-driven**: a family of APIs is installed with one `setattr`, and the
  cost of adding one is a single table entry;
- **Single-point adaptation**: differences exist only at the wrapper; everything
  else goes straight through to native operators;
- **Optimization reuse**: once lowering completes, the call enters native fusion
  and JIT code generation, so the frontend does not implement optimization again
  (see section 3 of the [layered architecture](torch-architecture.en.md)).

## 13. Checklist for adding a torch API

1. **Decide the owner first**: probe `hasattr(jt, name)` / `hasattr(jt.Var, name)`.
   Same contract → thin forwarding and capture the native object at import time;
   different contract → compat owns the wrapper and documents the difference; no
   native owner → compat is the final owner. The criteria are in
   [`agent/skills/torch-api-cohort-promotion/SKILL.md`](https://github.com/Jittor/jittor/blob/master/agent/skills/torch-api-cohort-promotion/SKILL.md).
2. **Register**: add to `API_PATHS` (`APPROXIMATE`) or `UNIMPLEMENTED_PATHS` in
   `compat/torch/api_manifest.py`.
3. **Fidelity**: use `register_fidelity()` to state the level and the restriction
   details.
4. **Three-part acceptance**: module-level identity (`torch.foo is <owner>.foo`,
   `__module__`, `__name__`) + `fidelity_of("torch.foo")` metadata + NumPy CPU
   fixed-point cross-check; device-related cases are then parameterized to the
   CUDA side.
5. **Structural gate**: `tests/structure/test_torch_api_surface.py` and the
   checked-in `torch_api_manifest.json`; when intentionally adding or removing,
   use `--regenerate` to sync in the same commit.
6. **Wrap up**: `JITTOR_TORCH_SHIM=1 python -m pytest tests/structure -q`, with
   the criterion that the set is **identical item by item** to before the change,
   not "no new failures".

## 14. Common pitfalls

- Treating an operator with only `Meta`/`FakeTensor` registered as a usable
  implementation;
- Not caching the native implementation before overriding a native method,
  making rollback impossible;
- Using wrappers like `_axis_to_dim` that break the object identity contract;
- Forgetting `_array_keep_dtype`, letting int64 silently lose bits;
- Using a broad `except` to swallow installation failures, turning a
  half-installed compatibility layer into apparent success;
- Treating CPU fallback as device support—after lowering, execution must happen
  on the declared real device.

## 15. Related documents

- [Layered architecture](torch-architecture.en.md): the overall design of the
  unified computation graph, lowering, and optimization;
- [Torch compatibility API](torch.en.md): support scope and validation coverage;
- [Torch shim](torch-shim.en.md): deployment, activation modes, and extension
  building;
- [Pipelined lazy execution](../notes/pipelined-execution.en.md): when the
  lowered graph actually executes.
