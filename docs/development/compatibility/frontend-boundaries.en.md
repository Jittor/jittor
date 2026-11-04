# Torch frontend ownership and dtype boundaries

[Section contents](../index.en.md) · [中文](frontend-boundaries.md)

Read this page when changing frontend installation, API identity or dtype conversion. Start with [source architecture](../source-architecture.en.md) for module ownership.

These are implementation contracts, not a new hardware acceptance report. The source document's review baseline was `e3c369acb`; this split preserves that evidence boundary.

For the frontend/graph design, see [Torch architecture](torch-architecture.en.md); rollback details are in [install transactions](install-transactions.en.md).

## Compatibility API

A standalone activation constructs its TorchNamespace before installation. The installers
publish APIs and context onto that target; a transactional "backend to owner" binding routes
leaf, retained-gradient and optimizer accounting to **the same state**. Historical native
attributes are aliases of that state, and a rolled-back installation step **replays against the
same state** on retry.

TorchNamespace owns its own public writes and deletions. A missing read may still use its native
owner; a deletion shadows that fallback locally. Transaction rollback restores **the exact**
local bindings and deletion state. InstallContext separates the installation target from its
native backend and **never inherits an install marker through a namespace fallback**. Publication
keeps the root's self-alias consistent between the registry and the import mapping. A standalone
activation owns real Tensor/Parameter subclasses and Module/NN adapters that reuse the native
Var/Op graph and mathematics, **without installing those APIs onto the native classes**. State
lookup uses private weak module bindings with transactional rollback; a standalone activation
**does not publish** leaf/retained/optimizer state aliases on the native modules, nor does it keep
an unloaded frontend alive through owner lookup.

The canonical Torch-style implementation is `jittor.compat.torch`. The historical
attribute/module spelling `jittor.torch_compat` loads its optional alias provider on explicit
import; **it is not a second source file**. Likewise the canonical Triton implementation is
`jittor.compat.triton`, with `jittor.triton_shim` kept as an object-identity alias.

The compatibility installers for the NN, numerical and tensor APIs, and the FlashAttention
adapter, are ordinary packages split by implementation family. A public callable that captures no
installation state may keep its module-level identity; **a stateful installation path keeps an
explicit context and its original registration order**. A native Parameter is a real Var
subclass; a native Module registers parameters and buffers by attribute name, **without reading
or writing Torch role markers**. Torch-specific assignment rules live in a separate Module
adapter or in explicit historical installers. The physical `compat/` tree is now the standalone
`jittor-compat` distribution, and **core packaging excludes it**. The explicit historical mode
still transforms native classes and **must not be conflated with a standalone activation**.

Basic indexing is tracked with native `VarView` rather than a parallel Python
`_torch_index_parent`/slice chain. Torch-specific slicing forms that have no native view record
yet attach one explicitly with `_set_view_of`. The detached `.data` API keeps its own
owner/path accounting; assignment uses a detached rvalue node, **so that stopping data aliasing
does not freeze its trainable owner**. The write-only strong-reference table for `requires_grad`
is removed. Leaf registration, retained-gradient tracking and optimizer registration are kept,
because they have real consumers. A standalone Tensor holder uses a weak identity index, while
leaf identity is determined by native graph queries. **An unrelated backward never erases the
registration of a live standalone holder**; retain_grad lasts until that holder's lifetime ends.
Its flag, gradients, data-view owner/path, device hint and RMSNorm cache all belong to
TensorObjectState. Historical non-weak-referenceable Vars keep their separate cleanup path.

`jittor.compat.shim` owns the runtime and deployment code for the optional top-level `torch`
interface, used by applications that import torch directly. `jittor.torch_shim` remains only as a
same-object historical alias. The shim delegates Torch-style semantics to `jittor.compat.torch`;
**neither the alias nor the deployed package owns a second implementation**.

Ordinary Jittor startup uses `_runtime.import_aliases` for native aliases and
`_runtime.compat_bootstrap` for optional activation. **It imports no `jittor.compat` module**;
only when compatibility is requested do missing optional packages produce installation errors.
Torch installers run after an explicit Torch-mode preflight, or through the deployed `torch` entry
point, or when the historical alias `jittor.torch_compat` is imported. **This prevents class-level
Torch transformations from changing the native Jittor API in unrelated processes.**

The vLLM implementation and its tests live in [jittor-vllm](https://github.com/JittorEcosystem/jittor-vllm), independently of Jittor and jittor-compat. The optional Torch stage discovers `jittor_vllm` in `jittor.module_patches`, targeting `jittor_vllm:register`. Its registrar assembles pre-import extension settings and post-import layer patches through the shared transaction mechanism. Missing installation is reported as `unavailable` without breaking Torch. Public Jittor primitives own the mathematical implementation; the downstream repository owns hardware acceptance. The root README links the ecosystem organization; the main repository has no adapter directory.

The ownership order is:

1. native Jittor semantics and generally useful operations;
2. reusable mechanisms in `jittor.compat`;
3. the optional import/deployment shim;
4. project-specific integrations outside the core distribution.

For the behavioural decision rules see
[Torch compatibility principles](principles.en.md). For which process state the
install and delayed hooks write, and which of it can be rolled back, see
[Torch install transactions](install-transactions.en.md).

## Runtime state and API ownership of the Torch frontend

A standalone frontend owns its own Tensor, Module and installation state, **without building a
second execution graph**; for APIs that can delegate to native mathematics, the native
implementation is authoritative.

- **Implementations have importable identities.** Installers publish existing module-level
  functions and classes, with the real implementation placed in the family module it belongs to;
  changing the `__qualname__` of a nested function does not count. A per-call algorithm callback
  may close over its inputs, but installation **must not** create a new public implementation on
  every attempt. Type factories only configure classes and bind module-owned behaviour:
  `NNFrontendOwner` in `nn_frontend.py` owns the layer-class cache, `LayerInitializer` and
  `nn_adoption.py` handle native construction and parameter adoption without modifying
  external/shared modules, and distribution construction, methods and properties use the
  descriptors in `distribution_adapters.py`, with one copy of state per frontend.
- **Resolve the owner at call time.** `get_install_context(native_backend)` resolves the
  explicitly bound frontend owner at call time: it is a read-only operation that **neither
  activates Torch nor creates an installation**, and it refuses when the owner is missing or
  inconsistent. Code that is currently installing uses the `InstallContext` it already holds. A
  captured native implementation is first placed in context-owned state and then published in
  the wrapper, so later lookups do not recursively hit the wrapper itself. The CUDA facade's
  mutable state (`CudaRuntimeState` in `installers/cuda/api.py`: logical streams, NVTX,
  memory-query cache and precision tier) also belongs to `InstallContext.state`; API identity is
  stable and the mutable state belongs to the current frontend.
- **Register fidelity truthfully.** The object finally published is registered with
  `register_fidelity`: `approximate` means limits remain, and `unimplemented` also covers
  existing annotations or no-op placeholders, not just functions that raise. Registration is
  keyed by API spelling; aliases may share an object while declaring different support behaviour.
  `api_manifest.py` records the final composition path after the family is installed,
  `fidelity_report()` returns sorted immutable records, and `fidelity_table()` renders the
  coverage table. **Neither registration nor namespace closure proves complete Torch semantics**:
  CUDA logical streams are serialized, event timing uses host timestamps after synchronization,
  and peak memory is sampled at query time; these are all registered limitations.
- **The root namespace is sealed after publication.** A standalone root namespace allows native
  fallback only during bootstrap; after publication an undeclared name raises `AttributeError`,
  and deleting a local name does not expose the native implementation. `native_api.py` declares
  stable `NativeOperation` objects for shared mathematical operations; they resolve a read-only,
  transaction-owned delegation table and enter the frontend's result-type/autograd scope. The
  native-only `flags`, `core` and `runtime` are not public Torch attributes. The FFT/linalg
  facades copy the public exports (excluding imported implementation modules) and keep their own
  namespace identity.

Runtime services and activation:

- `jt.runtime.service_state(namespace, factory=...)` owns extension state and **does not add
  private attributes to `jittor`**. A read without a factory creates no state; a successful
  factory runs only once per Runtime, a failed one can be retried, recursive construction of the
  same service is refused; the service itself decides the synchronization and transaction
  semantics of its fields.
- Shim activation uses the `jittor.torch.activation` service. **The install lock covers the
  activation-state check, the state transition and the installation itself**: concurrent callers
  wait, recursive activation on the same thread is refused, and a rollback conflict must release
  the lock and leave a queryable failure state (`jittor.compat.shim.activation_status()`).
- Tensor accounting uses `jittor.torch.tensor_states`, a Runtime-owned weak-owner table.
  `get_tensor_state()` resolves the explicit frontend binding, and `latest_optimizer()` resolves
  the weak optimizer reference. The historical leaf/retained/optimizer root aliases are adopted
  once and then removed, and installation failure rolls them back to their previous ownership.
- vmap's getitem-sinking hint is context-local and per-owner: use
  `TransformGetItemToIndex(owner)` and `getitem_transform_active(owner)`, not module attributes.
  Nesting and exceptional exit within the same scope restore the entering state, and another
  execution context does not inherit active modifications.
- `compat.torch.install` rejects a native module target before even creating an installation
  context; `activate` and bootstrap-time composition always choose the standalone namespace. The
  old false-mode argument and environment variable are explicit errors.

## dtype boundary

`torch.dtype` is an **immutable object, not a Python string**. `torch.long` and `torch.int64` are
the same object; `str`, `repr` and formatting all yield `torch.int64`; comparison with `"int64"`
or `"torch.int64"` is false. Pickling returns the canonical object and accepts the old
string-subclass pickle state. A dtype object is not callable: convert with
`tensor.to(dtype=...)`, and native Jittor code uses `jt.float32(...)`.

The boundary has two distinct operations, plus two registration points:

| Operation | Owner | Contract |
| --- | --- | --- |
| read a dtype name for metadata, dispatch, source generation or persistence | `jittor._core.dtypes.dtype_name(value)` | returns the bare canonical name; **grants no compute support** |
| consume a dtype before a Python/NumPy allocation | `jittor._core.dtypes.dtype_for_compute(value)` | goes through the native checked NanoString converter and rejects unsupported placeholders |
| register a frontend dtype class | `jittor._core.dtypes.register_dtype_type(type)` | registers the same exact class for Python positional dtype arguments and the C++ argument boundary |
| native dtype-argument consumers | `src/bindings/pyjt/py_dtype.{h,cc}` and `py_converter.h` | besides native and NumPy spellings, accepts only explicitly registered frontend types; any object with a `name` or `type` attribute is **not accepted** |

The native registry is shared by the core and its JIT extensions. The checked converter preserves
Python exceptions: a placeholder raises `NotImplementedError` before operator construction.
Canonical bare strings keep taking the native fast path; `torch.`-prefixed strings and registered
placeholder names resolve through the registered dtype objects. `ones`, `zeros` and `empty` also
recognize a registered dtype in the last positional argument -- otherwise treating it as a shape
component would silently fall back to the default float32.

The frontend defines 34 distinct dtype objects, 14 of which have a native allocation/compute
representation: bool, four signed integers, four unsigned integers, float16, bfloat16, float32,
float64 and complex64. The remaining 20 are used only by import-time dictionaries and metadata
and **cannot** be passed to factories, casts, random generation or the dtype argument of a native
operator. The set each backend actually supports is narrower than this framework-level set.

Native Python code **must not** treat `str(tensor.dtype)` as a kernel type name, nor compare a
frontend dtype object with a string; use `dtype_name` for both. Pass a dtype object directly to a
native operator, or use `dtype_for_compute` before handing it to NumPy. Preserve the dtype of
NumPy inputs; construct float64 directly rather than building float32 first and casting after
precision is lost. NumPy BF16 inputs are explicitly staged to float32 first and then go through
the native BF16 cast. Use `dtype_name` likewise when adding serialized state names or dtype
comparisons; do not reintroduce compatibility-layer imports into the native normalizer, and do
not restore string inheritance to make stale call sites work.
