# Native runtime, state and submission

[Section contents](../index.en.md) · [中文](native-runtime.md)

Read this page when changing runtime state, holder lifetime or graph submission. Start with [source architecture](../source-architecture.en.md) for module ownership.

These are implementation contracts, not a new hardware acceptance report. The source document's review baseline was `e3c369acb`; this split preserves that evidence boundary.

## Root module ownership

The only Python files that exist directly under `python/jittor/` are `__init__.py` and
`selftest.py`; counting `__init__.pyi`, the root has three source files in total. Argument
policy is in `_core.arg_policy`, namespace composition and installer ordering in
`_runtime.composition` and `_runtime.install_order`, and the timing API in `tools.benchmarking`.
Historical imports remain explicit same-object aliases.

`__init__.py` publishes the explicit `jittor._core.api.__all__` after the compiled-core
bootstrap. The composition module imports canonical objects from `_core.var`, `module`,
`function`, `hooks`, `flags` and `diagnostics`; **the implementation no longer shares one giant
`core_api.py` namespace**. Native save/load and the safe-pickle algorithm belong to
`serialization.native` and are re-exported by the same composition layer. `jittor._core.module`
owns the native Module implementation. Public root exports keep object identity, and historical
root-level pickle paths still load. The historical `jittor._runtime.core_api` import resolves
to the **same module object** as `jittor._core.api`, with no second implementation.

`_core/__init__.py` **does not export** callables or state objects named `var`, `flags` or
`hooks` -- those package attributes must keep resolving to their respective modules; object-level
exports belong to `_core.api` and the public Jittor root. API composition installs the exit hook
after importing its implementation domains, preserving the existing registration order.
`__init__.pyi` owns the type-annotation surface of the public root. `_core.flags` constructs the
single native Flags object, its runtime context and the runtime-scope API; both native and Torch
composition keep **the same object**.

### Configuration, runtime switches and introspection

`_runtime.flag_policy` classifies native flags for both binding generation and the Python API.
`_runtime.state` provides the immutable `jt.config`, the writable `jt.runtime` switches and the
read-only `jt.runtime.context` diagnostics. Runtime writes call the **original native setters**,
preserving their side effects; `jt.runtime.scope(...)` uses the existing reentrant flag-scope
implementation. Snapshots contain detached Python values, including copies of mapped values.
Execution and allocator counters are read-only through both the runtime and the old Flags
object.

`jt.introspection` provides three layers of supported read-only observation: **capabilities**
queries the named backend/device registries and existing library evidence; **policy** views
forward startup configuration and effective runtime settings; **counters** observe the
executor, allocator and live native graph services. **There is no second copy of Runtime state,
no implicit optional-library loading, no graph flushing and no writable observation path.**

Startup configuration includes compiler/tool paths, compile options, cache/source paths, CUDA
architectures and the cache lock policy. After backend post-processing and compatibility
composition finish, a **one-way native seal** rejects writes through any `Flags` instance,
including `jt.flags`, `compiler.flags` and `core.Flags()`; the compiler module also rejects late
public assignment to those fields. `jt.config` captures a detached immutable snapshot, with
architecture lists turned into tuples. **Set startup options through the environment before
importing Jittor.** New operators still accept local `extra_flags` and per-operator
`compile_options`; loading an extension **does not reopen startup configuration**. The
classification file participates in the binding generator's build fingerprint.

### Asynchronous submission boundaries

CUDA and ACL backend policies admit newly constructed inference or training-forward batches to
the worker while the calling thread builds the remaining graph; ACL also requires
`backend_fallback=error`. Backward graph construction and Python backward callbacks stay on the
calling thread. `src/runtime/async_exec.*` and the executor own thread placement, graph locking,
read waits and the policy that stands down after a second Python calling thread appears.

Every forward and training step still creates fresh operators; worker submission does not reuse
an execution graph or model results. `async_execution=0` disables the worker, while
`auto_flush_ops` and `async_flush_ops` control submission boundaries. Shapes and batch sizes
affect the tradeoff between earlier materialization and concurrent submission; compare matching
configurations.

### Native state and lifetime

The native held-root storage is in `src/runtime/holder_state.{h,cc}`. `RuntimeHolderState` owns
both the holder list and the weak-synchronization cursor; the executor, autograd, graph
inspection and memory diagnostics share the core accessors it exports. **It never runs the graph
when a holder is registered or removed.** Weak synchronization peeks first, then checks the
target cutoff, and advances only after that check. Removal repairs the cursor before liveness
release may re-enter the runtime. This owner is non-copyable and has process lifetime, to support
late extension/static holder destruction; it **does not own** the holders it points at. This
preserves the existing "changes must be serialized" requirement, **not a new thread-safety
guarantee**.

`src/runtime/runtime.{h,cc}` owns the process-lifetime `NativeRuntime`, which contains the
executor, held-root, traversal and device state. `runtime_executor()` and
`runtime_holder_state()` resolve to **the same owner** from core, JIT operators and backend
libraries. The executor starts with a null allocator pointer; **construction neither selects a
device nor initializes a backend.** Fork initialization resets its existing device state without
rebuilding inherited holders. `runtime_traversal_state()` shares stamp counters and the active
epoch count between core and JIT libraries. `TraversalEpoch` lives in `src/runtime/`; nested
traversal marks are restored before leaving the epoch, **including during exception unwinding**.
Fork preserves the counters to avoid clashing with inherited node stamps.

`runtime/device_state.h` holds `use_cuda`, `device_id`, `sync_run`, cached device count/current
device, device-switch hooks and peer-access accounting. `runtime/device.{h,cc}` provides device
operations; the old `misc/cuda_flags.*` is deleted. `DEFINE_RUNTIME_FLAG` registers
Python/environment access through exported storage accessors, **not through global variables or
dynamically initialized global references**. Setter correction, rollback and backend-switch
flushing keep their original order. CPU-only operator routing keeps a constant CPU, while flag
binding still reads the real runtime state. ROCm's callback selection is stated explicitly in
the header rather than relying on the old binary translator recognizing an old filename.
Domain-specific native flags not listed above as runtime-owned remain in their own modules --
**the Python-side lifetime split does not claim to have moved all C++ storage**.

`runtime/jit_policy` owns the math policy for CUDA kernels (`default`, `strict`, `backend`).
Ordinary and fused CUDA keys capture the policy before compilation, and the compiler translates
the captured value into launch flags. Switching policy first submits pending graphs under the
**old policy**. The explicit per-operator flag remains a local override. Torch preflight selects
this runtime policy instead of rewriting startup NVCC flags or creating another core build
configuration just to change kernel math. ACL preflight still removes the historical CUDA-only
strict flag from its startup environment.

### ABI boundaries for native extensions

The previously exported `Executor exe` data symbol is deleted. In-tree CUDA/ACL consumers and
the embedded CUDA template use `runtime_executor()` from `executor.h`. **Out-of-tree C++
extensions must update that access path and rebuild against the new header/core library; old
prebuilt extension binaries are incompatible.** The `tflag_count` symbol and the
`misc/traversal_epoch.h` path are likewise deleted; extensions using traversal internals must
switch to the runtime headers and rebuild. The `use_cuda`, `device_id` and `sync_run` data
symbols are deleted as well -- external native consumers must include `runtime/device.h` or
`runtime/device_state.h` and rebuild; **the Python `jt.flags` names are unchanged**.

### Domain packages and build tools

The implementations of the compiler, external-library setup, binding generation and CuPy
bootstrap are all under `build/`; the historical root-module imports are same-object aliases.
`distributions/`, `init/` and `linalg/` are public native packages; `selftest.py` is the
installed smoke-test entry point. **A new root-level file requires an ownership review and a
corresponding structural-gate update.**

The `__init__` files of these three domains explicitly re-export their implementation objects.
Linear algebra separates complex routines, decompositions, solving, norms and contractions, and
shares array helpers and result types. Distributions separate base contracts, constraints,
helpers, discrete/continuous/relaxed/multivariate families and KL divergence. Initialization
separates base filling, fan/gain rules, scaled initializers and truncated normal, and its facade
keeps the existing Var method bindings. Function metadata names the physical owner, while
historical public pickle globals continue to resolve through the facade. Every `.py` file under
`python/jittor` is now under 1,500 lines. The formerly huge compatibility NN, numerical, tensor
and FlashAttention modules have become ordinary packages with independent implementation
owners. **This source decomposition does not mean the standalone Torch architecture migration is
finished.**

Framework imports needed only at runtime are deferred to the call site, so that the import-cycle
surface does not keep growing. Six historical complex linear-algebra functions are lazily
re-exported as their original objects, preserving the concrete ComplexNumber type annotations
without making package bootstrap depend on the NN facade.

Contributed algorithms physically live in `contrib/{ccl,loss3d,math_util,einops}`. Their
historical package and submodule paths are **resolved on demand** as deprecated same-object
aliases, rather than eagerly importing all four domains. The `igamma.h` resource moves with its
owner into `contrib/math_util/src`; the source checkout and the installed resource lookup use
**the same module-relative path**. The public `contrib` package also owns the historical
composition helpers; `jittor.compat.contrib` is now an alias of it. Alias publication keeps the
package functions of the same name (such as `math_util.igamma`, `ccl.ccl_2d`), **rather than
replacing them with submodule objects**. einops parsing and transformation share the same
`EinopsError` class in `einops._errors`, re-exported by the public package, without an
"implementation-to-facade" import cycle.

The standalone build tooling physically lives in `build/utils` while keeping the runtime
namespace `jittor_utils`, so that tool imports can run **before** the Jittor core bootstrap. The
`jittor.build.utils` import is an alias of the same objects. This separates physical ownership
from the standalone bootstrap namespace without introducing duplicate tool implementations or an
eager reverse import of Jittor. The source and dependency scanners check the real `build/utils`
tree, not just that standalone import bridge.

The compiler is split by responsibility: `build/compiler.py` keeps startup state and
orchestration, `build/codegen.py` owns source generation, and `build/compilation.py` owns
compilation and custom-extension operations. **The generator fingerprint includes the extracted
owners**, so one change cannot silently reuse a stale build stamp.

Backend configuration is now a **frozen** `BuildConfig` returned by the selected provider, with
explicit services in `BuildContext`. **A provider does not modify compiler globals or append to
its source list**; bootstrap publishes the compatibility attributes centrally. Entry points are
loaded only for the selected backend, and selecting CPU explicitly skips CUDA discovery. Build
tools receive their binding/compiler services by injection and **no longer import Jittor**.
Tensor checkpoint algorithms are in `jittor.serialization`, and native save/load and safe pickle
are in `serialization.native`; the historical tool paths query the loader injected by the
runtime bootstrap.

## Native support-layer layout

The native `BackendRegistry` in `runtime/backend*` is owned by `NativeRuntime`. It publishes a
version-checked callback table whose names it owns and whose callbacks have process lifetime.
**Registering the CPU and CUDA descriptors does not initialize the driver**: a pure CPU build
still "knows about" CUDA but reports zero devices. `runtime/backends/` implements raw pool
selection, device operations, copies, synchronization and streams. The public allocator code
keeps the SFRL/NFEF/Temp/Stat composition and obtains raw pools from the registry -- **never the
other way round**.

Array creation, host/device migration, device-copy operators, fetch and swap-out transfers all
call that interface. `allocation_device()` derives the physical device from the allocator; **it
does not mistake a Var's retained device affinity for its current residency**.
Double-staged and deferred-free storage report their real pool device. Ordered peer copies
preserve stream dependencies for both source and destination, and fetch holds the block through
its callback. The old CUDA stream functions remain as adapters over registered stream hooks,
with the shared implementation in `src/runtime/backend_streams.cc` and the SDK operations in the
corresponding backend driver.

`core.registered_backends()` and `core.backend_device_count(name)` query the native registry.
Four historical accelerator-mode aliases in `jt.flags` emit a `DeprecationWarning` but keep their
setter behaviour. ACL publishes the canonical name `acl`; the old spelling `acl_legacy` resolves
to **the same** native descriptor and Python kernel table, without registering a duplicate
implementation. Python-side selection consumes the native device context; **there is no separate
Python allocator and no hard-coded backend capability prototype**.

## Distributed ownership

`jittor.distributed.process_group` owns `ProcessGroup`, `Work`, communicator creation and live
native rank/world queries. Launch and rendezvous are owned by `distributed.launch` and
`distributed.store`. The Torch installers delegate to these objects, keeping only Torch
spellings, argument adaptation and their own install/bootstrap transactions. The historical
`_JittorProcessGroup` and `_JittorWork` imports remain aliases, so old pickle global symbols
resolve to the same canonical classes. **Illegal native world-size metadata now raises**, rather
than silently being treated as a single-process runtime.

## Dynamic shapes and submission boundaries

Operator constructors and `Op::init()` establish edges, infer shape metadata and capture the
autograd policy, **without submitting work or calling callbacks**. A negative dimension records
the allocation upper bound of a data-dependent output, and is replaced by the actual non-negative
shape when the operator executes.

- `schedule_pending_from_python()` runs after a complete Python holder has been created;
  `Executor::submit_pending()` resolves the subgraph needed for dynamic results there, preserving
  the existing "Python sees concrete shapes" contract. Ordinary static results follow the
  configured lazy/immediate submission policy.
- Native C++ composition needs no intermediate Python holder. `Executor::run_sync()` resolves
  internal dynamic inputs in dependency order before constructing the execution plan, then
  refreshes downstream `infer_shape()` metadata -- this is an execution-time operation, **never
  construction-time recursion**. A dynamic output still unresolved after an explicit submission
  raises, rather than publishing a negative concrete shape.
- `VarHolder::item()` uses `sync(false, false)`: it submits the requested dependency graph
  without weak synchronization spreading to sibling branches. CUDA device-to-host copies use an
  event on the producer stream; the count readback of native where, candidate and CUB where goes
  through the same `backend_copy()` gateway, with allocator and GIL policy still owned by that
  gateway and the executor entry scope.
- Fetch construction only creates the fetch node. The public Python `fetch()` calls
  `core.submit_pending_fetches()` after construction, executing ready inputs and applying bounded
  pending-fetch backpressure; native callers may use the same explicit boundary or submit their
  own fetch graph normally. Device waits during shutdown are cleanup and are not affected by this
  contract.

`jt.submit_pending(*vars, device_sync=False)` is the boundary that **explicitly submits selected
pending roots**: every argument must be a `Var`, and its pending producer graph is submitted
through the existing `Var.submit_pending` binding; other holder roots stay pending, and the normal
lazy/automatic flush policy is unchanged. With `device_sync=True` these roots are also
synchronized so the host can consume them immediately. It returns the single root or a tuple of
roots, preserving object identity; empty input and non-Var arguments raise clear Python errors.
It is aimed at Function callbacks that know their exact output roots, fetch bridges and execution
adapters; **it does no implicit global flush, calls no GC, and does not make unrelated pending
graphs executable**. The implementation delegates to the existing executor partial-submission
path and owns no second graph, queue, liveness count or scheduling policy; CPU and accelerators
use the same boundary.

Keep these ownership rules during parallel development:

- A new dynamic operator implementation declares upper-bound output metadata, a pure
  `infer_shape()`, and execution-time final shape publication; **do not** revive the
  construction-time `run_sync()` workaround.
- The frontend submits only after both its language objects and the graph edges exist, and does
  not duplicate the executor's shape-dependency traversal.
- Backend providers implement ordered copies and event waits behind runtime callbacks; operator
  families do not introduce whole-device readback waits.
- Changes to construction, `submit_pending`, shape resolution and fetch backpressure are **one
  core-interface change** and must be coordinated together; ordinary operator or frontend
  additions only consume these interfaces.

Targeted regressions: `tests/core/test_dynamic_shape_submission.py`,
`tests/core/test_partial_graph_submit.py`, `tests/ops/test_where_op.py`,
`tests/data/test_fetcher.py`, and `tests/structure/core/test_var_holder_submission.py`.
