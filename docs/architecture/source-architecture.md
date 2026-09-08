# Source Architecture and Module Boundaries

- Status: Accepted
- Last reviewed: 2026-08-31
- Baseline: `f5e8e944` plus the boundary documentation changes described here
- Owner: Jittor core maintainers
- Review when: a public module moves, an implementation domain is added, or a
  runtime resource path changes

This document defines how Python source is decomposed inside Jittor. Repository,
packaging, and runtime-resource ownership is defined by the broader
[repository layout decision](repository-layout.md).

## Principles

1. **One physical owner.** A public domain is a normal package; it does not have
   a second private tree containing the real implementation.
2. **Imports follow ownership.** Implementation metadata and tracebacks report
   the module that owns the code. Re-export modules do not rewrite
   `__module__` recursively.
3. **Composition stays shallow.** A package `__init__.py` composes and exports
   public names. Large implementations live in meaningful child modules.
4. **Runtime paths are contracts.** Compiler-loaded C++/CUDA resources move only
   with an explicit compiler and packaging migration.
5. **Compatibility is layered.** Native framework capability, reusable
   compatibility mechanisms, import shims, and downstream integrations have
   distinct owners.
6. **Every move preserves behavior.** Refactors retain public names, callable
   identity where promised, pickling behavior where supported, backend dispatch,
   and focused regression coverage.

## Current domains

```text
python/
├── jittor/
│   ├── __init__.py              # root composition and runtime initialization
│   ├── __init__.pyi             # public root typing surface
│   ├── _core/                  # native Python API implementation domains
│   │   ├── api.py              # explicit composition after compiled-core bootstrap
│   │   ├── var.py              # tensor construction, operations and Var bindings
│   │   ├── module.py           # Module and parameter/buffer ownership
│   │   ├── function.py         # custom autograd contexts and gradient hooks
│   │   ├── hooks.py            # removable handles and hook support
│   │   ├── flags.py            # native scopes and the shared runtime state
│   │   ├── arg_policy.py       # explicit unsupported/ignored argument policy
│   │   └── diagnostics.py      # logs, profiling, process scopes and exit cleanup
│   ├── _runtime/
│   │   ├── core_api.py          # same-object legacy alias of _core.api
│   │   ├── composition.py       # explicit namespace publication
│   │   ├── install_order.py     # ordered installer registration and verification
│   │   └── state.py             # injected native Flags views, no bootstrap imports
│   ├── serialization/
│   │   └── native.py            # native save/load and safe-pickle implementation
│   ├── build/                   # compiler/bootstrap implementation ownership
│   │   ├── compiler.py          # startup state and build orchestration
│   │   ├── codegen.py           # native binding/registration source generation
│   │   ├── compilation.py       # compilation and custom-extension operations
│   │   └── utils/               # standalone utilities, imported as jittor_utils
│   ├── contrib/                 # contributed algorithms and composition helpers
│   │   ├── ccl/                 # connected-component labeling
│   │   ├── loss3d/              # Chamfer and earth-mover losses
│   │   ├── math_util/           # gamma functions and shared native resources
│   │   └── einops/              # vendored tensor-expression algorithms
│   ├── nn/                      # neural-network public API
│   │   ├── modules/             # stateful Module implementations
│   │   ├── functional/          # stateless tensor functions
│   │   ├── backends/            # cuDNN and read-only hook call adapters
│   │   ├── utils/               # construction helpers such as weight norm
│   │   └── attention.py
│   ├── autograd/                # functional automatic differentiation
│   ├── fft/                     # differentiable native FFT namespace
│   ├── linalg/                  # decompositions, solving, norms and contractions
│   ├── distributions/           # probability families and shared constraints
│   ├── init/                    # initialization families and shared fan/gain rules
│   ├── ops/                     # tensor, indexing, reduction and shape implementations
│   ├── misc/                    # deprecated same-object tensor API facade
│   ├── pool/                    # deprecated same-object pooling API facade
│   ├── optim/                   # optimizer facade and algorithm modules
│   ├── sparse/                  # COO tensors and sparse convolution
│   ├── compat/
│   │   ├── torch/               # canonical Torch-style API compatibility
│   │   ├── fsdp2/               # distributed FSDP2 compatibility
│   │   ├── triton/              # Triton API bridge and deployment command
│   │   ├── shim/                # Torch shim runtime and deployment command
│   │   ├── vllm/                # staged, relocatable vLLM integration
│   │   ├── module_patcher.py
│   │   └── external_backend.py
│   ├── selftest.py              # installed smoke test
│   ├── tools/                   # user tools, including benchmarking.py
│   ├── backends/                # source-checkout package path bridge
│   └── distributed/             # native launch, rendezvous and communication helpers
└── jittor_utils/                # standalone import bridge into jittor/build/utils
```

## Package composition contracts

### Neural network API

`jittor.nn` is the public package. Stateful layers live under `nn.modules`,
stateless operations under `nn.functional`, and optional accelerated paths under
`jittor.backends.cuda.kernels`. `nn.backends` retains call adapters, not CUDA
implementations. A public re-export must point at the canonical implementation
object; wrappers are justified only when they enforce a real API contract.

Dependency direction is:

```text
nn.modules -> nn.functional -> Jittor tensor/core operations
nn.backends ----------------> explicit backend/compiler interfaces
```

Functional modules must not import stateful layer implementations. Backend
adapters must remain optional and fail with an actionable capability error when
their toolchain is unavailable.

### Miscellaneous and pooling APIs

`jittor.ops` owns tensor, shape, indexing and composition operations. Historical
`jittor.misc` imports remain deprecated same-object facades, including the old
submodule paths. Public names and pickle globals resolve to the canonical
implementations; there is no second editable mathematics tree under `misc`.

Pooling mathematics and parameter validation live in the normal
`nn.functional.pooling` package, separated into average, 2-D/3-D core,
adaptive, 1-D and unpooling owners. `nn.modules.pooling` stores constructor
parameters and calls these stateless implementations; functional calls do not
construct a temporary Module. CPU/CUDA generated source remains single-source
with backend launch builders at the existing registration boundaries.

`jittor.pool` and its historical child modules are deprecated re-exports.
`nn.modules.pooling_legacy` keeps the three historical classes with their
original class names: `AvgPool2d` and `AvgPool3d` retain their forwarding layer
state for old pickles, while `AdaptiveAvgPool2d` retains its fixed-window
algorithm. The latter intentionally differs from the current NN overlapping-bin
algorithm; a layout move does not silently change its numerical rule. Adaptive
window intermediates are function locals, not persistent Module state.

The legacy `pool_use_code_op` switch has one owner in
`nn.functional.pooling._state`. `jt.pool`, `jt.nn` and the functional pooling
package expose live reads/writes of that same value, including temporary
attribute override/restore. Backend execution never reads a copied facade value.
NN constructs its functional and module owners before loading the old facade,
so the compatibility import does not create a bootstrap cycle.

`jittor.sparse` owns both coordinate-format sparse tensors and sparse neural
network kernels in separate child modules. The historical `jittor.nn.sparse`
name is a same-object alias of `jittor.sparse.convolution`; `jittor.nn` and
`jittor.nn.functional` re-export those canonical callables.

`jittor.autograd` owns functional automatic differentiation. `jittor.fft` owns
the differentiable FFT/shift/frequency namespace shared by native Jittor and
Torch mode. Concatenation and
indexing live in `jittor.ops`, pooling in `jittor.nn`, optimized softmax in
`jittor.backends.cuda.kernels.nn`, and weight normalization in `jittor.nn.utils`. Historical
root spellings are import aliases only; they do not retain physical source
files or wrapper implementations.

### Root module ownership

The only Python files directly under `python/jittor/` are `__init__.py` and
`selftest.py`; including `__init__.pyi`, the root has three source files.
Argument policy lives in `_core.arg_policy`, namespace composition and installer
ordering in `_runtime.composition` and `_runtime.install_order`, and timing APIs
in `tools.benchmarking`. Historical imports remain explicit same-object aliases.
`__init__.py` publishes the explicit `jittor._core.api.__all__` after compiled-core
bootstrap. The composition module imports canonical objects from `_core.var`,
`module`, `function`, `hooks`, `flags`, and `diagnostics`; implementations no
longer share one large `core_api.py` namespace. Native save/load and safe-pickle
algorithms belong to `serialization.native` and are re-exported by the same
composition layer. `jittor._core.module` owns the native Module implementation.
Public root exports retain object identity and
legacy root pickle paths remain loadable. The historical
`jittor._runtime.core_api` import resolves to the same module object as
`jittor._core.api`, with no second implementation.

`_core/__init__.py` does not export callable or state objects named `var`,
`flags`, or `hooks`: those package attributes must continue to resolve to their
modules. Object-level exports belong to `_core.api` and the public Jittor root.
The API composition installs exit hooks after importing its implementation
domains, preserving the existing registration order. `__init__.pyi` owns the
public root typing surface. `_core.flags` constructs the single native Flags
object, its runtime context and runtime scope API; native and Torch composition
retain that same object.
`_runtime.flag_policy` classifies native flags for both binding generation and
the Python API. `_runtime.state` provides immutable `jt.config`, writable
`jt.runtime` switches and read-only `jt.runtime.context` diagnostics. Runtime
writes call the original native setters, preserving their side effects;
`jt.runtime.scope(...)` uses the existing reentrant flag-scope implementation.
Snapshots contain detached Python values, including copies of mapping values.
Execution and allocator counters are read-only through both runtime and legacy
Flags objects.

Startup configuration includes compiler/tool paths, compiler flags, cache/source
paths, CUDA architectures and the cache-lock policy. After backend post-processing
and compatibility composition, a one-way native seal rejects writes through
every `Flags` instance, including `jt.flags`, `compiler.flags` and `core.Flags()`.
The compiler module also rejects late public assignments to these fields.
`jt.config` captures a detached immutable snapshot; architecture lists become
tuples. Set startup options in the environment before importing Jittor.
New operators still accept local `extra_flags` and per-op `compile_options`;
loading an extension does not reopen startup configuration. The classification
file participates in the binding-generator build fingerprint.

Native held-root storage lives in `src/runtime/holder_state.{h,cc}`.
`RuntimeHolderState` owns both the holder list and the weak-sync cursor; the
executor, autograd, graph inspection and memory diagnostics share its exported
core accessor. It never runs a graph when registering or removing a holder.
Weak sync peeks before checking the target cutoff and advances only after that
check. Removal repairs the cursor before liveness release can re-enter the
runtime. The owner is non-copyable and has process lifetime to support late
extension/static holder destruction; it does not own the pointed-to holders.
This preserves the existing serialized mutation requirement, not a new
thread-safety guarantee.

`src/runtime/runtime.{h,cc}` owns the process-lifetime `NativeRuntime`, containing
the executor, held-root, traversal and device state. `runtime_executor()` and `runtime_holder_state()`
resolve to that same owner from the core, JIT operators and backend libraries.
The executor starts with null allocator pointers; construction does not select
a device or initialize a backend. Fork initialization resets its existing device
state without recreating inherited holders. `runtime_traversal_state()` shares
the stamp counter and active-epoch count across core and JIT libraries.
`TraversalEpoch` lives in `src/runtime/`; nested traversal marks are restored
before leaving the epoch, including exception unwinding. Fork preserves the
counter to avoid colliding with inherited node stamps.

`runtime/device_state.h` stores `use_cuda`, `device_id`, `sync_run`, the cached
device count/current device, device-switch hooks and peer-access bookkeeping.
`runtime/device.{h,cc}` supplies device operations; the old `misc/cuda_flags.*`
files are removed. `DEFINE_RUNTIME_FLAG` registers Python/environment access
through exported storage accessors, not global variables or dynamically
initialized global references. Setter correction, rollback and backend-switch
flushing retain their previous ordering. CPU-only operator routing stays
constant CPU, while flag bindings still read the real runtime state.
ROCm callback selection is explicit in the header, no longer dependent on the
legacy binary converter recognizing the old filename. Domain-specific native
flags not listed as runtime-owned storage above remain in their owning modules;
the Python lifecycle partition does not claim to move all C++ storage.

`runtime/jit_policy` owns CUDA kernel math policy (`default`, `strict`, `backend`).
Ordinary and fused CUDA keys capture the policy before compilation, and the
compiler transforms startup flags according to that captured value. Switching
policy submits pending graphs under their previous policy. Explicit per-op
flags remain local overrides. Torch preflight selects this runtime policy,
rather than rewriting startup NVCC flags or creating a different core-build
configuration just to change kernel math. The ACL preflight still removes
legacy CUDA-specific strict flags from its startup environment.

The former exported `Executor exe` data symbol is removed. In-tree CUDA/ACL
consumers and embedded CUDA templates use `runtime_executor()` from `executor.h`.
Out-of-tree C++ extensions must update that access and rebuild against the new
headers/core library; old precompiled extensions are not binary compatible.
The former `tflag_count` symbol and `misc/traversal_epoch.h` path are also removed;
extensions using traversal internals must use the runtime header and rebuild.
The `use_cuda`, `device_id` and `sync_run` data symbols are also removed.
External native consumers must include `runtime/device.h` or
`runtime/device_state.h` and rebuild; Python `jt.flags` names remain unchanged.
Compiler, external-library setup, binding generation and CuPy bootstrap
implementations live under `build/`; historical root-module imports are
same-object aliases. `distributions/`, `init/`, and `linalg/` are public native
packages; `selftest.py` is the installed smoke-test entry point. New root files
require an ownership review and a corresponding structure-gate update.

The three domain initializers explicitly re-export their implementation objects.
Linear algebra separates complex routines, decompositions, solving, norms and
contractions, with shared array helpers and result types. Distributions separate
base contracts, constraints, helpers, discrete/continuous/relaxed/multivariate
families and KL divergence. Initialization separates basic filling, fan/gain
rules, scaled initializers and truncated normal; its facade retains the existing
Var method bindings. Function metadata names the physical owner, while historical
public pickle globals continue to resolve through the facades. All `.py` files
under `python/jittor` are now below 1,500 lines. The former large compatibility
NN, numerical, tensor and FlashAttention modules are normal packages with
separate implementation owners. This source decomposition does not establish
completion of the independent Torch architecture migration.
Runtime-only framework imports are deferred to calls to keep the import-cycle
surface from growing. The six legacy complex linalg functions are lazily
re-exported as their original objects, preserving concrete ComplexNumber type
annotations without making package bootstrap depend on the NN facade.

Contributed algorithms physically live under `contrib/{ccl,loss3d,math_util,
einops}`. Their historical packages and child-module paths are deprecated
same-object aliases, resolved on demand rather than by eagerly importing all
four domains. The `igamma.h` resource follows its owner into
`contrib/math_util/src`; source checkout and installed resource lookup use the
same module-relative path. The public `contrib` package also owns the historical
composition helpers; `jittor.compat.contrib` now aliases it. Alias publication
preserves same-named package functions such as `math_util.igamma` and
`ccl.ccl_2d` instead of replacing them with child-module objects. Einops parsing
and transformation share one `EinopsError` class in `einops._errors`, re-exported
by the public package without an implementation-to-facade import cycle.

Standalone build utilities physically live in `build/utils` while retaining the
`jittor_utils` runtime namespace, so utility imports can run before Jittor core
bootstrap. `jittor.build.utils` imports alias those same objects. This separates
physical ownership from the standalone bootstrap namespace without introducing
duplicate utility implementations or an eager inverse import of Jittor. Source
and dependency scanners inspect the real `build/utils` tree, not just the
standalone import bridge.

The compiler is split by responsibility: `build/compiler.py` retains startup
state and orchestration (1,472 lines at this migration), `build/codegen.py` owns
source generation (626 lines), and `build/compilation.py` owns compilation and
custom-extension operations (534 lines). Generator fingerprints include the
extracted owners so a change cannot silently reuse an obsolete build stamp.
Compatibility cleanup and independent Torch packaging remain separate work;
these native moves alone do not close the entire Python layout task.

Backend configuration is now a frozen `BuildConfig` returned by the selected
provider, with explicit services in `BuildContext`. Providers do not mutate
compiler globals or append to its source list; bootstrap publishes compatibility
attributes centrally. Entry points are loaded only for the selected backend,
and explicit CPU selection bypasses CUDA discovery. The build utilities receive
their binding/compiler services by injection and no longer import Jittor.
Tensor checkpoint algorithms live in `jittor.serialization`, with native
save/load and safe-pickle code in `serialization.native`; legacy utility
paths query runtime-injected loaders after bootstrap. See
[backend build configuration](backend-build-configuration.md) for the service
protocol, cache compatibility and pre-bootstrap/hardware limits.

### CUDA Resource Layout

The checkout's `backends/cuda/` is the physical CUDA owner. Library operators
live under `kernels/<library>`, library wrappers and headers under
`libraries/<library>/{src,include}`, and common support under `src` and `include`.
Native indexing/where/candidate/transpose implementations live under
`kernels/core`; diagnostic kernels live under `kernels/debug`. Python CUDA
implementations and source builders live under `kernels/{nn,misc,math,pooling,
sparse,ccl,loss3d}`. ACL KV implementations live under `backends/acl/kernels`.

Python imports use the canonical `jittor.backends` namespace in both checkouts
and wheels. A source-only path bridge and explicit package-directory mappings
avoid a second editable implementation tree under `python/`. Old NN module
spellings remain same-object aliases. `nn/` has no CUDA kernel modules or ACL
KV module; `nn/backends` contains only its initializer, cuDNN adapter and hook view.

Shared indexing and pooling mathematics remain single-source. The registration
generator composes backend indexing fragments with the shared source into an
atomically published, content-stable JIT source, preserving each segment's
`#line` mapping. CPU-only builds still compile the pure host loop-schedule helper.
The legacy `type/cuda_atomic.h` include forwards to the backend-owned header;
it no longer contains the CUDA implementation.

Resource lookup distinguishes source checkouts and installed packages.
The whole-tree source conversion mechanism has been removed.
A directory containing only `__pycache__`
cannot mask the real source owner. Packaging tests check every backend file in
the sdist, wheel and isolated installation.

The shared C++ core is under top-level `src` (packaged as `jittor/src`), and
MPI/NCCL/HCCL wrappers are under `backends/comm`. Neither `python/jittor/src`
nor `python/jittor/extern` remains in the checkout. The external FlashAttention
integration retains its separate compatibility-package migration. `fused_adamw` has no
CUDA algorithm to relocate; its existing ACL implementation and shared error
entry do not establish CUDA support.

### Native Support Layout

The native `BackendRegistry` in `runtime/backend*` is owned by `NativeRuntime`.
It publishes version-checked callback tables with owned names and stable
process-lifetime callbacks. Registering CPU and CUDA descriptors does not
initialize a driver; a CPU-only build still knows CUDA but reports zero devices.
`runtime/backends/` implements raw pool selection, device operations, copies,
synchronization and streams. Public allocator code retains the SFRL/NFEF/Temp/
Stat composition and obtains raw pools from the registry, never the reverse.

Array creation, host/device migration, device-copy operators, fetch and swap
transfers call this interface. `allocation_device()` derives the physical
device from the allocator; it does not mistake a Var's retained device affinity
for its current residency. Dual staging and delayed-free storage report their
actual pool device. Ordered peer copies preserve both source and destination
stream dependencies, and fetch retains blocks through its callback.
The old CUDA stream functions remain adapters into the registered stream hook;
shared implementation lives in `src/runtime/backend_streams.cc`, with SDK
operations in the corresponding backend driver.

`core.registered_backends()` and `core.backend_device_count(name)` query the
native registry. The four legacy accelerator-mode aliases in `jt.flags` emit
`DeprecationWarning` but retain their setter behavior. ACL publishes the canonical
name `acl`; the old `acl_legacy` spelling resolves to that same native descriptor
and Python kernel table without registering duplicate implementations.
Python selection consumes the native device
context; there is no separate Python allocator or hard-coded backend-capability
prototype.

### Native Operator Dispatch

`ops/op_register` publishes immutable `OpDef` objects with stable process-local
`OpId` values. An instantiated graph pins its definition; replacing a registry
entry affects new graphs, not live graphs. Each definition combines backend-keyed
`Kernel` callbacks with a `Codegen` interface for source fragments, preparation,
optimization and source metadata. Shape and gradient semantics remain on graph
operators. Replacement and unregister/re-register receive unique compilation
identities; ordinary and fused keys include the identities of their definitions
and fused children. Initial registrations retain stable disk-cache keys.
The executor, parallel compiler, tracer and relay use these registered
interfaces, including a dedicated fused implementation that retains its context
and relay cache. The old virtual execution methods are source adapters, not the
executor's dispatch path.

Generated core and extension registrations bind concrete operator implementations
with `register_op_definition<T>`. CUDA libraries register accelerator-only kernels;
MKL registers CPU kernels. Core CPU-only operators declare their backend mask.
`core.backend_supported_ops(name)` enumerates these registered implementations;
individual shape/dtype restrictions still apply. Missing implementations raise
instead of falling through to an empty virtual `run()`.

Optional libraries publish typed semantic capabilities in their own translation
units. Core replacement sites and matmul/conv tuners query those capabilities,
without naming CUB, cuRAND, cuTT, cuBLAS, cuDNN or MKL implementations. Capability
lookup resolves constructors at use time, so a lookup before a library loads does
not permanently cache a miss. `core.backend_supported_capabilities(name)` exposes
the available semantic families. Backend selection precedes source-fragment
generation, including dual-source CodeOp cache lookup.

Native extensions must rebuild: `Op` layout changed, `OpInfo` is now the compatibility
alias for `OpDef`, and source metadata lives under `definition.codegen`. Converted
ACL/ROCm/Corex backends retain their legacy build path. Host-only syntax checks are
not CANN ABI or device verification; those machines must build and execute the
changed backend before hardware support is claimed.

`jittor_utils.compile_module` compiles its generated wrapper and argument-printer
definitions as one translation unit. Two compiler inputs previously overwrote a
single depfile, omitting extension headers and silently reusing obsolete ABI
layouts. The changed command invalidates those old cache entries automatically;
subsequent header changes are tracked without renaming the extension or deleting
the cache. Already loaded extension modules still require a fresh process.

### Python Kernel Dispatch

`_runtime.dispatch` owns Python kernel registrations. Each entry declares its
operator, backend, accepted tensor dtypes, shape/gradient predicate and priority.
`select_kernel` returns the actual implementation; `try_dispatch` and the
`optional_kernel` adapter use the same selection. A miss permits an explicit
same-device generic implementation, not an implicit move to the CPU. Predicate
and implementation errors propagate without trying another kernel. Legacy
mode-specific restrictions are registration qualifiers rather than separate
backend guards at call sites.

`core.dispatch_context(inputs)` returns the runtime target and input-selected
device without materializing tensors or probing a driver. Python recursively
collects Vars in positional and keyword containers and records all their dtypes.
The native query checks mixed device inputs, preserving the bounded pending
scalar retargeting rule. Storage residency alone is not execution policy:
pending and host-staged inputs can still target the accelerator. Device-keyed
FFT and attention caches use this context, including non-default device ids.

CUDA/legacy library adapters, matrix/conv/RNN selection, normalization/inference,
indexing/scan and other native domains use this table. Non-ACL converted CUDA
implementations keep their explicitly declared ROCm/Corex registrations where
the old guards allowed them; this does not certify those devices. The old
`_runtime.registry` prototype and its bytearray allocator are removed. Root
flatten/clamp/outer now register portable implementations in the same table.

`_runtime.backend_libraries` owns loaded modules, their derived `.ops`, resources,
loader callbacks and availability policies. Missing queries are not permanently
cached; explicit loading propagates errors. MKL disablement is checked before
both cached-module lookup and loading, and reenabling can reuse the loaded
module. `compile_extern.*` and root library attributes remain dynamic read-only
queries, not mutable snapshots. Existing bootstrap ordering is retained; fully
lazy core import remains a separate task.

`nn.backends.hooks` is a read-only compatibility view into this table. ACL
providers publish implementations directly; the view never stores an independent
callback. Internal tests use `override_kernel` for scoped replacement or absence,
and restore the prior registration on exit. Direct legacy hook/library attribute
assignment is rejected. The ACL source converter remains a separate migration;
Python kernel publication no longer replaces the native public API.

### ACL Kernel Registration

The ACL build entry point is `jittor.backends.acl`. SDK support translation
units live in `backends/acl/src`, SDK-facing headers in `include/{aclops,aclnn}`,
and native operator translation units in `kernels/native`. The provider uses
an explicit 45-file build inventory (42 core, 3 registration), preserving the
previous ordering without accidentally globbing the independent backend and
workspace runtime sources. Source builders include `aclops/aclops.h` through
the backend include root; actual SDK `acl/acl.h` references are unchanged.
MPI, NCCL and HCCL resources live under `backends/comm/{mpi,nccl,hccl}` with
matching `inc`, `src` and `ops` directories. NCCL also owns its no-MPI header.
Compiler lookup uses `backend_root(..., "comm")` in both checkouts and wheels;
the removed `python/jittor/extern` path is not an include root or package input.

`backends/acl/kernels/install.py` publishes module-level tensor, neural-network
and normalization implementations in the existing Python dispatch table.
The paired SDK source builders live in `backends/acl/kernels/ops`; old Python
module paths are same-object aliases. Native functions and Module classes keep
their own identities, validation, parameter management and generic mathematics.
There is no `change_function` or `warp` installer. An unsupported Python variant
returns `None` to its same-device generic owner; execution errors propagate.

The native registry composes ACL `OpImplementation` values when definitions
are published, including extensions loaded after initialization. Installing
the composer also handles existing definitions transactionally. Definitions
remain immutable and existing graphs keep their pinned values. A stable startup
version preserves cross-process JIT keys; dynamic implementation replacement
still receives a unique identity. A late extension cannot acquire ACL support
merely by avoiding an initialization-time scan.

`Kernel.compile` replaces the global compilation hook. ACL registers distinct
fused, mapped, primitive and explicitly unsupported paths; fused relay selection
still follows tuning. SDK-native extensions such as HCCL explicitly declare
their compiler. Unsupported fallback entries do not appear as implemented
operators or capabilities, and other backends' constructors are not deleted.

`jt.code(..., backend="acl")` identifies the accelerator source as ACL SDK code.
It does not change the active device. Constructors validate the marker, cache
keys include it, ordinary and multi-output gradients inherit it, and a wrong
accelerator target is rejected before execution. `cpu_src` remains independent.
Third-party code that relied on `// aclop` or another incidental `acl` substring
must add the marker; comment-based recognition is deliberately removed.

Typed native Getitem/Setitem entries cover basic positive-step slices, integer
indices, new axes, ellipses, empty selections and broadcast assignment. A pure
checked address plan coalesces contiguous suffixes; device copies are queued on
the existing ACL computation stream. No tensor data is staged through the CPU.
Only exact shared mappings are no-ops; unsafe overlap, advanced/string indexing,
negative steps and native reduction assignment are explicitly unsupported by
this entry. Existing Python ACL builders retain their separate variants.
Scalar broadcast copies are conservative and are not a performance claim.
Native integer views retain the producer needed for chained writeback; basic
index gradients make assignment casts explicit without changing indexed-add
accumulation. Basic indexed assignment now follows native `VarView` records;
the former Python parent-chain writeback is removed. This does not imply that
every advanced indexing or cross-backend view variant has hardware coverage.

ACL post-processing only publishes its native implementations. Its pinned-host,
compiler-concurrency and reduction requirements belong to the backend descriptor
and are consumed by the allocator, compiler and reduction owners; public flags
are not overwritten. BackendOps ABI 3 rejects older descriptors and extensions
must rebuild. The legacy whole-tree SDK translation (`process_acl`,
`process_jittor_source`) is gone; real CANN/NPU verification is still required.

### Distributed Ownership

`jittor.distributed.process_group` owns `ProcessGroup`, `Work`, communicator
creation and live native rank/world queries. Launching and rendezvous are owned
by `distributed.launch` and `distributed.store`. The Torch installer delegates
to these objects, retaining only Torch spelling, argument adaptation and its
installation/bootstrap transaction. Historical `_JittorProcessGroup` and
`_JittorWork` imports remain aliases, so old pickle globals resolve to the same
canonical classes. Invalid native world-size metadata now raises instead of
being silently treated as a single-process runtime.

### Backend Fallback Policy

`NativeRuntime` owns `backend_fallback`, exposed through both `jt.flags` and
`jt.runtime`. The default is `warn`; `error` rejects an automatic cross-backend
computation, and `allow` permits it without warning. Invalid assignments leave
the previous policy intact. The executor checks CPU-only execution before
migrating inputs; array staging, fetch and explicit device transfers are not
computational fallback. A generic kernel on the requested device is not a
cross-backend fallback either.

The legacy ACL executor preflights the complete fused group or standalone
operation before execution. Only an explicitly unsupported operation/variant
can request CPU fallback. SDK, shape and kernel execution failures clean up and
propagate their original exception; they are not routing signals. A permitted
fallback restores the prior execution mode, operator flags and fused context
even if CPU execution fails. Family-internal SDK resource cleanup still has
separately tracked work; host-only tests do not establish NPU hardware support.

`core.backend_fallback_count()` counts cross-backend decisions, including denied
attempts, not completed CPU computations. Hardware gates use `error`.
`_runtime.fallback.forbid_backend_fallbacks()` also checks a count delta after a
normal return, detecting attempts whose exceptions were swallowed by a caller.
It preserves a primary exception and does not synchronize implicitly: the caller
must execute and synchronize the work inside the scope. NPU pytest fixtures and
standalone ecosystem runners use this interface instead of parsing log wording.

The C++ `src/misc/` directory no longer exists. Support code is grouped by its
actual role; this is a source-layout change, not a change to helper algorithms
or a claim that the backend registry migration is complete.

| Owner | Support Code |
| --- | --- |
| `src/debug/` | CPU/CUDA NaN checking and diagnostics |
| `src/runtime/` | Device streams, float32 precision policy, traversal indexing, RingBuffer, collective dtype and rendezvous helpers |
| `src/type/` | Nano types and scalar math, atomic, intrinsic and numeric-limit helpers used by generated kernels |
| `src/utils/` | Generic strings, hashes, containers, shared pointers and cleanup helpers |
| `src/third_party/` | Vendored miniz |

Both generated includes and backend source transformations use these paths.
Source extensions that included `misc/...` must update their includes before
rebuilding. Basenames are unchanged so existing ROCm/Corex conversion rules
retain their dispatch identity. Moving these support files does not complete
the separate `init`/profiler/lock or Python-binding layout migrations.

On a reused transformed-source cache, native files absent from the original
source tree are moved out of `src/` and `extern/` before compilation. They are
preserved under `<backend>_source_stale_*` in the cache directory, not deleted.
This prevents old and new translation units from being compiled together after
a source move. Non-native cache artifacts are left alone.

### Compatibility APIs

Independent activation constructs its TorchNamespace before installation.
Installers publish APIs and context on that target; a transactional backend-to-owner
binding routes leaf, retained-gradient and optimizer bookkeeping to one state.
Legacy native attributes alias that state, and rolled-back installation steps
are replayed against the same state on retry.

The optional TorchNamespace owns its public writes and deletions. Missing reads
may still use its native owner; deletion masks that fallback locally. Transaction
rollback restores the exact local binding and deletion state. InstallContext
separates the installation target from its native backend and never inherits
install markers through namespace fallback. Publication keeps the root self-alias
consistent in the registry and import mapping. Independent activation owns real
Tensor/Parameter subclasses and Module/NN adapters, reusing the native Var/Op
graph and mathematics without installing those APIs on native classes.

The canonical Torch-style implementation is `jittor.compat.torch`. The legacy
attribute/module spelling `jittor.torch_compat` loads its optional alias provider
on explicit import; it is not a second source file. Likewise, the canonical Triton
implementation is `jittor.compat.triton`, with `jittor.triton_shim` retained as
an object-identity alias.

Compatibility installers for NN, numerical and tensor APIs, and the
FlashAttention adapter, are normal packages split by implementation family.
Public callables without installation-state captures can retain module-level
identity; stateful installation paths keep explicit context and their original
registration order. Task 7.12 remains open: the remaining per-Tensor and runtime
state must be consolidated and native Torch-role dependencies removed. The
physical `compat/` tree is now the independent `jittor-torch` distribution; core
packaging excludes it. The explicit legacy mode still
adapts native classes and must not be confused with independent activation.

Basic indexing uses native `VarView` tracking instead of a parallel Python
`_torch_index_parent`/slice chain. Torch-specific slice forms that do not yet
have a native view record explicitly attach one with `_set_view_of`. The
detached `.data` API retains its separate owner/path bookkeeping; assignment
uses a detached right-hand-side node so stopping the data alias does not freeze
its trainable owner. The write-only strong-reference table for `requires_grad`
has been removed. Leaf registration, retained-gradient tracking and optimizer
registration remain because they have actual consumers; they are not replaced
by the view migration.

`jittor.compat.shim` owns the runtime and deployment code for the optional
top-level `torch` surface used by applications that import Torch directly. The
name `jittor.torch_shim` is retained only as a same-object legacy alias. The shim
delegates Torch-style semantics to `jittor.compat.torch`; neither the alias nor
the deployed package owns a second implementation.

Plain Jittor startup uses `_runtime.import_aliases` for native aliases and
`_runtime.compat_bootstrap` for optional activation. It does not import any
`jittor.compat` module; a missing optional package produces an installation
error only when compatibility is requested. The
Torch installer runs after an explicit Torch-mode preflight, through a deployed
`torch` entry point, or when the historical `jittor.torch_compat` alias is
imported. This prevents class-level Torch adaptations from changing native
Jittor APIs in unrelated processes.

`jittor.compat.vllm` is a staged exception to the normal rule that
project/version glue lives in an optional integration distribution. It may use
only public Jittor APIs plus the public module-patcher mechanism, must remain
relocatable, and activates only when vLLM is imported. Its exit condition is a
versioned, installable vLLM plugin that preserves the maintained structure,
correctness, and performance gates. The device platform and worker adapter stay
outside the core repository while this extraction is incomplete.

The ownership order is:

1. native Jittor semantics and broadly useful operations;
2. reusable mechanisms in `jittor.compat`;
3. optional import/deployment shims;
4. project-specific integrations outside the core distribution.

See [Torch compatibility principles](torch-compatibility-principles.md) for the
behavioral decision rules.

## Import and initialization rules

- Module imports must not compile kernels, download assets, mutate the source
  checkout, or silently install external packages.
- Registration is idempotent. Re-importing a compatibility module must not wrap
  the same callable twice or create a second module object.
- Optional dependency checks happen at the operation boundary unless import-time
  discovery is itself the API.
- Broad exception handlers may annotate and re-raise a failure; they must not
  convert a partially installed compatibility surface into apparent success.
- Expensive imports stay out of collection-only structure tests.
- A top-level definition may not be silently replaced by a later definition in
  the same file. Cross-file identical implementations are scanned as well;
  retained duplicates require a narrow reviewed category such as standalone
  deployment entry points, backend code-generation templates, model-local
  architecture blocks, or legacy serialization readers.

## Runtime resources

The following trees are consumed by compiler or packaging code using physical
paths and therefore require special review:

- `src/` (installed as `jittor/src/`)
- `backends/acl/{include,kernels/native,src}/`
- `backends/comm/`
- `python/jittor/contrib/math_util/src/`
- `compat/shim/cpp_extension/` (optional `jittor-torch` distribution)

A move is complete only when source checkouts, sdists, wheels, cold JIT builds,
and installed smoke tests all agree. Directory aesthetics alone are not a reason
to relocate these resources.

## Refactor protocol

For each module move:

1. Inventory definitions, assignments, imports, registrations, and consumers.
2. Define the canonical destination and any compatibility alias explicitly.
3. Move a coherent domain slice without unrelated behavior changes.
4. Compare the moved definition set and public exports mechanically where
   possible.
5. Test import identity, public calls, dynamic dispatch, serialization where
   applicable, and the relevant CPU/accelerator behavior.
6. Delete the transitional source path and add it to the structure gate.
7. Update durable documentation and active links in the same change.

Do not preserve two editable implementations after a move. Compatibility must
delegate to the canonical object and have an exit condition.

## Acceptance

A source-layout change is acceptable when:

- imports and public names retain their documented behavior;
- no legacy implementation tree or root-level compatibility file remains;
- the wheel contains every required runtime resource and excludes repository-only
  tests/tools;
- `bash agent/scripts/check_repo_layout.sh` passes;
- focused tests, structure tests, and every affected backend gate pass;
- any deliberate incompatibility is documented in release notes.
