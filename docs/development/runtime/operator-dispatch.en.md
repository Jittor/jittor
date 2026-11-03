# Operator registration, dispatch and fallback

[Section contents](../index.en.md) · [中文](operator-dispatch.md)

Read this page when changing operator registration, kernel selection or backend fallback. Start with [source architecture](../source-architecture.en.md) for module ownership.

These are implementation contracts, not a new hardware acceptance report. The source document's review baseline was `e3c369acb`; this split preserves that evidence boundary.

## CUDA resource layout

The checkout's `backends/cuda/` is the physical owner of CUDA. Library operators are in
`kernels/<library>`, library wrappers and headers in `libraries/<library>/{src,include}`, and
common support in `src` and `include`. The native index/where/candidate/transpose
implementations are in `kernels/core`, and diagnostic kernels in `kernels/debug`. Python-side
CUDA implementations and source constructors are in
`kernels/{nn,misc,math,pooling,sparse,ccl,loss3d}`. The ACL KV implementation is in
`backends/acl/kernels`.

Python imports use the canonical `jittor.backends` namespace in both the checkout and the wheel.
A source-only path bridge plus an explicit package-directory mapping avoid a second editable
implementation tree under `python/`. Old NN module spellings remain same-object aliases. **There
is no CUDA kernel module and no ACL KV module in `nn/`**; `nn/backends` contains only its
`__init__` file, the cuDNN adapter and the hook views.

Shared indexing and pooling mathematics keep a single source. The registration generator
combines backend index fragments with shared source into one **atomically published,
content-stable** JIT source, preserving each segment's `#line` mapping. Pure CPU builds still
compile that host-only loop-scheduling helper. The historical `type/cuda_atomic.h` include
forwards to the backend-owned header, which no longer contains a CUDA implementation.

Resource lookup distinguishes the source checkout from the installed package. **The whole-tree
source transformation mechanism has been removed.** A directory containing only `__pycache__`
cannot shadow the real source owner. Packaging tests check **every** backend file in the sdist,
the wheel and an isolated installation.

The shared C++ core is in top-level `src` (packaged as `jittor/src`), and the MPI/NCCL/HCCL
wrappers in `backends/comm`. `python/jittor/src` and `python/jittor/extern` **no longer exist in
the checkout**. The external FlashAttention integration keeps its separate compatibility-package
migration. `fused_adamw` has no movable CUDA algorithm -- its existing ACL implementation and
shared error entry point **do not constitute CUDA support**.

## Native operator dispatch

`ops/op_register` publishes immutable `OpDef` objects with a process-stable `OpId`. **An
instantiated graph pins its definition**; replacing a registry entry affects new graphs, not live
ones. Each definition combines a per-backend-indexed `Kernel` callback with a `Codegen` interface
(source fragment, prepare, optimize, source metadata). Shape and gradient semantics remain on
the graph operator. Replacement, as well as unregister/re-register, receives a unique
compilation identity; ordinary and fused keys include its identity and those of the fused
sub-items. The first registration keeps a stable on-disk cache key. The executor, parallel
compiler, tracer and relay use these registered interfaces, including a dedicated fusion
implementation that keeps its context and relay cache. **The old virtual execution method is only
a source adapter, not the executor's dispatch path.**

Generated core and extension registrations bind concrete operator implementations with
`register_op_definition<T>`. CUDA library registration covers accelerator-only kernels, MKL
registers CPU kernels, and core CPU-only operators declare their backend mask.
`core.backend_supported_ops(name)` enumerates these registered implementations; **individual
shape/dtype limits still apply**. A missing implementation **throws**, instead of falling into
an empty virtual `run()`.

Optional libraries publish typed semantic capabilities in their own translation units. The
core's replacement points and the matmul/conv tuner query these capabilities without **naming**
CUB, cuRAND, cuTT, cuBLAS, cuDNN or MKL implementations. A capability lookup resolves the
constructor at use time, so **one lookup before a library is loaded does not permanently cache
the "miss"**. `core.backend_supported_capabilities(name)` exposes the available semantic
families. Backend selection precedes source-fragment generation, including the cache lookup for
two-source CodeOps.

**Native extensions must rebuild**: the `Op` layout changed, `OpInfo` is now a compatibility alias
of `OpDef`, and source metadata lives under `definition.codegen`. The converted ACL/ROCm/Corex
backends keep their historical build paths. **A host-only syntax check is not CANN ABI or device
validation**; those machines must build and execute the changed backend before claiming hardware
support.

`jittor_utils.compile_module` compiles the generated wrapper and the argument printer definitions
as **one translation unit**. Previously the two compiler inputs overwrote the same depfile, missed
the extension headers and silently reused a stale ABI layout. The changed command automatically
invalidates old cache entries; subsequent header changes are then tracked correctly, without
renaming the extension or deleting the cache. **An already loaded extension module still needs a
new process.**

## Python kernel dispatch

`_runtime.dispatch` owns Python kernel registration. Each entry declares its operator, backend,
accepted tensor dtypes, shape/gradient predicates and priority. `select_kernel` returns the real
implementation; the `try_dispatch` and `optional_kernel` adapters use the same selection logic.
**A miss permits only one explicitly declared same-device generic implementation, never an
implicit move to CPU.** Errors from predicates and implementations propagate upwards, **without
trying another kernel**. Historical mode-specific limits become registration qualifiers rather
than separate backend guards at the call site.

`core.dispatch_context(inputs)` returns the runtime target and the device determined by the
inputs, **without materializing tensors or probing drivers**. Python recursively collects Vars
from positional and keyword containers and records all their dtypes. The native query checks for
mixed-device inputs and preserves the bounded pending-scalar redirection rule. **Storage
residency itself is not an execution policy**: pending and host-staged inputs can still target an
accelerator. The per-device-indexed FFT and attention caches use this context, including
non-default device ids.

CUDA/historical library adapters, matrix/convolution/RNN selection, normalization/inference,
and native domains such as indexing/scan all use this table. Converted CUDA implementations that
are not ACL keep their explicitly declared ROCm/Corex registrations where the old guards allowed
them -- **this does not constitute certification for those devices**. The old `_runtime.registry`
prototype and its bytearray allocator are deleted. Root-level flatten/clamp/outer now register
portable implementations in the same table.

`_runtime.backend_libraries` owns loaded modules, the `.ops` derived from them, resources, loader
callbacks and availability policy. **A missed query is not permanently cached**; an explicit load
propagates errors. MKL disabling is checked **before** the cached module lookup and load, and
re-enabling can reuse an already loaded module. `compile_extern.*` and the root-level library
attributes remain dynamic read-only queries, not mutable snapshots. The existing bootstrap order
is unchanged; fully lazy core imports are a separate task.

`nn.backends.hooks` is a read-only compatibility view of this table. ACL providers publish
implementations directly, and that view **never stores an independent callback**. Internal tests
use `override_kernel` for scoped replacement or absence, restoring the previous registration on
exit. **Direct assignment to the historical hook/library attributes is rejected.** The ACL source
transformer remains a separate migration; Python kernel publication no longer replaces the native
public API.

## ACL kernel registration

The ACL build entry point is `jittor.backends.acl`. The SDK support translation units are in
`backends/acl/src`, the SDK-facing headers in `include/{aclops,aclnn}`, and the native operator
translation units in `kernels/native`. The provider uses an explicit 45-file build list (42 core
+ 3 registration), preserving the original order, **so a wildcard cannot accidentally pull the
standalone backend and workspace runtime sources in**. Source constructors include
`aclops/aclops.h` through the backend include root; references to the real SDK `acl/acl.h` stay
unchanged.

MPI, NCCL and HCCL resources are under `backends/comm/{mpi,nccl,hccl}`, each with corresponding
`inc`, `src` and `ops` directories; NCCL additionally owns its no-MPI headers. Compiler lookup
uses `backend_root(..., "comm")` in both the checkout and the wheel; the deleted
`python/jittor/extern` path **is neither an include root nor a package input**.

`backends/acl/kernels/install.py` publishes module-level tensor, neural-network and
normalization implementations into the existing Python dispatch table. The companion SDK source
constructors are in `backends/acl/kernels/ops`; the old Python module paths are same-object
aliases. Native functions and Module classes keep their own identity, validation, argument
management and common mathematics. **There is no `change_function` or `warp`-style installer.**
An unsupported Python variant returns `None` to its same-device generic owner; execution errors
propagate upwards.

The native registry composes ACL's `OpImplementation` values when a definition is published,
**including extensions loaded after initialization**. Installing a compositor also handles
existing definitions **transactionally**. Definitions stay immutable and existing graphs keep
their pinned values. A stable startup version keeps JIT keys stable across processes; dynamic
implementation replacement still gets a unique identity. **A late extension cannot gain ACL
support merely by avoiding the initialization-time scan.**

`Kernel.compile` replaces the global compile hook. ACL registers fused, mapping, primitive and
explicitly unsupported paths separately; fused-relay selection still follows tuning. SDK-native
extensions such as HCCL declare their compiler explicitly. **An unsupported fallback entry does
not present itself as an implemented operator or capability**, and other backends' constructors
are not deleted.

`jt.code(..., backend="acl")` marks accelerator source as ACL SDK code. **It does not change the
current device.** Constructors validate the marker, the cache key includes it, ordinary and
multi-output gradients inherit it, and a wrong accelerator target is rejected before execution.
`cpu_src` stays separate. Third-party code that relied on `// aclop` or any other incidental
`acl` substring **must add the marker** -- comment-based recognition was deliberately removed.

Typed native Getitem/Setitem entries cover basic positive-stride slicing, integer indexing, new
axes, ellipsis, empty selection and broadcast assignment. A pure, checked address plan merges
contiguous suffixes; device copies are ordered on the existing ACL compute stream. **No tensor
data is staged through the CPU.** Only an exactly identical shared mapping is a no-op; unsafe
overlaps, advanced/string indexing, negative strides and native reduction assignment are
**explicitly declared unsupported by this entry point**. The existing Python ACL constructors
keep their separate variants. Scalar broadcast copies are conservative and **make no performance
claim**. Native integer views preserve the producer needed for chained write-back; the gradient
of basic indexing makes the assignment conversion explicit without changing indexed-add
accumulation. Basic indexed assignment now follows the native `VarView` record, the previous
Python-side parent-chain write-back having been removed. **This does not mean every advanced
indexing or cross-backend view variant has hardware coverage.**

ACL post-processing publishes only its native implementations. Its requirements for pinned host
memory, compiler concurrency and reduction are **backend descriptor** properties, consumed by
the allocator, compiler and reduction owners; **public flags are not overridden**. BackendOps ABI
3 rejects older descriptors, and extensions must rebuild. The historical whole-tree SDK
transformation (`process_acl`, `process_jittor_source`) is gone; **real CANN/NPU validation is
still required**.

## Backend fallback policy

`NativeRuntime` owns `backend_fallback`, exposed through `jt.flags` and `jt.runtime`. The default
is `warn`; `error` rejects automatic cross-backend computation, and `allow` permits it without a
warning. **An illegal assignment keeps the previous policy unchanged.** The executor checks for
CPU-only execution before migrating inputs; array staging, fetch and explicit device transfers
are **not computational fallbacks**, and a generic kernel on the requested device is **not** a
cross-backend fallback either.

The historical ACL executor preflights the full fused group or the standalone operation before
execution. **Only an explicitly unsupported operation/variant may request a CPU fallback.** SDK,
shape and kernel execution failures clean up and propagate their original exception -- **they are
not routing signals**. A permitted fallback restores the previous execution mode, operator flags
and fusion context even if the CPU execution fails. Separate work remains to track SDK resource
cleanup within families; **host-only tests do not constitute NPU hardware support**.

`core.backend_fallback_count()` counts cross-backend **decisions** (including rejected attempts),
not completed CPU computation. Hardware gates use `error`.
`_runtime.fallback.forbid_backend_fallbacks()` also checks the counter delta after a normal
return, thereby catching attempts whose exceptions the caller swallowed. It preserves the
primary exception and **does not synchronize implicitly**: the caller must execute and
synchronize the work inside the scope. NPU pytest fixtures and standalone ecosystem runners use
this interface **rather than parsing log wording**.

The C++ `src/misc/` directory no longer exists; support code is grouped by its actual role. **This
is a source-layout change, not a change to helper algorithms, and it does not mean the backend
registry migration is complete.**

| Owner | Support code |
| --- | --- |
| `src/debug/` | CPU/CUDA NaN checking and diagnostics |
| `src/runtime/` | device streams, float32 precision policy, traversal indices, RingBuffer, collective dtype and rendezvous helpers |
| `src/type/` | Nano types and scalar math, atomics, intrinsics and the numerical-limit helpers used by generated kernels |
| `src/utils/` | generic string, hash, container, shared-pointer and cleanup helpers |
| `src/third_party/` | the bundled miniz |

Generated includes and backend source transformation both use these paths. **Source extensions
that used to include `misc/...` must update the include before rebuilding.** Basenames stay
unchanged, so existing ROCm/Corex transformation rules keep their dispatch identity. Moving these
support files **is not** the completion of the separate layout migrations of `init`/profiler/lock
or of the Python bindings.

When reusing a converted source cache, native files that do not exist in the original source tree
are moved out of `src/` and `extern/` before compilation and **preserved** under
`<backend>_source_stale_*` in the cache directory instead of being deleted. This prevents old and
new translation units from being compiled together after a source move. Non-native cache
artifacts are unaffected.
