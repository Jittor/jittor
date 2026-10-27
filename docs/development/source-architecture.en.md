# Source architecture and module boundaries

- Status: accepted
- Last reviewed: 2026-10-05 (folded in the cleanup-period contracts for backend builds, dynamic
  shapes, dtype and frontend runtime state)
- Baseline: `e3c369acb`
- Owner: Jittor core maintainers
- Recheck when: a public module moves, a new implementation domain is added, or a runtime
  resource path changes

This document defines how Jittor's internal Python source is decomposed. Ownership of the
repository, packaging and runtime resources is defined by the higher-level
[repository layout decision](repository-layout.en.md).

## Principles

1. **One physical owner.** A public domain is an ordinary package; it **no longer has a second
   private tree hiding the real implementation**.
2. **Imports follow ownership.** Implementation metadata and tracebacks report **the module
   that owns that code**. Re-export modules do not recursively rewrite `__module__`.
3. **Composition stays shallow.** A package's `__init__.py` composes and exports public names;
   large implementations live in meaningful submodules.
4. **Runtime paths are contracts.** C++/CUDA resources loaded by the compiler move only
   together with an explicit compiler and packaging migration.
5. **Compatibility is layered.** Framework-native capabilities, reusable compatibility
   mechanisms, import shims and downstream integrations each have a different owner.
6. **Every move preserves behaviour.** A refactor preserves public names, promised callable
   identity, pickle behaviour within the supported range, backend dispatch, and targeted
   regression coverage.

## Current domain partitioning

```text
python/
├── jittor/
│   ├── __init__.py              # 根组合与运行时初始化
│   ├── __init__.pyi             # 公开根的类型标注面
│   ├── _core/                   # 原生 Python API 的实现域
│   │   ├── api.py               # 编译核心引导之后的显式组合
│   │   ├── var.py               # 张量构造、运算与 Var 绑定
│   │   ├── module.py            # Module 与参数/缓冲归属
│   │   ├── function.py          # 自定义自动微分上下文与梯度钩子
│   │   ├── hooks.py             # 可移除句柄与钩子支持
│   │   ├── flags.py             # 原生作用域与共享运行时状态
│   │   ├── arg_policy.py        # 显式的不支持/忽略参数策略
│   │   └── diagnostics.py       # 日志、性能分析、进程作用域与退出清理
│   ├── _runtime/
│   │   ├── core_api.py          # _core.api 的同对象历史别名
│   │   ├── composition.py       # 显式的命名空间发布
│   │   ├── install_order.py     # 有序的安装器注册与校验
│   │   └── state.py             # 注入的原生 Flags 视图，不做引导期导入
│   ├── serialization/
│   │   └── native.py            # 原生 save/load 与安全 pickle 实现
│   ├── build/                   # 编译器/引导的实现归属
│   │   ├── compiler.py          # 启动状态与构建编排
│   │   ├── codegen.py           # 原生绑定/注册的源码生成
│   │   ├── compilation.py       # 编译与自定义扩展操作
│   │   └── utils/               # 独立工具，以 jittor_utils 名义导入
│   ├── contrib/                 # 贡献算法与组合助手
│   │   ├── ccl/                 # 连通域标记
│   │   ├── loss3d/              # Chamfer 与推土机损失
│   │   ├── math_util/           # gamma 函数与共享原生资源
│   │   └── einops/              # 内置的张量表达式算法
│   ├── nn/                      # 神经网络公开 API
│   │   ├── modules/             # 有状态的 Module 实现
│   │   ├── functional/          # 无状态的张量函数
│   │   ├── backends/            # cuDNN 与只读的钩子调用适配
│   │   ├── utils/               # weight norm 等构造助手
│   │   └── attention.py
│   ├── autograd/                # 函数式自动微分
│   ├── fft/                     # 可导的原生 FFT 命名空间
│   ├── linalg/                  # 分解、求解、范数与缩并
│   ├── distributions/           # 概率分布族与共享约束
│   ├── init/                    # 初始化族与共享的 fan/gain 规则
│   ├── ops/                     # 张量、索引、归约与形状实现
│   ├── misc/                    # 已弃用的同对象张量 API 门面
│   ├── pool/                    # 已弃用的同对象池化 API 门面
│   ├── optim/                   # 优化器门面与算法模块
│   ├── sparse/                  # COO 稀疏张量与稀疏卷积
│   ├── compat/
│   │   ├── torch/               # 规范的 Torch 风格 API 兼容
│   │   ├── fsdp2/               # 分布式 FSDP2 兼容
│   │   ├── triton/              # Triton API 桥接与部署命令
│   │   ├── shim/                # Torch shim 运行时与部署命令
│   │   ├── module_patcher.py
│   │   └── external_backend.py
│   ├── selftest.py              # 安装后的冒烟测试
│   ├── tools/                   # 用户工具，含 benchmarking.py
│   ├── backends/                # 源码 checkout 的包路径桥接
│   └── distributed/             # 原生启动、rendezvous 与通信助手
└── jittor_utils/                # 通向 jittor/build/utils 的独立导入桥
```

## Package composition contract

### Neural network API

`jittor.nn` is the public package. Stateful layers live in `nn.modules`, stateless operations
in `nn.functional`, and optional accelerated paths in `jittor.backends.cuda.kernels`.
`nn.backends` holds **call adapters, not CUDA implementations**. Public re-exports must point
at the canonical implementation object; that holds only when the wrapper really enforces a real
API contract.

The dependency direction is:

```text
nn.modules -> nn.functional -> Jittor 张量/核心运算
nn.backends ----------------> 显式的后端/编译器接口
```

**A functional module must not import a stateful layer implementation.** Backend adapters must
stay optional and fail with an **actionable capability error** when their toolchain is
unavailable.

### Miscellaneous and pooling APIs

`jittor.ops` owns tensor, shape, indexing and composition operations. The historical
`jittor.misc` import remains a **deprecated same-object facade**, including the old submodule
paths. Public names and pickle global symbols resolve to the canonical implementation; **there
is no second editable math implementation tree under `misc`.**

Pooling mathematics and argument validation live in the ordinary `nn.functional.pooling`
package, split into separate owners for average pooling, 2D/3D cores, adaptive, 1D and
unpooling. `nn.modules.pooling` only keeps constructor arguments and calls these stateless
implementations; **a functional call does not construct a temporary Module**. CPU/CUDA generated
source keeps a single source of truth, and backend emission constructors stop at the existing
registration boundary.

`jittor.pool` and its historical submodules are **deprecated re-exports**.
`nn.modules.pooling_legacy` keeps three historical classes under their original class names:
`AvgPool2d` and `AvgPool3d` keep their forwarding-layer state for old pickles, while
`AdaptiveAvgPool2d` keeps its fixed-window algorithm -- **the latter deliberately differs from
the current NN's overlapping-bin algorithm; a layout move does not silently change its
numerical rule.** The adaptive-window intermediates are function locals, not persistent Module
state.

The sole owner of the historical switch `pool_use_code_op` is
`nn.functional.pooling._state`. `jt.pool`, `jt.nn` and the functional pooling package expose
live reads and writes of **the same value**, including temporary attribute override and
restoration. **Backend execution never reads a copied facade value.** NN constructs its
functional and module owners before loading the legacy facade, so this compatibility import
does not create a bootstrap cycle.

`jittor.sparse` owns coordinate-format sparse tensors and sparse neural-network kernels in
separate submodules. The historical name `jittor.nn.sparse` is a same-object alias of
`jittor.sparse.convolution`; `jittor.nn` and `jittor.nn.functional` re-export those canonical
callables.

`jittor.autograd` owns functional automatic differentiation. `jittor.fft` owns the
differentiable FFT/shift/frequency namespace shared by native Jittor and Torch mode.
Concatenation and indexing are in `jittor.ops`, pooling in `jittor.nn`, the optimized softmax in
`jittor.backends.cuda.kernels.nn`, and weight normalization in `jittor.nn.utils`.
**Historical root-level spellings are import aliases only**, with no physical source file or
wrapper implementation kept.

### Root module ownership

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

The previously exported `Executor exe` data symbol is deleted. In-tree CUDA/ACL consumers and
the embedded CUDA template use `runtime_executor()` from `executor.h`. **Out-of-tree C++
extensions must update that access path and rebuild against the new header/core library; old
prebuilt extension binaries are incompatible.** The `tflag_count` symbol and the
`misc/traversal_epoch.h` path are likewise deleted; extensions using traversal internals must
switch to the runtime headers and rebuild. The `use_cuda`, `device_id` and `sync_run` data
symbols are deleted as well -- external native consumers must include `runtime/device.h` or
`runtime/device_state.h` and rebuild; **the Python `jt.flags` names are unchanged**.

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

### CUDA resource layout

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

### Native support-layer layout

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

### Native operator dispatch

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

### Python kernel dispatch

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

### ACL kernel registration

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

### Distributed ownership

`jittor.distributed.process_group` owns `ProcessGroup`, `Work`, communicator creation and live
native rank/world queries. Launch and rendezvous are owned by `distributed.launch` and
`distributed.store`. The Torch installers delegate to these objects, keeping only Torch
spellings, argument adaptation and their own install/bootstrap transactions. The historical
`_JittorProcessGroup` and `_JittorWork` imports remain aliases, so old pickle global symbols
resolve to the same canonical classes. **Illegal native world-size metadata now raises**, rather
than silently being treated as a single-process runtime.

### Backend fallback policy

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

### Compatibility API

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
`jittor-torch` distribution, and **core packaging excludes it**. The explicit historical mode
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

The vLLM implementation and its dedicated tests live in the main repository's
`adapters/jittor_adapters/vllm` and `adapters/tests/vllm`, are distributed by the existing
`jittor-torch-adapters` distribution, and **are not distributed by Jittor or jittor-torch**. The
optional Torch stage selects the `jittor.module_patches` entry point named `jittor_vllm`, pointing
at `jittor_adapters.vllm:register`. Its `register(callback) -> None` registrar assembles
pre-import extension settings and registers post-import layer patches through the shared
transaction mechanism. **When not installed it reports `unavailable` without breaking Torch.**
The public Jittor primitives remain the mathematical implementation itself. This extraction
**does not include** the separately maintained Ascend platform and worker sources, **nor does it
claim complete NPU serving or hardware validation**.

The ownership order is:

1. native Jittor semantics and generally useful operations;
2. reusable mechanisms in `jittor.compat`;
3. the optional import/deployment shim;
4. project-specific integrations outside the core distribution.

For the behavioural decision rules see
[Torch compatibility principles](../compatibility/principles.en.md). For which process state the
install and delayed hooks write, and which of it can be rolled back, see
[Torch install transactions](torch-install-transactions.en.md).

### Runtime state and API ownership of the Torch frontend

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

### dtype boundary

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

## Backend build configuration

### Configuration values and provider hooks

`jittor_utils.build_config.BuildConfig` (implemented in
`python/jittor/build/utils/build_config.py`) is a **frozen value**: compiler paths and flags,
cache/source roots, backend capabilities, extra core sources, environment overrides and backend
resources. Source lists are tuples; resource and environment mappings copy their inputs and are
read-only; loaded modules and driver handles count as resources, not copies of compiler state.
`evolve(...)` returns a new configuration without modifying its input.

A backend provider implements `configure(context) -> BuildConfig`. `BuildContext` provides
explicit compilation, dynamic-loading and library-publication services and **no source-rewriting
service**: a backend that needs different code registers its own implementation and contributes
its own translation units, and no provider receives the shared source tree to rewrite.
`config.jittor_path` points at the checkout being compiled, is set once by bootstrap, and is
never redirected to a derived copy afterwards. A provider must not import `jittor.compiler`,
assign to compiler attributes, append to its source list, or modify the process environment;
bootstrap applies the returned environment changes and publishes the compatibility compiler
attributes once. `compiler.build_config` is the finally published value, and
`compiler.make_backend_context()` hands the same configuration to `install_extern(context)` and
`post_process(context)`.

`jittor_utils.backend_discovery` describes the selected SDK provider with a Protocol (Python 3.7
takes `Protocol` from the conditional dependency `typing_extensions`):

| Entry | Input | Result and ownership |
| --- | --- | --- |
| `configure` | an immutable `BuildContext` | a new `BuildConfig`; modifies neither compiler globals nor the input configuration |
| `install_extern` | a context with the library-publication service | boolean: whether this provider handled external-library installation |
| optional `post_process` | a context after core construction | binds runtime operators; the coordinator ignores the return value |

Optional hooks are modelled separately as `BackendPostProcess`. Discovery imports only the
selected entry point and validates all required hooks and every supplied optional hook before
`configure` compiles anything; a missing hook or a wrong signature raises `TypeError` naming the
provider and method. A missing SDK and runtime failures remain explicit failures of the selected
provider, **not an implicit CPU fallback**. ACL puts extra source files in its return value and
keeps the compiled registration initializer and the loaded runtime library as resources; Corex
removes OpenMP from the common and kernel flags before publication; ROCm declares its HIP runtime
translation unit as backend source, the SDK root as a resource, and publishes libraries through
an injected registration callback. None of the three rewrites shared source any more.

The Python build protocol **does not replace** the versioned `BackendOps` table in
`src/runtime/backend.h`. That table owns device, memory, stream and event callbacks and execution
capabilities; the native registry validates the ABI and the required callbacks and copies the
table, code and allocator pools must outlive its Runtime, and stream/event handles belong to the
device they declare. The current version is **ABI 3**: the version and struct size are checked
before copying a descriptor or reading its policy tail, and extensions built against an older
version must rebuild. ACL's requirements for pinned host memory, compiler concurrency and
reduction are written in this descriptor (see "ACL kernel registration" above) and are selected
according to the actual accelerator target: an ACL build running a CPU scope uses the ordinary
CPU policy. Allocation is the authority on residency; frontend placement is another graph
constraint, see [device and placement](../notes/device-placement.en.md).

### Discovery and selection

- Providers register in the entry-point group of the `jittor.backends` package (`acl`, `rocm`,
  `corex`). Installed metadata is used when available; a source checkout has matching lazy
  fallback entries for these three built-in providers.
- `JT_BACKEND=cpu|cuda|acl|rocm|corex` selects a provider explicitly; `npu` is an alias for
  `acl`, `hip` an alias for `rocm`, and any other name selects a third-party entry point. Without
  an explicit selection, optional backends are recognized from SDK environment variables or the
  usual SDK compiler paths; **when several SDKs are configured, explicit selection is required**,
  instead of whichever module happens to be imported first.
- Unselected providers are not imported, their `check()` does not run, and discovery neither
  invokes their compiler nor initializes their device. A configuration failure of the selected
  provider propagates upwards and is never silently swapped for another backend.
- With CPU explicitly selected, it returns before any CUDA installation check, executable
  lookup, driver query or download, even if a conflicting `nvcc_path` is set.
- The historical `has_cuda` build field describes "whether `HAS_CUDA` accelerator support was
  compiled in" and **does not prove an NVIDIA driver exists**: a successfully configured
  ACL/ROCm/Corex provider sets it as well, and `is_cuda` identifies NVIDIA compilation
  separately. Corex provides that macro itself and removes the inherited `IS_CUDA`.
- The ordinary CPU/CUDA cache fingerprint keeps its original fields and values; an explicit
  backend selection and configured optional SDK paths are added to the fingerprint fields only
  when present.

### Tool layer and resources

- `jittor_utils.compile_module` takes `ModuleBuildServices` (binding-generation callback, command
  formatter, compiler path and cache/source roots). Compiler bootstrap installs the default
  services and standalone users may pass them explicitly; with neither, it raises before writing
  any build artifact, telling the user to import jittor first -- **it never imports Jittor just to
  find the services**. There is no whole-tree source transformer that copies the checkout into a
  per-backend cache and rewrites it.
- Tensor serialization is implemented in `jittor.serialization`, not in the build utility
  package. The historical `jittor_utils.load_pytorch`, `load_pytorch_old` and `save_pytorch`
  query the loader injected by the runtime bootstrap: the functions and classes reached are
  canonical objects and the module object itself is not an alias; historical pickle `GLOBAL`
  names resolve as usual after bootstrap. Importing the old utility module on its own is
  harmless, but accessing its runtime API before `import jittor` raises explicitly.
- CUDA implementations have exactly one physical owner, `backends/cuda`: `kernels/` holds
  operator kernels and Python implementations, `libraries/<name>/{include,src}` holds library
  support code, and `include/` and `src/` hold common support resources; the Python module names
  are `jittor.backends.cuda.kernels.*` and `jittor.backends.acl.kernels.*`. The source package
  `python/jittor/backends` is only a path bridge and does not copy backend implementations;
  setuptools maps each backend package into the wheel's `jittor/backends/`, and the sdist keeps
  the top-level layout and package mappings.
- `jittor_utils.backend_resources.backend_root(jittor_path, name)` resolves the resource root in
  the source, installed or converted layout without importing Jittor: a source checkout prefers
  the top-level backend; an installed candidate requires a real package marker, so a leftover
  directory containing only `__pycache__` cannot redirect compilation. Nothing produces a
  converted layout any more; recognizing it exists only so installed old trees stay resolvable.
- The host-only index scheduler remains in CPU builds; the accelerator code-generation
  translation unit and the NaN-checking CUDA source enter accelerator builds only.

### Validation boundary

Offline tests cover immutable inputs, entry-point selection, isolation of unselected providers,
injected compilation services, and forged ACL/ROCm/Corex configurations;
`tests/structure/test_core_source_is_not_ported.py` rejects rewriting services on
`BuildContext`, any product function that "mirrors the whole tree and rewrites native source",
and rebinding `jittor_path` outside bootstrap. `tests/build/test_compile_module_dependencies.py`,
`tests/serialization/test_native_serialization_ownership.py` and
`tests/bindings/test_load_pytorch_strides.py` contain JIT-dependent checks and run serially under
the supported cache policy like other native validation. **None of these prove CANN/ROCm/Corex
ABI or device correctness**: configure the SDK on the target machine, set `JT_BACKEND`, disable
CPU fallback in the suite that supports that assertion, and test ordinary computation, extension
loading and failure propagation before claiming hardware support. For what to run on each
backend see
[`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md).

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

## Import and initialization rules

Torch's dtype objects and their native/NumPy consumption points follow the "dtype boundary"
section above.

- A module import **must not** compile kernels, download resources, modify the source checkout,
  or silently install external packages.
- **Registration must be idempotent.** Re-importing a compatibility module must not wrap the same
  callable twice or create a second module object.
- Optional dependencies are checked at the **operation boundary**, unless the import-time
  discovery itself is that API.
- A broad exception handler may annotate and **re-raise** a failure; it **must not** turn a
  half-installed compatibility interface into an apparent success.
- Expensive imports do not enter collection-only structural tests.
- A top-level definition in the same file **must not** be silently replaced by a later
  definition. Identical implementations across files are scanned as well; retained duplicates
  must belong to a narrow, reviewed category, such as an independent deployment entry point,
  backend code-generation templates, model-local architecture blocks, or historical
  serialization readers.

## Runtime resources

The following trees are consumed by the compiler or packaging code **by physical path**, so
changing them needs special review:

- `src/` (installed as `jittor/src/`)
- `backends/acl/{include,kernels/native,src}/`
- `backends/comm/`
- `python/jittor/contrib/math_util/src/`
- `compat/shim/cpp_extension/` (the optional `jittor-torch` distribution)

**A move is complete only when the source checkout, sdist, wheel, cold JIT build and installed
smoke test all agree.** Directory tidiness alone is not a reason to move these resources.

## Refactoring process

Every module move:

1. inventory definitions, assignments, imports, registrations and consumers;
2. explicitly define the canonical destination and any compatibility aliases;
3. move **one coherent domain slice**, carrying no unrelated behavioural changes;
4. compare the definition sets and public exports before and after **mechanically** where
   possible;
5. test import identity, public calls, dynamic dispatch, serialization where applicable, and the
   relevant CPU/accelerator behaviour;
6. delete the transitional source path and add it to the structural gate;
7. update the long-term documentation and active links **in the same change**.

**Two editable implementations must not survive after a move.** Compatibility must delegate to
the canonical object, and must have exit criteria.

## Acceptance

A source-layout change is acceptable when all of the following hold:

- imports and public names keep their documented behaviour;
- no historical implementation tree or root-level compatibility file remains;
- the wheel contains all required runtime resources and excludes repository-only tests/tools;
- `bash tools/check_repo_layout.sh` passes;
- targeted tests, structural tests and every affected backend gate pass;
- any **deliberate incompatibility** is recorded in the release notes.
