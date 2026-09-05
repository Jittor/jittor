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
│   ├── _runtime/
│   │   ├── core_api.py          # native Python API after core bootstrap
│   │   └── state.py             # injected native Flags views, no bootstrap imports
│   ├── nn/                      # neural-network public API
│   │   ├── modules/             # stateful Module implementations
│   │   ├── functional/          # stateless tensor functions
│   │   ├── backends/            # explicit optimized backend adapters
│   │   ├── utils/               # construction helpers such as weight norm
│   │   └── attention.py
│   ├── autograd/                # functional automatic differentiation
│   ├── fft/                     # differentiable native FFT namespace
│   ├── misc/                    # general tensor and shape operations
│   ├── pool/                    # pooling functions and modules
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
│   ├── src/                     # JIT compiler and operator C++/CUDA sources
│   └── extern/                  # backend resources loaded by path
└── jittor_utils/                # compiler, installation, and release helpers
```

## Package composition contracts

### Neural network API

`jittor.nn` is the public package. Stateful layers live under `nn.modules`,
stateless operations under `nn.functional`, and optional accelerated paths under
`nn.backends`. A public re-export must point at the canonical implementation
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

`jittor.misc` groups tensor, shape, and composition operations. `jittor.pool`
owns pooling functions and modules. New code is placed by semantic ownership,
not by the size of the destination file. Circular imports are resolved by
narrowing dependencies or moving shared primitives to the lower-level owner,
not through mutable proxy objects.

`jittor.sparse` owns both coordinate-format sparse tensors and sparse neural
network kernels in separate child modules. The historical `jittor.nn.sparse`
name is a same-object alias of `jittor.sparse.convolution`; `jittor.nn` and
`jittor.nn.functional` re-export those canonical callables.

`jittor.autograd` owns functional automatic differentiation. `jittor.fft` owns
the differentiable FFT/shift/frequency namespace shared by native Jittor and
Torch mode. Concatenation and
indexing live in `jittor.misc`, pooling in `jittor.pool`, optimized softmax in
`jittor.nn.backends`, and weight normalization in `jittor.nn.utils`. Historical
root spellings are import aliases only; they do not retain physical source
files or wrapper implementations.

### Root module ownership

The entries directly under `python/jittor/` are an exact reviewed set.
`__init__.py` composes the runtime, while `jittor._runtime.core_api` is the one
large native Python API implementation loaded after the compiled core. Public
root exports retain object identity with that implementation, and legacy root
pickle paths remain loadable. `__init__.pyi` owns the public root typing surface.
`_runtime.state` owns `RuntimeContext` and `RuntimeState`; `core_api` re-exports
the same classes and constructs the live `jt.runtime` view after native bootstrap.
The classes read the injected native `Flags` object without copying live state.
Their snapshot returns detached Python values, not tensors. This is a Python
module boundary, not a migration of C++ global-state ownership.

Native held-root storage lives in `src/runtime/holder_state.{h,cc}`.
`RuntimeHolderState` owns both the holder list and the weak-sync cursor; the
executor, autograd, graph inspection and memory diagnostics share its exported
core accessor. It never runs a graph when registering or removing a holder.
Weak sync peeks before checking the target cutoff and advances only after that
check. Removal repairs the cursor before liveness release can re-enter the
runtime. The owner is non-copyable and has process lifetime to support late
extension/static holder destruction; it does not own the pointed-to holders.
This preserves the existing serialized mutation requirement, not a new
thread-safety guarantee. `exe`, traversal counters and native flags remain
separate pending migrations.
`compiler.py`, `compile_extern.py`,
`pyjt_compiler.py`, and `init_cupy.py` are compiler or device bootstrap
boundaries; `distributions.py`, `init.py`, and `linalg.py` are public native
domains; `selftest.py` is the installed smoke-test entry point. New root files
require an ownership review and a corresponding structure-gate update.

### Compatibility APIs

The canonical Torch-style implementation is `jittor.compat.torch`. The legacy
attribute/module spelling `jittor.torch_compat` is an alias created during Jittor
initialization; it is not a second source file. Likewise, the canonical Triton
implementation is `jittor.compat.triton`, with `jittor.triton_shim` retained as
an object-identity alias.

`jittor.compat.shim` owns the runtime and deployment code for the optional
top-level `torch` surface used by applications that import Torch directly. The
name `jittor.torch_shim` is retained only as a same-object legacy alias. The shim
delegates Torch-style semantics to `jittor.compat.torch`; neither the alias nor
the deployed package owns a second implementation.

Plain Jittor startup installs only alias resolution and native domains. The
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

- `python/jittor/src/`
- `python/jittor/extern/`
- `python/jittor/math_util/src/`
- `python/jittor/compat/shim/cpp_extension/`

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
