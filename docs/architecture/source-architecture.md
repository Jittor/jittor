# Source Architecture and Module Boundaries

- Status: Accepted
- Last reviewed: 2026-08-12
- Baseline: `582fc51d`
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
│   ├── nn/                      # neural-network public API
│   │   ├── modules/             # stateful Module implementations
│   │   ├── functional/          # stateless tensor functions
│   │   ├── backends/            # explicit optimized backend adapters
│   │   ├── attention.py
│   │   └── sparse.py
│   ├── misc/                    # general tensor and shape operations
│   ├── pool/                    # pooling functions and modules
│   ├── optim/                   # optimizer facade and algorithm modules
│   ├── compat/
│   │   ├── torch/               # canonical Torch-style API compatibility
│   │   ├── fsdp2/               # distributed FSDP2 compatibility
│   │   ├── triton/              # Triton API bridge and deployment command
│   │   ├── shim/                # Torch shim runtime and deployment command
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

## Runtime resources

The following trees are consumed by compiler or packaging code using physical
paths and therefore require special review:

- `python/jittor/src/`
- `python/jittor/extern/`
- `python/jittor/math_util/src/`
- `python/jittor/utils/data.gz`
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
