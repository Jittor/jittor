# Backend builds and runtime resources

[Section contents](../index.en.md) · [中文](backend-build.md)

Read this page when changing backend providers, build configuration or packaged runtime resources. Start with [source architecture](../source-architecture.en.md) for module ownership.

These are implementation contracts, not a new hardware acceptance report. The source document's review baseline was `e3c369acb`; this split preserves that evidence boundary.

## Configuration values and provider hooks

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
reduction are written in this descriptor (see [ACL kernel registration](../runtime/operator-dispatch.en.md#acl-kernel-registration)) and are selected
according to the actual accelerator target: an ACL build running a CPU scope uses the ordinary
CPU policy. Allocation is the authority on residency; frontend placement is another graph
constraint, see [device and placement](../../notes/device-placement.en.md).

## Discovery and selection

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

## Tool layer and resources

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

## Validation boundary

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
[`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/manuals/deferred-hardware.md).

## Runtime resources

The following trees are consumed by the compiler or packaging code **by physical path**, so
changing them needs special review:

- `src/` (installed as `jittor/src/`)
- `backends/acl/{include,kernels/native,src}/`
- `backends/comm/`
- `python/jittor/contrib/math_util/src/`
- `compat/shim/cpp_extension/` (the optional `jittor-compat` distribution)

**A move is complete only when the source checkout, sdist, wheel, cold JIT build and installed
smoke test all agree.** Directory tidiness alone is not a reason to move these resources.
