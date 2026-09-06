# Backend Build Configuration

- Status: Accepted
- Owner: Jittor build maintainers
- Scope: Build inputs, backend discovery, and utility dependency direction
- Hardware status: CPU/CUDA runtime checks are separate from offline provider contracts;
  ACL, ROCm, and Corex still require validation on their target machines.

## Configuration Ownership

`jittor_utils.build_config.BuildConfig` is a frozen value containing compiler
paths and flags, cache/source roots, backend capabilities, extra core sources,
environment overrides, and backend resources. Source inventories are tuples;
resource and environment mappings copy their inputs and are read-only.
Opaque loaded modules and driver handles are resources, not copies of compiler
state. `evolve(...)` returns another configuration without modifying its input.

A backend provider implements `configure(context) -> BuildConfig`.
`BuildContext` supplies explicit compilation, source transformation, dynamic
loading, and library-publication services. Providers must not import
`jittor.compiler`, assign compiler attributes, append to its source list, or
modify the process environment. Bootstrap applies returned environment changes
and publishes the compatibility compiler attributes once. `compiler.build_config`
contains the final published value; `compiler.make_backend_context()` provides
the same configuration to `install_extern(context)` and `post_process(context)`.

ACL's converter and extra source files are returned in the value. Corex removes
OpenMP from both common and kernel flags before publication. ROCm selects its
existing ABI-specific conversion object into `cache_path/rocm`, keeps the driver
and converter alive as resources, and publishes its libraries through the
injected registry callback. These changes do not replace the legacy backend
source conversions or improve their kernel algorithms.

ACL post-processing only initializes its operator registrations. It does not
rewrite pooling selection, the user's host-allocation/parallel-compilation
switches, or `amp_reg`. The copied native backend descriptor owns execution
requirements instead: ACL compilation cannot enter the unsafe parallel compiler,
ACL array staging selects the actual pinned host pool without the dual-storage
path, and ACL reductions retain the existing native low-precision/dtype policy.
Reduction and host-staging requirements are selected for the actual accelerator
target; an ACL build running a CPU scope uses the ordinary CPU policies. The
compiler restriction belongs to the selected compiler backend and cannot be
bypassed by forcing parallel compilation.

These fields extend `BackendOps` to ABI version 2. Version and exact structure
size are checked before the descriptor is copied or its policy tail is read;
extensions providing version-1 tables must rebuild. Default CPU/CUDA descriptors
retain their existing choices. Host-only contracts and syntax checks do not
prove ACL allocation, asynchronous copying, or low-precision execution on CANN.

## Discovery

Providers are registered in the `jittor.backends` package entry-point group.
Installed metadata is used when present; source checkouts have matching lazy
fallback entries for the three built-in providers. Python 3.7 uses the declared
`importlib-metadata` compatibility dependency.

`JT_BACKEND=cpu|cuda|acl|rocm|corex` selects a provider explicitly; `npu` aliases
`acl`. Other names select third-party entry points. Without an explicit choice,
SDK environment variables or the conventional SDK compiler paths identify the
optional backend. Multiple configured SDKs require an explicit choice instead
of selecting whichever module happened to import first. Unselected providers
are not imported, their `check()` functions are not run, and discovery does not
invoke their compilers or initialize their devices. A selected provider's
configuration failure propagates rather than silently selecting another backend.

Explicit CPU selection returns before CUDA installation checks, executable
searches, driver queries, or downloads, even when a conflicting `nvcc_path` or
automatic-install setting is present. Optional legacy providers still retain
their pre-existing shared toolchain discovery; removing every such SDK coupling
requires target-machine verification and is not claimed here.

The historical `has_cuda` build field describes compilation of `HAS_CUDA`
accelerator support, not proof of an NVIDIA driver. A successfully configured
ACL/ROCm/Corex provider sets it even when the initial CPU configuration found no
NVCC; `is_cuda` separately identifies NVIDIA compilation. Before this migration,
ACL/ROCm added `-DHAS_CUDA` without updating the Python field, and Corex could
depend on NVIDIA discovery even to receive that macro. Corex now supplies it and
removes inherited `IS_CUDA`. CUDA-compatible library setup remains available to
Corex, while ACL uses its fake-library adapters and ROCm its injected loaders.

Ordinary CPU/CUDA cache fingerprints retain their previous fields and values.
Explicit backend selection and configured optional SDK paths add fingerprint
fields only when present. Build stamps include the configuration/discovery code;
stamp invalidation still uses normal dependency checks, not a new cache root.

## Utility Boundary

`jittor_utils.compile_module` receives `ModuleBuildServices`: a binding-generator
callback, command formatter, compiler path, and cache/source roots. Compiler
bootstrap installs its default services; standalone users can pass services
explicitly. Without either, the function raises before writing build artifacts
and directs the caller to import Jittor first. It never imports Jittor to find
its own services. Source conversion similarly accepts a configuration and
returns a new value, preserving obsolete transformed sources in a cache archive.

Tensor serialization implementations live in `jittor.serialization`, not in the
build utility package. Historical `jittor_utils.load_pytorch`,
`load_pytorch_old`, and `save_pytorch` modules query loaders injected by runtime
bootstrap. Accessed functions and classes are the canonical objects; the module
objects themselves are not aliases. Historical pickle `GLOBAL` names continue
to resolve after bootstrap. Importing an old utility module alone is harmless,
but accessing its runtime API before `import jittor` now raises explicitly;
independent pre-bootstrap unpickling is not claimed compatible. Registering the
loaders does not import Torch or eagerly load the serialization implementations.

## Backend Resources

CUDA implementations have one physical source owner under top-level
`backends/cuda`: `kernels/` holds operator kernels and Python implementations,
`libraries/<name>/{include,src}` holds library support code, and `include/` and
`src/` hold common support resources. ACL's extracted Python KV-cache kernels
live under `backends/acl/kernels`. Their Python module names are
`jittor.backends.cuda.kernels.*` and `jittor.backends.acl.kernels.*`.

The source package `python/jittor/backends` contains only a path bridge; it does
not duplicate backend implementations. Setuptools maps the two backend packages
into `jittor/backends/` in a wheel, and the source distribution preserves the
top-level layout and package mapping. Headers, CUDA sources, host code-generation
translation units, and Python kernels are all runtime resources.

`jittor_utils.backend_resources.backend_root(jittor_path, name)` resolves the
source, installed, or converted resource root without importing Jittor. A source
checkout prefers its top-level backend. Installed candidates require a real
package marker, so a leftover directory containing only `__pycache__` cannot
redirect compilation away from the actual resources.

Legacy source conversion copies and transforms moved native files into
`<converted-jittor>/backends/` with the same callback used for core sources.
The returned configuration records these roots in `resources['backend_roots']`
and rewrites include paths to the converted copies. Installed trees are not
transformed twice. Obsolete converted native paths are archived rather than
compiled alongside their replacement. Runtime library discovery includes both
kernel and library-support sources, including cuTT's separate wrapper.

The backend's pure host indexing scheduler remains part of CPU builds; the
accelerator code-generation translation units and NaN-checking CUDA source are
included only in accelerator builds. The core `src/` tree is not otherwise
relocated. NCCL retains its distributed resource location, and the remaining
ACL/ROCm/Corex sources and legacy conversion machinery are not claimed migrated
by this resource-layout change. It does not complete the larger layout or lazy
initialization tasks.

## Validation

Offline tests cover immutable inputs, entry-point selection, unselected-provider
isolation, source-cache migration, injected compilation services, and fake
ACL/ROCm/Corex configuration. They do not establish CANN/ROCm/Corex ABI or device
correctness. On the target machine, configure the SDK and select `JT_BACKEND`,
then run its maintained backend suite with CPU fallback disallowed where the
suite supports that assertion. Test ordinary computation, extension loading,
and failure propagation before claiming hardware support.

The runtime handoff also includes
`tests/compiler/test_compile_module_dependencies.py`,
`tests/core/test_native_serialization_ownership.py`, and
`tests/core/test_load_pytorch_strides.py`. Their JIT-dependent checks must run
serially against the same supported cache policy as other native validation.
