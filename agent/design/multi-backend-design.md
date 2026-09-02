# Multi-Backend Architecture: Current State and Proposed Design

Jittor runs on CUDA, Ascend ACL, ROCm and Corex. This document describes how
that works today, why the mechanism does not scale, and what to replace it
with. It is a proposal: nothing here is implemented.

## 1. How a backend is selected today

Three separate mechanisms, at three different times.

**Build time — first match wins.** `jittor_utils.add_backend(mod)` appends to a
list (`jittor_utils/__init__.py:796`); `compiler.py:1331` walks it and stops at
the first module whose `check()` returns true. That module then rewrites the
compiler configuration in place: ACL sets `compiler.nvcc_path = tikcc_path`
and strips `-std=c++14`, ROCm points it at `hipcc` and switches to C++17. The
chosen backend is therefore a property of the *installed toolchain*, decided
before any user code runs.

**Compile time — the source tree is rewritten.**
`jittor_utils.process_jittor_source(device_type, callback)` copies the entire
Jittor source tree into `<cache>/acl_jittor/` (or `rocm_jittor/`), passes every
`.cc`/`.h`/`.cu` file through a text callback, and repoints
`compiler.jittor_path` at the copy. ACL's callback (`acl_jittor.cc:218`,
`process_acl`) tokenises each file and performs 17 kinds of substitution:
`cudaMemcpy` → the ACL equivalent, `cudaStreamDestroy`, `cudaGetLastError` →
`0`, `_cudaGetErrorEnum` → `acl_error_to_string`, and so on. Files that resist
substitution are special-cased by name: `profiler.cc` has `.cc` rewritten to
`.tikcc`, and `pass_manager.cc` has `run_pass<FloatAtomicFixPass>();` replaced
with the literal `WTF` so the file fails to compile that pass away. The core
provides one escape hatch for this, the `JPU(x)` macro, which expands to `;`
in a normal build (`src/types.h:242`).

**Run time — there is no backend selector.** `use_acl`, `use_rocm`,
`use_corex` and `use_device` are pyjt *aliases of `use_cuda`*
(`compiler.py:224`): one integer, four names. `jt.flags.use_acl = 1` sets
`use_cuda = 1`. The runtime knows whether an accelerator is on; it does not
know which one. Code that needs to distinguish reads `jt.compiler.has_acl`,
a build-time constant.

## 2. Consequences

**A backend is a build, not a device.** One process cannot use CUDA and NPU
together, and a wheel cannot support both. There is no runtime query for "what
backend is this Var on" because a Var has neither a device nor a backend
field. The 96 `use_acl`/`has_acl` sites and 222 `use_cuda` sites in
`python/jittor` are all asking one of two questions — "was ACL selected at
build time?" or "is the accelerator on?" — in a syntax that looks like it is
asking about the current device.

**The port is a text transformation of the core.** Every change to
`executor.cc`, `allocator.cc` or `profiler.cc` can silently change ACL
behaviour, because ACL's correctness depends on the *lexical shape* of core
source it never sees reviewed together with. The `WTF` patch is the clearest
symptom: a core optimisation pass is disabled by deliberately corrupting its
call site. Adding a backend means writing a translator from CUDA-shaped C++ to
your API, not implementing an interface.

**Operator integration has three parallel dispatch paths.** For one operator
such as layer norm:

- CUDA: a kernel in `extern/cuda/<lib>/ops/*.cc`, glob-compiled by
  `setup_cuda_lib` into one shared object per library, exposed as
  `jt.cudnn.ops.*`, and selected in Python by hand-written guards in
  `nn/backends/*.py` (39 such guards).
- ACL: a `*_op_acl.cc` runner plus a `*_op.py` `jt.Function` that emits
  `jt.code(cuda_src='// aclop …')`; execution is routed by searching that
  string for `"acl"` (`acl_op_exec.cc:588`).
- Python surface: `change_function()` monkeypatches 30+ `jt.nn.*` names at
  import (`__init__.py:160`), each wrapped by a `warp()` closure that tests
  `jt.flags.use_acl` on every call (`acl_compiler.py:803`).

ACL currently carries 133 files and 13.6k lines for 39 operators; CUDA 95
files and 11.6k lines; ROCm 374 lines, because ROCm gets its operators from
the rewritten CUDA source rather than its own. Nothing enumerates which
operators a backend actually implements.

**Fallback is exception-driven and invisible.** `try_exec_and_fallback_cpu`
(`acl_op_exec.cc:223`) runs the ACL path, catches any exception, logs
`fallback cpu` and runs on CPU. A performance cliff and a correctness bug
produce the same log line, and the ecosystem harness has to grep that line to
assert no fallback occurred.

**There is no backend contract.** `add_backend` requires only `check()`;
`install_extern()` and `post_process()` are probed with `hasattr`. No document
and no contract test defines what a backend must provide, so each implements a
different subset. ACL's `post_process()` mutates four unrelated globals
(`pool_use_code_op`, `use_cuda_host_allocator`, `use_parallel_op_compiler`,
`amp_reg`) — configuration a backend needs, expressed as global side effects.

**Testing does not cross backends.** `tests/backends/` holds 6 NPU files, 1
ROCm file, 3 Triton files and a `parity` suite that compares CPU against CUDA
only. There is no matrix asserting that an operator behaves identically on
every backend that claims it.

## 3. Proposed design

Four layers, each independently useful; the ordering below is also the
implementation order.

### 3.1 Device as a value, backend as a registry

```
struct Device { BackendId backend; int index; };   // {cuda,0}, {acl,1}, {cpu,0}
```

`Var` carries a `Device`. Op outputs inherit their inputs' device; a source op
takes the current device; mixing devices in one op is a construction-time
error, as in torch. `jt.flags.use_cuda` becomes a compatibility alias meaning
"current device is not CPU", and the `use_acl`/`use_rocm`/`use_corex` aliases
are deprecated in favour of `jt.current_device()`.

A backend registers itself once:

```
BackendRegistry::register({
    .name          = "acl",
    .device_count  = …,
    .allocator     = …,   // per-device pools
    .set_device    = …,   // no process restart
    .memcpy        = …,   // H2D, D2H, peer
    .synchronize   = …,
    .stream        = …,   // may be a stub initially
});
```

This is the interface a new backend implements. It is C++, it is
version-checked, and it has a contract test.

### 3.2 An operator dispatch table instead of text rewriting and monkeypatching

```
OpRegistry::register("conv2d", BackendId::acl, &acl_conv2d);
```

The executor looks up `(op->name(), op->device().backend)` and calls the
registered kernel; a miss is an explicit, queryable condition rather than a
caught exception. This replaces all three of today's paths at once: the CUDA
Python guards, the `cuda_src.find("acl")` string search, and the
`change_function()` monkeypatch. `backend.supported_ops()` becomes a real
query, and the Python layer stops asking "which build is this".

### 3.3 Explicit fallback policy

`jt.flags.backend_fallback ∈ {error, warn, allow}`, default `warn`, with the
harness setting `error`. A fallback names the operator, the backend and the
reason. Exceptions from a backend kernel are bugs, not a routing mechanism.

### 3.4 Porting by implementation, not translation

With 3.1 and 3.2 in place, `process_jittor_source` and `process_acl` are
deleted. A backend consists of: a registry entry, a set of registered
kernels, and a build fragment that compiles only its own sources. The core
source tree stops being an input to the port, so core changes no longer
require re-validating every backend's text substitutions.

## 4. Migration

Each stage is separately verifiable and leaves the tree working.

| Stage | Change | Verified by |
| --- | --- | --- |
| 1 | `Device` on `Var`, per-device allocators, in-place `set_device` | multi-device tests on CUDA; existing gates unchanged |
| 2 | `BackendRegistry` with CPU and CUDA as its first two members; `use_cuda` reimplemented on top of it | full gates; no behaviour change expected |
| 3 | `OpRegistry` dispatch; CUDA library ops registered through it; Python guards in `nn/backends/*.py` removed | CUDA gate plus the ecosystem harness |
| 4 | ACL registered as a third backend; its `.py`/`_op_acl.cc` pairs converted to registry entries; `change_function()` deleted | NPU tests on Ascend hardware |
| 5 | `process_jittor_source` and `process_acl` deleted; ROCm ported to the registry | ROCm tests on hardware |
| 6 | Cross-backend contract suite: one operator matrix run against every registered backend | new gate tier |

Stages 1 to 3 need only CUDA hardware and are where the structural debt is
removed. Stages 4 and 5 need Ascend and ROCm machines and are where the text
rewriting finally goes away.

## 5. Scope

Out of scope for this document: stream and event semantics beyond a single
stream per device (needed eventually, designed with 3.1); heterogeneous
execution of one graph across two backends; and the JIT kernel-source parser
(`KernelIR`), which has the same "C++ as text" problem in a different place
and deserves its own proposal.
