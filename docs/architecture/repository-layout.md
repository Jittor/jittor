# Jittor Repository Layout

- Status: Accepted
- Date: 2026-08-11
- Last reviewed: 2026-08-31
- Baseline: `f5e8e944` plus the boundary documentation changes described here
- Scope: Jittor 2.0 repository structure, packaging, compatibility ownership,
  tests, tooling, and documentation
- Supersedes: the long-term `facade.py + _private/` target and the decision to
  ship `jittor.test` as a public package

## Context

The repository grew from an upstream source tree by accumulating compatibility
layers, backend adapters, tests, release scripts, and generated documentation.
The resulting layout has three systemic problems:

1. Packaging is not authoritative. A short explicit package list and recursive
   `package_data` globs happen to place undeclared Python packages in wheels,
   while deeper runtime resources can be omitted.
2. Domain modules use a public file plus a parallel `_domain/` implementation
   package. Runtime proxies and rewritten `__module__` metadata are then needed
   to work around the artificial boundary.
3. Framework capability, reusable patching mechanisms, Jittor semantic fixes,
   and downstream project glue are mixed in the Torch compatibility layer.

This RFC defines the end state. Earlier facade splits remain useful migration
work, but their private-package shape is transitional rather than the target.

## Decision Baseline

At the time of this decision, commit `47e7ec75` contained source-refactor
batches 10 and 11. Its accepted
evidence includes exact AST comparisons for the moved miscellaneous functions,
public identity and dynamic dispatch tests, 79 FSDP2 public names, 70 callable
contracts, 37 deep distributed registrations, CPU/CUDA regression, and a real
two-GPU NCCL step. Those results remain the behavioral baseline while the
temporary package shape changes.

That wheel baseline had 1,053 entries and already contained `UnpackRaw.cuh`;
the packaging task was to make that resource explicit and tested, not to claim
it was missing. At that point, the deploy helper still omitted
`flash_attn_interface.py` when copying stubs into a deployment tree.

## Goals

- Make installed artifacts derive from one auditable package and resource
  manifest.
- Use conventional Python domain packages whose physical module paths express
  ownership.
- Separate reusable Torch compatibility from project-specific integrations.
- Move tests, examples, tools, benchmarks, and documentation out of the runtime
  package unless they are required after installation.
- Turn correctness, packaging, lint, and performance requirements into automated
  gates.
- Preserve Jittor's JIT runtime resource paths and public import compatibility
  throughout the migration.

## Non-goals

- Replacing Jittor's meta-operator or unified graph design with PyTorch
  internals.
- Moving runtime C++/CUDA resources merely to make the source tree look uniform.
- Claiming backend capability from import or structure tests alone.
- Keeping every historical reflection path forever. Where a deliberate public
  metadata change is made, old serialized artifacts receive an explicit
  compatibility path and a migration test.

## Hard Path Invariants

`python/jittor` is the runtime `jittor_path`, not only a Python package. The
following paths remain physically stable unless the compiler contract is first
changed and independently validated:

- `python/jittor/src/**`, including extensionless resources and the source files
  selected by compiler ordering rules;
- `python/jittor/extern/**`, including the `<backend>/{inc,ops,src}` layout and
  ACL filename dispatch conventions;
- `python/jittor/utils/{dlink_compiler.py,dumpdef.py}`;
- `python/jittor/math_util/src/*.h`;
- `python/jittor/compat/shim/cpp_extension/{include,src}/**` as the canonical
  installed ABI resource boundary for extension builds;
- `python/jittor_utils` as a sibling of `python/jittor`;
- the literal `__version__ = '...'` assignment in `python/jittor/__init__.py`
  while release and cache readers still parse it directly.

These invariants are checked as runtime contracts. They are not exemptions from
packaging completeness: every installed resource required by those contracts
must be present in the wheel.

## Target Layout

The tree below is the destination decided on 2026-09-02. The reasoning, a
source-to-destination table for every move, the packaging coupling and the
sequencing live in [`agent/design/target-layout.md`](../../agent/design/target-layout.md).
It replaces the earlier tree in this section, which described the layout as it
stood after the 2.0 domain-package migration rather than where it should go.

```text
.
├── pyproject.toml  README.md  LICENSE.txt  AGENTS.md  CONTRIBUTING.md  noxfile.py
├── src/                      # C++ core, beside the Python package, not package data
│   ├── core/                 # node var op graph grad executor fused_op
│   ├── type/  mem/  codegen/  ops/  runtime/  bindings/  third_party/  tests/
├── backends/                 # one shape per backend: build fragment + kernels/ + registry entries
│   ├── cpu/  cuda/  acl/  rocm/  corex/
│   └── comm/                 # mpi nccl hccl
├── python/jittor/            # pure Python
│   ├── _core/                # var module function flags hooks
│   ├── build/                # compiler compile_extern pyjt_compiler cuda_wheel install_cuda + jittor_utils
│   ├── ops/                  # today's misc/, with tensor_ops split by domain
│   ├── nn/  optim/  autograd/  fft/  sparse/  dataset/  transform/  models/
│   ├── linalg/  distributions/  init/
│   ├── distributed/  contrib/  tools/
├── compat/                   # separate distribution (jittor-torch): torch shim fsdp2 vllm triton
├── tools/  tests/  docs/  examples/  benchmarks/
└── agent/                    # manuals/ skills/ scripts/ only
```

Three rules the old tree did not state:

- The C++ core and the backends are not Python package data. They live beside
  the package and are built by an explicit step, not by `import jittor`.
- One concept, one place. A CUDA kernel lives under `backends/cuda/` whichever
  Python layer calls it; a build tool lives under `jittor/build/`; a tensor
  operation lives under `jittor/ops/`.
- Layout moves are the last step of each refactor phase, never the first. A
  move only makes sense once the code it moves has one shape; see the
  "布局收尾" rows of [`agent/design/refactor-plan.md`](../../agent/design/refactor-plan.md).

The exact entry set asserted by `test_runtime_root_has_an_exact_reviewed_entry_set`
freezes the *current* tree, not this one. It is converted into rule-based checks
(plan task 0.19) before the first move lands; until then every move updates that
set in the same commit. Public 1.x import paths (`jittor.ccl.ccl_2d`,
`jittor.pool.AvgPool2d`, `jittor.misc.*`) survive every move as deprecated
forwarding modules for one major version.

## Decision 1: Domain Packages

Stage 3 converged `nn.py + _nn/`, `misc.py + _misc/`, `pool.py + _pool/`, and
`torch_compat.py + _torch_compat/` to the normal `nn/`, `misc/`, `pool/`, and
`compat/torch/` packages. The following rules are the resulting contract.

- A domain package's `__init__.py` is its composition and re-export surface.
- Implementations live under meaningful paths such as `nn.modules.conv` and
  `nn.functional.padding`.
- Implementations may import the initialized domain interfaces they actually
  depend on. The end state does not contain duplicated `_JittorRuntimeProxy`
  service locators.
- Callable metadata reports the real implementation module, matching mainstream
  Python library practice. New code does not recursively rewrite `__module__` to
  hide ownership.
- Historical imports such as `from jittor import nn` remain valid.
- Pickle compatibility is verified against fixtures made before the move.
  Compatibility aliases or a narrowly scoped unpickling mapping are retained
  only where an old artifact cannot otherwise load.
- Backend post-processing and monkeypatch points become explicit registries or
  documented extension hooks, not accidental mutation of facade globals.

The four Stage 3 facade/private-package pairs and the later FSDP2 migration
scaffold no longer exist. The maintained Torch API lives in
`jittor.compat.torch`, FSDP2 support lives in `jittor.compat.fsdp2`, and
deployment and import patching live in `jittor.compat.shim`. No new domain may
adopt a facade/private-package pairing.

The runtime root is also closed by an exact structure contract. Its remaining
Python files are limited to runtime composition, compiler/device bootstrap,
the native `distributions`, `init`, and `linalg` domains, and the installed
self-test. Compatibility files such as `contrib.py`, `weightnorm.py`,
`lr_scheduler.py`, `sparse.py`, or `torch_fsdp2_compat.py` are not valid root
owners.

## Canonical And Legacy Imports

The physical target and compatibility entry points are:

| Canonical implementation | Compatibility entry point |
| --- | --- |
| `jittor.nn` package | existing `from jittor import nn` |
| `jittor.autograd` | `jittor.gradfunctional` |
| `jittor.fft` | `torch.fft` in Torch mode |
| `jittor.misc.concatenation` and `jittor.misc.indexing` | `jittor.contrib` |
| `jittor.nn.backends.softmax_cuda` | `jittor.other.code_softmax` |
| `jittor.nn.utils.weight_norm` | `jittor.weightnorm` |
| `jittor.optim.legacy_schedulers` | `jittor.lr_scheduler` |
| `jittor.sparse.convolution` | `jittor.nn.sparse` |
| `jittor.compat.torch` | `jittor.torch_compat` |
| `jittor.compat.shim` | `jittor.torch_shim` and deployed `torch` aliases |
| `jittor.compat.fsdp2` | `jittor.torch_fsdp2_compat` |
| `jittor.compat.triton` | `jittor.triton_shim` |

Compatibility paths must resolve to the canonical objects and must not own a
second implementation or installer. The historical `jittor.torch_compat` entry
may idempotently activate the canonical installer because plain Jittor startup
now deliberately preserves native semantics. Where callers can observe
`sys.modules` identity today, the migration preserves one shared module object
rather than parallel facades with copied state.

The structure gate scans same-file shadowed definitions and exact cross-file
implementation bodies. A small reviewed set remains where independence is a
real contract: filesystem-only deployment commands, ACL code-generation
templates, architecture-local model helpers, and the old/new PyTorch checkpoint
readers. A new duplicate group fails until it is consolidated or explicitly
justified in that gate.

`jittor.nn.functional` becomes a physical package and remains the canonical
object registered as `torch.nn.functional` when the Torch shim is active. The
conversion must remove the current dynamic-module constructor without allowing
the shim to overwrite the package or duplicate registrations.

## Decision 2: Four Compatibility Layers

Torch compatibility is separated by ownership rather than downstream project:

1. **Framework capability** belongs in Jittor domains. Examples include
   parameterized attention, RoPE, normalization, mesh, and production sparse
   convolution operations.
2. **Reusable mechanism** belongs in `jittor.compat`. A registry-based module
   patcher owns import timing and restoration; a parameterized external-backend
   loader owns source discovery, manifests, builds, and cache identity.
3. **Jittor root-cause fixes** belong in core semantics. Tensor gradient state,
   `Parameter` identity, and shape hashability must not be papered over in each
   downstream library.
4. **Project/version glue** lives in optional integration distributions such as
   `jittor-trellis`, `jittor-gs`, and `jittor-hf-compat`, registered through
   entry points. Mainline Jittor does not install a permanent TRELLIS or Gaussian
   Splatting finder for every process.

`jittor.compat.vllm` is the explicit staged exception while the vLLM integration
and its external device plugin are still converging. Structure tests constrain
it to public Jittor APIs and the module-patcher entry point so extraction does
not require a core API rewrite. It must move to a versioned, installable plugin
once that plugin preserves the maintained correctness and performance gates;
the external platform and worker adapter are not owned by `python/jittor`.

Installers must report each attempted patch independently. A broad
`try/except Exception: pass` around a chain of patches is forbidden because one
failure otherwise hides all later compatibility work.

## Packaging Boundary

- `pyproject.toml` is the authoritative project metadata and build-system entry.
- Package discovery includes every directory under `python/` that contains an
  `__init__.py`; a structure test compares the filesystem and build metadata.
- Runtime non-Python resources are declared explicitly through `MANIFEST.in` and
  package-data configuration. Recursive catch-all globs are not used to smuggle
  undeclared packages into a wheel.
- Tests, demos, notebooks, release utilities, and benchmarks are not wheel
  payload unless a documented installed command requires them.
- `jittor.selftest` is the small installed smoke test used by install scripts and
  container health checks.
- Every packaging change is checked by a member SHA-256 plus exact-path wheel
  baseline. Additions, content changes, and removals require reviewed hashes or
  path allowances, and every allowance must be consumed by the candidate so a
  stale wheel cannot silently pass by reverting an approved transition.

## Test Boundary

The repository test suite lives under top-level `tests/` and is collected by
pytest. It is not an installed `jittor` subpackage. Existing
`unittest.TestCase` tests remain valid under pytest and were not rewritten merely
for the move.

The migration inventory was reconciled from the original 233-module estimate to
239 top-level test modules. Its 123 cross-test edges across 76 files were replaced
with explicit helpers derived from the actual graph; all 239 modules received a
recorded destination rather than relying on the older partial category estimate.

- Shared test utilities live in `tests/_helpers/`; test modules do not import
  one another as an implicit helper API.
- Structure, CPU, CUDA, and NPU suites are separate CI layers.
- Test selection uses names and markers, never `listdir()` positions.
- The C++ `test.h` resource lives with its compiler include consumers under
  `python/jittor/src/utils/`.
- Installed-environment checks use `jittor.selftest`, not a shipped test package.

## Tool And Example Boundary

Repository-operated commands live under top-level `tools/`; teaching and runnable
samples live under `examples/`. Neither tree contains `__init__.py` or participates
in runtime package discovery. They are included in the source distribution for
maintainers, but direct and sdist-derived wheels exclude them.

Stage 6 retires the published `jittor.vcompiler` package as a deliberate Jittor
2.0 breaking change. It compiled a private C++ extension during import, had no
supported in-repository consumer, and has no direct replacement; custom operator
authors should use the maintained `compile_custom_op` and `compile_custom_ops`
interfaces instead. The package-local `version` file is also retired because
project metadata and `jittor.__version__` are the version authorities.

Stage 6 retires the old LLVM alignment pass and the unused
`compiler.compile_extern()` build hook together. The active compiler startup did
not call that hook; keeping a private pass tied to LLVM compiler internals added
an unsupported distribution payload without affecting the runtime JIT path.
The normal Clang cold import, forward/backward graph, and custom JIT regressions
remain the acceptance gate for the supported compiler path.

## Tooling And Delivery

- Ruff supplies formatting and linting; mypy is introduced with an explicit,
  ratcheted scope rather than an unmaintainable all-at-once exception list.
- Pre-commit runs fast deterministic checks, while nox is the canonical local and
  CI command entry point.
- `.github/ci-baseline.env` is the single maintained definition for the host
  runner, Python version, CPU CI image, CUDA requirement, hardware runner
  labels, container matrix, and release validation matrix. The reusable
  `_ci-baseline.yml` workflow exports those values; individual workflows consume
  its outputs instead of restating versions or runner labels.
- CPU tests and packaging run in the CPU image from that baseline. CUDA and NPU
  jobs are a formal exception to container parity: they run on labeled
  self-hosted hosts because the device, driver, toolkit, and vendor runtime are
  host capabilities. macOS and Windows release checks are the corresponding
  native-platform exception. These jobs still consume the common Python,
  toolkit, runner-label, or platform-matrix values that apply to them.
- Releases use `python -m build` and PyPI trusted publishing. Jittor ships
  Python plus JIT source resources rather than a prebuilt platform extension, so
  its one canonical wheel remains `py3-none-any`. The release builds that wheel
  once and installs the same artifact on Linux, macOS, and Windows. The
  cibuildwheel checks assert that a platform wheel is *not* produced; they do not
  replace the canonical wheel with three platform-specific wheels.
- ASV tracks speed and memory by commit using `--python=same`, selected revisions,
  a Jittor cache isolated from unit tests, and externally stored results and HTML
  reports.
- Documentation uses Sphinx, MyST, standard gettext/sphinx-intl localization,
  and jupytext-managed tutorials.

## Migration Order

1. Record this RFC and supersede conflicting architecture rules.
2. Make packaging complete and establish an immutable wheel-content baseline.
3. Add tooling, CI, release, container, and performance infrastructure.
4. Convert domain facade/private pairs into domain packages.
5. Separate compatibility capability, mechanism, root-cause fixes, and glue.
6. Add installed self-test, move the repository tests, and adopt pytest.
7. Move tools/examples and remove proven dead distribution payloads, including
   the uncalled `compiler.compile_extern()` hook and its private `extern/llvm`
   plugin source.
8. Finish remaining module decomposition and reduce root initialization to a
   composition root.
9. Modernize documentation, localization, tutorials, and governance.

## Stage Gates

Every stage must provide all applicable evidence below before it is considered
complete:

- member-hash and exact-path wheel comparison against the accepted baseline;
  additions, content changes, and removals require exact, fully consumed
  allowlists tied to the phase that changes the distribution boundary;
- cold import from an isolated wheel installation;
- no regression in the affected CPU matrix;
- at least one real accelerator regression, with NPU evidence required before a
  capability is described as NPU-supported;
- `agent/scripts/check_repo_layout.sh` and repository structure checks;
- documentation of commands, results, known skips, and unverified backends;
- a focused commit that does not stage unrelated user changes.

Compatibility extraction additionally requires the TRELLIS and Gaussian
Splatting end-to-end paths to retain their established numerical and performance
results after installation from optional integration packages.

## Consequences

The migration temporarily contains both old and new shapes, and some public
callables will eventually report more specific implementation modules. In
exchange, package contents become auditable, dependency direction is visible in
the filesystem, third-party integration cost becomes opt-in, and correctness and
performance requirements become enforceable rather than documentary.
