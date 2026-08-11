# Jittor Repository Layout

- Status: Accepted
- Date: 2026-08-11
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

## Current Baseline

Commit `47e7ec75` contains source-refactor batches 10 and 11. Its accepted
evidence includes exact AST comparisons for the moved miscellaneous functions,
public identity and dynamic dispatch tests, 79 FSDP2 public names, 70 callable
contracts, 37 deep distributed registrations, CPU/CUDA regression, and a real
two-GPU NCCL step. Those results remain the behavioral baseline while the
temporary package shape changes.

The current wheel baseline has 1,053 entries and already contains
`UnpackRaw.cuh`; the packaging task is to make that resource explicit and tested,
not to claim it is presently missing. The deploy helper still omits
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
- `python/jittor/utils/{asm_tuner.py,dlink_compiler.py,dumpdef.py,data.gz}`;
- `python/jittor/math_util/src/*.h`;
- `python/jittor/compat/shim/cpp_extension/{include,src}/**` as the canonical
  installed ABI resource boundary for extension builds;
- `python/jittor/other/code_softmax.py` while the CUDA fast path loads it by
  filesystem location;
- `python/jittor_utils` as a sibling of `python/jittor`;
- the literal `__version__ = '...'` assignment in `python/jittor/__init__.py`
  while release and cache readers still parse it directly.

These invariants are checked as runtime contracts. They are not exemptions from
packaging completeness: every installed resource required by those contracts
must be present in the wheel.

## Target Layout

```text
.
├── pyproject.toml
├── README.md
├── LICENSE.txt
├── AGENTS.md
├── CONTRIBUTING.md
├── noxfile.py
├── .pre-commit-config.yaml
├── .github/workflows/
├── asv.conf.json
├── benchmarks/
├── docs/
├── examples/
├── tests/
├── tools/
└── python/
    ├── jittor/
    │   ├── __init__.py
    │   ├── nn/
    │   │   ├── __init__.py
    │   │   ├── modules/
    │   │   ├── functional/
    │   │   └── attention.py
    │   ├── misc/
    │   ├── pool/
    │   ├── optim/
    │   ├── compat/
    │   │   ├── torch/
    │   │   ├── shim/
    │   │   ├── fsdp2/
    │   │   ├── triton/
    │   │   ├── module_patcher.py
    │   │   └── external_backend.py
    │   ├── selftest.py
    │   ├── src/
    │   ├── extern/
    │   ├── utils/
    │   ├── math_util/
    │   └── other/
    └── jittor_utils/
```

The tree is a destination, not authorization for a flag-day move. Each migration
must leave the repository buildable and preserve the hard path invariants.

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

## Canonical And Legacy Imports

The physical target and compatibility entry points are:

| Canonical implementation | Compatibility entry point |
| --- | --- |
| `jittor.nn` package | existing `from jittor import nn` |
| `jittor.compat.torch` | `jittor.torch_compat` |
| `jittor.compat.shim` | `jittor.torch_shim` and deployed `torch` aliases |
| `jittor.compat.fsdp2` | `jittor.torch_fsdp2_compat` |
| `jittor.compat.triton` | `jittor.triton_shim` |

Compatibility paths must resolve to the canonical objects and must not run a
second installer. Where callers can observe `sys.modules` identity today, the
migration preserves one shared module object rather than parallel facades with
copied state.

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

The old LLVM alignment pass is not retired in Stage 6. A real Clang 14 cold-build
gate failed in the existing core before the pass could be evaluated, so
`compiler.compile_extern()` and `extern/llvm/jt_alignment_from_assumptions.cc`
remain together. Removing either side requires a later successful Clang cold
import, forward/backward graph, and custom JIT regression.

## Tooling And Delivery

- Ruff supplies formatting and linting; mypy is introduced with an explicit,
  ratcheted scope rather than an unmaintainable all-at-once exception list.
- Pre-commit runs fast deterministic checks, while nox is the canonical local and
  CI command entry point.
- CI has structure, CPU, CUDA self-hosted, and NPU layers.
- Releases use `python -m build` and PyPI trusted publishing. Jittor's canonical
  artifact remains `py3-none-any`; cibuildwheel is used on Linux, macOS, and
  Windows to assert that no platform wheel is accidentally produced, followed
  by installation of the same canonical wheel on all three platforms.
- Container and CI baselines share a maintained OS/Python/CUDA definition.
- ASV tracks speed and memory by commit using `--python=same`, selected revisions,
  and a Jittor cache isolated from unit tests.
- Documentation uses Sphinx, MyST, standard gettext/sphinx-intl localization,
  and jupytext-managed tutorials.

## Migration Order

1. Record this RFC and supersede conflicting architecture rules.
2. Make packaging complete and establish an immutable wheel-content baseline.
3. Add tooling, CI, release, container, and performance infrastructure.
4. Convert domain facade/private pairs into domain packages.
5. Separate compatibility capability, mechanism, root-cause fixes, and glue.
6. Add installed self-test, move the repository tests, and adopt pytest.
7. Move tools/examples and remove proven dead distribution payloads. In
   particular, `extern/llvm` is not removable while the Clang/Linux compiler path
   scans and builds it; deletion requires a replacement consumer and regression
   evidence first.
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
