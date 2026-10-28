# Repository layout

- Status: accepted
- Date: 2026-08-11
- Last reviewed: 2026-09-09
- Baseline: `f5e8e944` plus the boundary-documentation changes described here
- Scope: the repository structure, packaging, compatibility ownership, tests, tools and
  documentation of Jittor 2.0
- Supersedes: the long-lived `facade.py + _private/` goal, and the decision to publish
  `jittor.test` as a public package

## Background

This repository grew out of an upstream source tree by accumulating a compatibility layer,
backend adaptations, tests, release scripts and generated documentation, which produced three
systemic problems:

1. **Packaging was not authoritative.** A short explicit package list plus a recursive
   `package_data` glob happened to put undeclared Python packages into the wheel, while
   deeper runtime resources could be missed.
2. **Domain modules took the "public file plus a parallel `_domain/` implementation package"
   form**, which then needed runtime proxies and rewritten `__module__` metadata to work
   around that artificial boundary.
3. **Framework capabilities, reusable patch mechanisms, Jittor semantic fixes and
   downstream-project glue code were mixed together inside the Torch compatibility layer.**

This document defines the end state. The earlier facade split remains useful migration work,
but its private-package form is **a transitional state, not the goal**.

## Goals

- Derive the installed artifacts **from a single auditable list of packages and resources**;
- Use ordinary Python domain packages, so that **the physical module path expresses
  ownership**;
- Separate reusable Torch compatibility from project-specific integrations;
- Move tests, examples, tools, benchmarks and documentation out of the runtime package unless
  they are genuinely needed after installation;
- Turn correctness, packaging, lint and performance requirements into **automatic gates**;
- Preserve Jittor's JIT runtime resource paths and public imports throughout the migration.

## Non-goals

- Replacing Jittor's meta-operators or unified graph design with PyTorch internals;
- Moving runtime C++/CUDA resources purely to make the source tree look tidy;
- Claiming backend capability from imports or structural tests alone;
- Preserving every historical reflection path forever. Any deliberate change to public
  metadata gives old serialized artifacts an explicit compatibility path and a migration test.

## Hard path invariants

`python/jittor` is the runtime's `jittor_path`, not merely a Python package. The following
paths stay physically stable until **the compiler contract has been changed first and
validated independently**:

- `src/**`, installed as `jittor/src`, including the source files selected by the compiler's
  ordering rules;
- `backends/**`, mapped to the installed `jittor.backends` packages;
- `python/jittor/build/{dlink_compiler.py,dumpdef.py}` and `python/jittor/tools/tracer.py`
  (the tracer is reached by the binding layer through its explicit module name);
- `python/jittor/contrib/math_util/src/*.h`;
- `compat/shim/cpp_extension/{include,src}/**`, **owned exclusively** by the optional
  `jittor-torch` distribution as the installed ABI resource boundary for extensions;
- `python/jittor_utils` as a sibling directory of `python/jittor`;
- the literal `__version__ = '...'` assignment in `python/jittor/__init__.py` -- as long as
  release and cache readers still parse it directly.

These invariants are checked **as runtime contracts**. They are **not an exemption from
packaging completeness**: every installed resource these contracts need must appear in the
wheel.

## Documentation ownership

External documentation is a single `docs/` tree, aimed at users and contributors: the entry
point, tutorials, API reference, user guides, Torch compatibility, mechanism notes
(`docs/notes/`), performance, release notes, community, and contributor documentation
(`docs/development/`). **The source language is Chinese.** Every page is maintained in
pairs: `X.md` is the Chinese default and `X.en.md` is the English page in the same
directory; the root `README.md` and `README.en.md` are paired the same way. Each index
(`index.md` / `index.en.md`) links only within its own language.

**The repository keeps no process documents such as task boards, plans, handoff notes or
assignments.** Tasks, status and handoff are recorded in GitHub issues and PRs, and validation
evidence goes into PR descriptions; items awaiting hardware acceptance are in
[`agent/manuals/deferred-hardware.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/deferred-hardware.md),
and defects and limitations are in the
[issue ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md).
Process material from the 2.0 cleanup was deleted at close-out (it remains in Git history):
design contracts that still hold moved into `docs/development/`, `docs/notes/` and
`docs/compatibility/`, validation reports that are still the only evidence for an open issue
were condensed into `docs/results/`, and the wheel-content baseline moved into
`tools/release/baselines/`.

Active indexes and inbound links must resolve; archived conclusions and their original test
evidence are **not rewritten into present-day success**. The documentation checks maintain this
"active/archived" boundary.

`agent/` contains only `manuals/` and `skills/`; repository check scripts always live in
`tools/`. The workflow entry point is
[`agent/manuals/agent-index.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/agent-index.md).
The community project list is in `docs/community/`; the ASV source configuration is
`benchmarks/asv/asv.conf.json` -- nox derives absolute input and external output paths from
that configuration and does not move benchmark run state into the working tree.
All benchmarks belong under [`benchmarks/`](https://github.com/Jittor/jittor/blob/2.0-refactor/benchmarks/README.md):
`asv/` holds the performance regression suite, `torch_compat/` the real-model comparisons, and
`acl/` the Ascend measurements, numerical cross-checks and CANN probes.

## Tool and package-resource ownership

Repository commands live in `tools/`: build checks in `tools/build/`, documentation governance
in `tools/docs/`, release-artifact checks in `tools/release/`, and the layout entry point is
`tools/check_repo_layout.sh`. `python/jittor/tools/` keeps the user-facing NVTX, jtune, tracing
and `jt.benchmark` implementations -- **it is not a second maintainer tool directory**.

Each distribution declares its runtime resources in its own
`[tool.setuptools.package-data]` in `pyproject.toml`. The core maps the data-only `jittor.src`
namespace onto the checkout's `src/`, and setuptools copies the files accordingly, **without a
second suffix list in setup.py**. The backends' package-dir mappings and the same resource
table preserve their installed `jittor/backends/` paths. The standalone compat project owns its
own shim resources and the Torch entry point.

`tools/build/generate_manifest.py` derives the two MANIFEST.in files from those tables plus the
"source only" selections of `[tool.jittor.sdist]`. It generates **exact paths**, to avoid the
non-recursive behaviour of `**` in MANIFEST; source filenames containing spaces use anchored
single-character patterns, because MANIFEST has no whitespace-quoting syntax. Cache and
generated-output directories are removed before enumeration. **After adding source-only files
or changing a declaration, re-run the generator**; `--check` is enforced by packaging sessions
and by the structural gate. Both sdists build independently with setuptools, without importing
the runtime and without relying on sibling distributions to read the resource list.

## Target layout

The tree below is the layout decided on 2026-09-02 and now in place. The item-by-item
source-to-destination table and the migration ordering were part of the cleanup process and are
preserved only in Git history; the layout itself is guarded by the structural gates and by
`tools/check_repo_layout.sh`.

```text
.
├── pyproject.toml  README.md  README.en.md  LICENSE.txt  AGENTS.md  CONTRIBUTING.md  noxfile.py
├── src/                      # C++ 核心，与 Python 包并列，不是包数据
│   ├── core/                 # node var op graph grad executor fused_op
│   ├── type/  mem/  codegen/  ops/  runtime/  bindings/  third_party/  tests/
├── backends/                 # 每个后端一种形态：构建片段 + kernels/ + 注册项
│   ├── cpu/  cuda/  acl/  rocm/  corex/
│   └── comm/                 # mpi nccl hccl
├── python/jittor/            # 纯 Python
│   ├── _core/                # var module function flags hooks
│   ├── build/                # compiler compile_extern pyjt_compiler cuda_wheel install_cuda + jittor_utils
│   ├── ops/                  # 原 misc/，tensor_ops 按领域拆分
│   ├── nn/  optim/  autograd/  fft/  sparse/  dataset/  transform/  models/
│   ├── linalg/  distributions/  init/
│   ├── distributed/  contrib/  tools/
├── compat/                   # 独立发行物（jittor-torch）：torch shim、fsdp2、triton；vLLM 属于 adapters/
├── tools/  tests/  docs/  examples/  benchmarks/
└── agent/                    # 只有 manuals/ 与 skills/
```

Three rules the old tree did not state:

- **The C++ core and the backends are not Python package data.** They sit beside the package,
  produced by explicit build steps rather than by `import jittor`.
- **One concept, one location.** A CUDA kernel lives in `backends/cuda/` no matter which Python
  layer calls it; build tools live in `jittor/build/`; tensor operations live in `jittor/ops/`.
- **Moving the layout is the last step of every refactoring phase, never the first.** Moving is
  only meaningful once the code being moved has converged into one form.

The public 1.x import paths (`jittor.ccl.ccl_2d`, `jittor.pool.AvgPool2d`, `jittor.misc.*`)
survive for one major version after each move as **deprecated forwarding modules**.

## Decision 1: domain packages

- A domain package's `__init__.py` is its **composition and re-export surface**.
- Implementations live under meaningful paths, for example `nn.modules.conv`,
  `nn.functional.padding`.
- Implementations may import the initialized domain interfaces they **actually depend on**. In
  the end state **there is no** duplicated `_JittorRuntimeProxy` service locator.
- Callable metadata reports the **real implementation module**, matching common Python
  library practice. New code **no longer recursively rewrites `__module__` to hide
  ownership**.
- Historical imports such as `from jittor import nn` keep working.
- **Pickle compatibility is verified with fixtures made before the move.** Compatibility
  aliases or a narrowly scoped deserialization map survive only when old artifacts genuinely
  cannot be loaded otherwise.
- Backend post-processing and monkeypatch points become **explicit registries or documented
  extension hooks**, not incidental mutations of facade globals.

The four facade/private-package pairs of Stage 3 and the later FSDP2 migration scaffolding no
longer exist. The maintained Torch API is in `jittor.compat.torch`, FSDP2 support in
`jittor.compat.fsdp2`, and deployment and import patching in `jittor.compat.shim`. **No new
domain may adopt a facade/private-package pair.**

The runtime root is also closed by an exact structural contract: the remaining Python files are
limited to runtime composition, compiler/device bootstrap, the native `distributions`/`init`/
`linalg` domains, and the installed self-check. Compatibility files such as `contrib.py`,
`weightnorm.py`, `lr_scheduler.py`, `sparse.py` and `torch_fsdp2_compat.py` are **not legitimate
root owners**.

## Canonical imports and historical imports

| Canonical implementation | Compatibility entry point |
| --- | --- |
| the `jittor.nn` package | the existing `from jittor import nn` |
| `jittor.autograd` | `jittor.gradfunctional` |
| `jittor.fft` | `torch.fft` in Torch mode |
| `jittor.misc.concatenation` and `jittor.misc.indexing` | `jittor.contrib` |
| `jittor.nn.backends.softmax_cuda` | `jittor.other.code_softmax` |
| `jittor.nn.utils.weight_norm` | `jittor.weightnorm` |
| `jittor.optim.legacy_schedulers` | `jittor.lr_scheduler` |
| `jittor.sparse.convolution` | `jittor.nn.sparse` |
| `jittor.compat.torch` | `jittor.torch_compat` |
| `jittor.compat.shim` | `jittor.torch_shim` and the deployed `torch` alias |
| `jittor.compat.fsdp2` | `jittor.torch_fsdp2_compat` |
| `jittor.compat.triton` | `jittor.triton_shim` |

**A compatibility path must resolve to the canonical object and must not own a second
implementation or installer.** The historical `jittor.torch_compat` entry point may activate
the canonical installer idempotently, because ordinary Jittor startup now deliberately stays
native. Where callers can observe `sys.modules` identity today, the migration keeps **the same
shared module object** rather than a parallel facade with copied state.

The structural gate scans for shadowed definitions within a file and for identical
implementation bodies across files. There is a small set of reviewed exceptions, because their
independence is a real contract: filesystem-only deployment commands, ACL code-generation
templates, architecture-local model helpers, and the readers of old and new PyTorch
checkpoints. **Any newly appearing duplicate group fails until it is merged or explicitly
explained in that gate.**

## Decision 2: four layers of compatibility

Torch compatibility is split by **ownership**, not by downstream project:

1. **Framework capabilities** belong to Jittor's domains. Examples are parameterized
   attention, RoPE, normalization, mesh and production sparse-convolution operations.
2. **Reusable mechanisms** belong to `jittor.compat`. The registry-based module patcher owns
   import timing and restoration; the parameterized external-backend loader owns source
   discovery, manifest, build and cache identity.
3. **Root-cause fixes in Jittor** belong to the core semantics. Tensor gradient state,
   `Parameter` identity and shape hashability **must not be patched separately in every
   downstream library**.
4. **Project/version glue** lives in optional integration distributions, for example
   `jittor-trellis`, `jittor-gs`, `jittor-hf-compat`, registered through entry points.
   **Mainline Jittor installs no resident TRELLIS or Gaussian Splatting finder into every
   process.**

The vLLM glue code and its dedicated tests are owned by `adapters/jittor_adapters/vllm` and
`adapters/tests/vllm` in the main repository, distributed with the existing
`jittor-torch-adapters` distribution, **without a separate repository or release**. Neither the
core nor `jittor-torch` packages that implementation, nor the abandoned `jittor.compat.vllm`.
The optional Torch stage only discovers a module-patch entry point named `jittor_vllm`; when it
is absent the stage stays optional. That adapter uses the shared, owner-aware rollback mechanism
to save pre-import extension settings and post-import patches.

**An installer must report every attempted patch independently.** A broad
`try/except Exception: pass` wrapped around a series of patches is forbidden -- a single failure
would otherwise hide all subsequent compatibility work.

## Packaging boundary

- `pyproject.toml` is the authoritative project metadata and build-system entry point.
- Package discovery includes every directory under `python/` that contains `__init__.py`; a
  structural test compares the filesystem with the build metadata.
- Non-Python runtime resources are **declared explicitly** through `MANIFEST.in` and
  package-data configuration. **No recursive catch-all glob smuggles undeclared packages into
  the wheel.**
- Tests, demos, notebooks, release tooling and benchmarks are **not wheel payloads** unless a
  documented installed command needs them.
- `jittor.selftest` is the small installed smoke test used by install scripts and container
  health checks.
- Every packaging change is checked against a wheel baseline of **member SHA-256 plus exact
  paths**. Additions, content changes and deletions each need a reviewed hash or path
  exemption, and **every exemption must be consumed by the candidate wheel** -- so a stale
  wheel cannot silently pass by reverting an approved transformation.

## Test boundary

The repository test suite lives in the top-level `tests/`, is collected by pytest, and is **not
an installed `jittor` subpackage**. Existing `unittest.TestCase` tests keep working under
pytest; none were rewritten merely for the move.

- Shared test utilities live in `tests/_helpers/`; test modules **do not import each other** as
  an implicit helper API.
- The structure, CPU, CUDA and NPU suites are **separate CI layers**.
- Test selection uses names and markers, **never `listdir()` positions**.
- The C++ `test.h` resource lives in `src/utils/` together with the compiler-include consumers.
- Installed-environment checks use `jittor.selftest` rather than shipping a test package.

## Tool and example boundary

Repository operations commands live in the top-level `tools/`, and teaching and runnable
samples in `examples/`. Neither tree **contains `__init__.py`**, and neither participates in
runtime package discovery. They are included in the source distribution for maintainers, but
both a directly built wheel and an sdist-derived wheel exclude them.

As a deliberate Jittor 2.0 breaking change, Stage 6 retired the published `jittor.vcompiler`
package: it compiled a private C++ extension during import, had no supported consumer in the
repository, and has **no direct replacement**; custom-operator authors should use the
maintained `compile_custom_op` and `compile_custom_ops`. The in-package `version` file was
retired as well, because project metadata and `jittor.__version__` are the version authority.

Stage 6 also retired the old LLVM alignment pass and the unused `compiler.compile_extern()`
build hook: the active compiler startup path never calls that hook, and keeping a private pass
bound to LLVM compiler internals only adds unsupported distribution payload without affecting
the runtime JIT path at all.

## Toolchain and delivery

- **Ruff** provides formatting and lint; **mypy** is introduced with an explicit, gradually
  tightened scope rather than an unmaintainable one-off exemption list.
- **pre-commit** runs fast deterministic checks, and **nox** is the canonical command entry
  point for local and CI use.
- `.github/ci-baseline.env` is the **single maintained definition** of host runners, Python
  versions, the CPU CI image, CUDA requirements, hardware runner labels, the container matrix
  and the release-validation matrix. The reusable `_ci-baseline.yml` workflow exports these
  values, and the individual workflows consume its outputs **instead of each repeating
  versions or runner labels**.
- CPU tests and packaging run in the CPU image of that baseline. CUDA and NPU jobs are
  **formal exceptions** to container uniformity: they run on labelled self-hosted hosts,
  because devices, drivers, toolkits and vendor runtimes are host capabilities. macOS and
  Windows release checks are the corresponding native-platform exceptions. These jobs still
  consume the common Python, toolkit, runner-label or platform-matrix values that apply to
  them.
- Releases use `python -m build` and PyPI trusted publishing. Jittor distributes Python plus
  JIT source resources rather than prebuilt platform extensions, so its only canonical wheel
  is `py3-none-any`. A release builds that wheel once and installs the same artifact on Linux,
  macOS and Windows. The cibuildwheel check asserts that **no platform wheels are produced**,
  rather than replacing the canonical wheel with three platform-specific ones.
- **ASV** tracks speed and memory per commit, using `--python=same`, selected revisions, a
  Jittor cache isolated from unit tests, and externally stored results and HTML reports.
- **Documentation** uses Sphinx and MyST, with Chinese as the source language, and tutorials
  managed by jupytext.

## Phase gates

Before a phase is considered complete, it must supply all of the following applicable evidence:

- a wheel comparison against the accepted baseline by **member hash plus exact path**;
  additions, content changes and deletions each need an exact, fully consumed allowlist tied to
  the phase that changes the distribution boundary;
- a cold-start import from an isolated wheel installation;
- no regression in the affected CPU matrix;
- **at least one real accelerator regression**; before a capability may be described as "NPU
  supported", NPU evidence must come first;
- `tools/check_repo_layout.sh` and the repository structural checks;
- the commands, results, known skips and unverified backends are recorded;
- a **focused commit** that carries no unrelated user changes.

Extracting the compatibility layer additionally requires that the end-to-end TRELLIS and
Gaussian Splatting paths keep their established numerical and performance results after moving
to optional integration packages.

## Consequences

During the migration **old and new forms coexist**, and some public callables will ultimately
report a more specific implementation module. In exchange: package contents become auditable,
dependency direction is visible in the filesystem, third-party integration cost becomes
optional, and correctness and performance requirements **turn from documentation into
enforceable gates**.
