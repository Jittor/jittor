# Contributing to Jittor

Jittor accepts bug fixes, features, operators, tests, documentation, and
performance work. By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.en.md) and the project's
[governance rules](GOVERNANCE.en.md).

## Before you start

- Search [existing issues](https://github.com/Jittor/jittor/issues) before filing
  a new report. For behavior changes, describe the intended contract before
  implementation.
- Keep one pull request focused on one coherent change. Avoid generated files,
  runtime caches, model weights, and unrelated formatting churn.
- Reproduce a bug before changing code. Record the smallest command, device,
  input, expected result, and actual result.
- Do not claim accelerator support from import success. Exercise the operation
  on every backend the change advertises.

## Development setup

Runtime compatibility is declared in [`pyproject.toml`](pyproject.toml). The
package supports Python 3.7 or newer; repository tooling uses pinned Python 3.11
environments and also compiles the tree with a real Python 3.7 interpreter.

```bash
git clone https://github.com/<your-account>/jittor.git
cd jittor
python -m pip install -e . -e ./compat
python -m pip install -r requirements/dev-tools.txt
python -m jittor.selftest
```

Jittor compiles C++ and CUDA code on demand. Use an isolated cache for each
checkout or concurrent job:

```bash
export JITTOR_HOME="${JITTOR_HOME:-$HOME/.cache/jittor-dev}"
export cache_name="${cache_name:-local-dev}"
```

For CUDA, set `nvcc_path` when `nvcc` is not already on `PATH`. Backend-specific
CI requirements are documented in [`agent/manuals/environment.md`](agent/manuals/environment.md).

## Repository map

```text
.
├── src/                 # C++ core: core/ type/ mem/ codegen/ ops/ runtime/ bindings/
├── backends/            # one directory per backend: cpu/ cuda/ acl/ rocm/ corex/ comm/
├── python/jittor/       # the Python runtime package (pure Python)
├── python/jittor_utils/ # bootstrap for the build utilities in jittor/build/utils/
├── compat/              # the optional `jittor-torch` distribution: torch shim, FSDP2, triton
├── adapters/            # the optional `jittor-torch-adapters` distribution
├── tests/               # repository pytest suite; not shipped in wheels
├── examples/            # runnable examples and MyST notebook sources
├── benchmarks/          # ASV regressions, Torch comparisons, and ACL measurements
├── tools/               # build, docs, lint, release and maintenance commands
├── docs/                # the single user- and contributor-facing documentation tree
├── agent/               # operational manuals and skills (manuals/ and skills/ only)
├── pyproject.toml       # authoritative package, tool, and pytest configuration
└── noxfile.py           # reproducible local and CI sessions
```

The C++ and CUDA sources are **beside** the Python package, not inside it:
`src/` is the core and each `backends/<name>/` is one backend. A source checkout
resolves them next to `python/jittor/`; a wheel installs the same trees as
`jittor/src/` and `jittor/backends/<name>/`. Those paths are loaded by path at
build time and have packaging and compiler contracts, as do
`python/jittor/contrib/math_util/src/` and
`compat/shim/cpp_extension/{include,src}/`. Read the
[repository layout decision](docs/development/repository-layout.en.md) and
[source architecture](docs/development/source-architecture.en.md) before moving
them.

## Making a change

Follow the style already used by the subsystem you edit. New Python code must
remain Python 3.7 compatible unless project metadata changes in the same
reviewed change.

### Python

- Use four spaces and descriptive names.
- Keep public APIs documented and typed where the surrounding module is typed.
- Do not hide import or installation failures with broad exception handlers.
- Add a focused regression test for every corrected behavior.

### C++ and CUDA

- Follow nearby naming, formatting, and generated-code conventions.
- Preserve CPU behavior when changing a CUDA path and vice versa.
- Treat dtype inference, broadcasting, empty tensors, non-contiguous layouts,
  gradients, and device dispatch as part of an operator's contract.
- First-time extension/JIT builds sharing a cache must run serially.

## Tests

The repository suite lives at root [`tests/`](tests/) and is collected by
pytest. Existing `unittest.TestCase` tests are valid pytest tests. Test
configuration and registered markers are in [`pyproject.toml`](pyproject.toml).

Put tests in the matching domain and name modules `test_*.py`. Apply the narrowest
backend or lifecycle marker (`structure`, `cpu`, `cuda`, `rocm`, `npu`, `mpi`,
`slow`, `network`, or `manual`). Skips and expected failures must state the exact
unsupported contract; do not turn failures into unconditional skips.

```bash
# Focused test
python -m pytest tests/nn/test_nn_capabilities.py -v

# One test case
python -m pytest \
  tests/nn/test_nn_capabilities.py::TestAttentionCapabilities::test_layout_lengths_and_cumulative_cache \
  -v

# Collection contract
python -m pytest --collect-only -q tests
```

### Nox gates

[`noxfile.py`](noxfile.py) is the canonical command surface. `python -m nox`
with no argument runs the default gate: `lint`, `format`, `typing`,
`structure`, `cpu`, `packaging`, `py37`, `py312` and `py313`.

```bash
python -m nox
python -m nox -s structure
python -m nox -s smoke        # the pull-request CPU tier
python -m nox -s cpu
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s ecosystem    # nightly: downstream libraries against real PyTorch
python -m nox -s benchmark
python -m nox -s docs
python -m nox -s docs_links
python -m nox -s tutorials
```

Between edits, [`tools/run_test_suite.py`](tools/run_test_suite.py) is cheaper
than a gate. It runs native and Torch-compatibility tests in separate processes
with separate JIT caches, because Torch mode is process-global, and reports one
combined result:

```bash
python tools/run_test_suite.py --tier core    # after an edit
python tools/run_test_suite.py --tier smoke   # before opening a pull request
python tools/run_test_suite.py                # the whole tree, on demand
```

The `structure` session checks repository layout, source and wheel contents, a
wheel built from the sdist, and the installed `jittor.selftest`. Hardware
sessions require a provisioned backend and accept pytest targets after `--`.
`docs` builds the documentation with warnings treated as errors,
`docs_links` validates documentation links, and `tutorials`
materializes MyST sources in temporary storage and executes the maintained
notebook smoke tests.

```bash
python -m nox -s cuda -- tests/nn/test_nn_capabilities.py
```

For changes spanning shared semantics, run the focused test first, then the
relevant CPU and accelerator gates. Use separate `JITTOR_HOME` or `cache_name`
values for concurrent jobs; do not run benchmarks and tests against one cache.

## Documentation

`docs/` is the only documentation tree, its source language is Chinese, and
every page belongs to exactly one section:

| Section | Contents |
| --- | --- |
| [`docs/guides/`](docs/guides/) | task guides: hardware, debugging, memory, platform support and limitations |
| [`docs/notes/`](docs/notes/) | long-lived framework mechanisms (precision, placement, async execution) |
| [`docs/compatibility/`](docs/compatibility/) | the Torch compatibility layer and its acceptance rules |
| [`docs/development/`](docs/development/) | repository layout, source architecture, test system, known-issue write-ups |
| [`docs/research/`](docs/research/) | proposals that are not yet decided |
| [`docs/results/`](docs/results/) | reproducible maintainer conclusions, each with status, commit, scope, maintainer and recheck condition |
| [`docs/releases/`](docs/releases/) | release notes |

- Update [`README.md`](README.en.md) for installation or first-use changes. It is
  the single authoritative root README, maintained as `README.md` for Chinese and
  `README.en.md` for English; do not add a third, per-language or generated README.
- Keep raw logs, caches and benchmark artifacts out of the documentation tree.
- Use relative links between documentation pages, and a
  `https://github.com/Jittor/jittor/blob/master/<path>` URL for a repository
  file outside `docs/`; both forms are checked.
- Run `python -m nox -s docs` and `python -m nox -s docs_links` after any
  documentation change, and the structure gate after moves.

## Pull requests

A pull request should include:

- the problem and intended behavior;
- the implementation boundary and compatibility impact;
- exact tests run and their backend/environment;
- known limitations, skips, or deferred follow-up;
- documentation and release-note changes when users are affected.

Before requesting review, inspect the staged diff, avoid `git add -A`, and ensure
that only files belonging to the change are staged. Reviewers prioritize
correctness, public compatibility, backend behavior, packaging, and regression
coverage over cosmetic consistency.
