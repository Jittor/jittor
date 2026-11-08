# Test system

[Section contents](../index.en.md) · [中文](test-system.md)

Start here to choose a test tier and run it in the correct process mode. pytest runs the repository tests; nox provides reproducible environments. Tests are not installed in runtime wheels.

- Writing assertions: [numerical and coverage evidence](test-evidence.en.md).
- Reading a failed or skipped run: [isolation and completion](test-isolation.en.md).
- Investigating a slow run: [JIT cache](jit-cache.en.md).

Prerequisite: a configured development checkout; see [contributing](../contributing.en.md). The times below are historical measurements, not runtime guarantees.

## Three tiers: core / smoke / full

A full run takes a dozen-plus minutes at the start, which is too expensive as an answer to "did
what I just changed break", so there are three tiers answering three different questions:

| Tier | Question | Selection | Measured |
| --- | --- | --- | --- |
| `core` | did a single edit break something immediately | a **whitelist** in `tiers.CORE_FILES`, one file per basic surface | 44 s (two-process mode, serial) |
| `smoke` | how long should a PR wait | the whole tree **minus** `tiers.SLOW_FILES` | about 390 s (4 workers) |
| `full` | what state is this tree actually in | the whole tree | a dozen-plus minutes and up |

Only `full` is a statement about this tree; the other two are statements about time -- which is
also why `core` uses a whitelist and `smoke` a blacklist: a file missing from `smoke` is **a hole
in coverage**, so the default must be inclusion; a file missing from `core` is only **found a
little later**, since the other two tiers still go red, so a selection is acceptable.

`core` is deliberately serial: it is small enough that workers buy little, and this way it also
runs in checkouts without pytest-xdist installed. Its budget is checked **arithmetically** by
`tests/structure/test_gate_tiers.py` (summing each file's measured seconds and comparing against
the budget), not by asserting wall-clock time -- the latter fails on a busy machine in a way that
looks exactly like a real regression.

```bash
python tools/run_test_suite.py --tier core    # After an edit
python tools/run_test_suite.py --tier smoke   # Before a PR
python tools/run_test_suite.py                # Full run when needed
```

Adding a file to `core` spends everyone's time on every edit; the entry must state **the measured
seconds** and **which basic surface it guards**, and the structural test checks both.

## Commands

```bash
# Tier selection, native collection and a focused Torch-mode module
python tools/run_test_suite.py --tier core
python tools/run_test_suite.py
python -m pytest --collect-only -q tests
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -v tests/ops/test_ops.py

# Select an operator or marker
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cpu PYTHONPATH=python \
    python -m pytest tests/ops/test_ops.py -k exp
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -m structure tests/structure

# Compat tests through their canonical package path
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
    python -m pytest python/jittor/compat/tests/torch

# Reproducible gates
python -m nox -s structure
python -m nox -s cpu
python -m nox -s optional
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s nccl
```

nox sessions establish isolated state and caches. **Runs started concurrently by hand must also
use different `JITTOR_HOME` or `cache_name` values; the first build of a new JIT operation or
extension should run serially.**

Why the compatibility command takes a detour: `compat/` carries its own `__init__.py`, so
`pytest compat/tests/torch` directly makes pytest import it as a **top-level** package `compat`,
while `compat/_aliases.py`'s `from .._runtime import ...` requires it to be `jittor.compat`, so
every case fails with
`ImportError: attempted relative import beyond top-level package` -- collection looks fine (1888
collected), and the errors all land in setup, which is easily misread as "cannot collect".
`python/jittor/compat` is a symlink to `compat/`, and entering through it resolves the package
name to `jittor.compat.tests.torch`, so the relative import holds. The gate does not have this
problem because `_run_pytest_once` first runs `pip install -e compat/`, installing `jittor-compat`
as a real package.

## Directory layout

The main repository has two test roots sharing one pytest policy:

```text
tests/                    # native tests
├── conftest.py           # registers _helpers.pytest_policy
├── _fixtures/            # versioned fixtures
├── _helpers/             # shared process, gate, child-process and capability helpers
├── opinfo/               # operator metadata, samples, references and skip policy
├── ops/                  # OpInfo forward and gradient checks
├── autograd/ core/ type/ mem/ runtime/ bindings/   # native graph, dtype, memory, runtime and bindings
├── codegen/              # code generation, passes and tuners
├── build/                # Python build and compilation tooling
├── nn/ optim/ init/ linalg/ distributions/ serialization/ data/ contrib/ models/
├── backends/             # mirrors backend domains; parity/ compares devices
├── distributed/          # distributed behaviour in native process mode
├── debug/ utils/ tools/  # diagnostics and tools
├── integration/          # examples, notebooks and component workflows
└── structure/            # repository, packaging and static contracts
compat/tests/             # optional jittor-compat tests
├── torch/                # Torch-mode tensor behaviour
├── structure/            # name ownership and namespace publication
├── fsdp2/                # FSDP2
└── triton/               # Triton bridge
```

Both roots load `tests/_helpers/pytest_policy.py` through a thin conftest. The helper is a
monorepo development dependency: **neither the Jittor wheel nor the compat wheel installs any
test package**, so running `compat/tests` requires a full checkout that includes
`tests/_helpers`; a standalone adapter sdist can still run its own host tests when that
directory is absent.

Structural contracts are placed under `tests/structure/<source-domain>` according to their
source domain, while cross-domain repository governance stays at the `tests/structure` root;
structural tests owned by compat live in `compat/tests/structure`. The ACL dtype recorder
(`tests/structure/backends/acl/test_acl_dtype_preservation.py`) is a CPU mock contract with an
explicit native-process exception, and **does not inherit** the fallback fixture and `npu` marker
of real ACL hardware tests. `tests/core/test_setitem.py` is deliberately kept at that path.

**Process mode belongs to each file, not to the directory it sits in.** The authoritative
selection is in `tests/_helpers/process_modes.py`, and the full runner derives the native and
Torch selection sets from it through `tests/_helpers/gate_scope.py`; there is no argv-activated
mechanism. Hardware markers are derived from the path: a path containing `acl` gets `npu`, one
containing `distributed`, `comm` or `fsdp2` gets `mpi`, and anything under `structure` gets
`structure`.

`nox -s full` or `tools/run_test_suite.py` runs the two-process full suite; `nox -s structure`
includes each structural root (downstream-specific tests live in ecosystem repositories) and runs the
native CPU recorder separately. To select cases directly, use physical paths.

**Determinism of generated inputs survives layout moves.** `tests/_helpers/layout_seed_paths.json`
records the old paths from before the move; `layout_seed.seed_nodeid` swaps the new path back to
the old prefix (keeping the class, method and parametrization suffix) only before the input
generator hashes the key, so a moved case generates the same inputs as before the move. New
files and new methods use their new canonical paths; a new class reusing an old method name, and
a new explicit method whose name happens to match an old generator prefix, both count as new
cases. pytest nodeids, logs, selectors, failure reports and coverage are **never** rewritten by
this mapping.

Test modules may import `tests/_helpers` and `tests/opinfo` through the pytest Python path
configured for the suite. **Another test module must not be imported as an implicit helper API.**
A shared utility containing non-trivial comparison or device logic needs its own targeted test.

## Process-mode isolation

Installing Torch compatibility is **process-level** and changes public methods, dtype promotion,
reduction defaults and lazy execution. Therefore **native tests and Torch-facing tests run in
different pytest processes**.

`tests/_helpers/process_modes.py` owns the `TORCH_MODE_PATHS` list; `tests/conftest.py` ignores
those paths during broad native collection and rejects an explicit selection in the wrong mode; the caller must set `JITTOR_TORCH_SHIM=1` explicitly. The shared OpInfo and device-parity suites use Torch-facing signatures and therefore
belong to the Torch process.

`tools/run_test_suite.py` is **the entry point for the complete suite**: it runs the native and
Torch sessions with their own separate state, caches and mode variables, then aggregates the
results. Running `python -m pytest tests` directly **is deliberately the native session only**
and must not be reported as full-suite coverage.

A test that asserts `result_type`, Torch cast aliasing, typed tensor names or Torch-specific
defaults belongs to the Torch session -- even if the same file also carries lower-level
contracts and lives under `tests/core/`.

An in-process independent PyTorch oracle requires `REAL_TORCH_SITE` to point at a site-packages
directory containing PyTorch's binary `_C` extension, which pytest preloads before Jittor.
**Without this explicit oracle, tests skip the optional PyTorch comparison**, even when a
deployed Jittor-backed `torch` stub can be found -- **that stub must never be treated as an
independent reference**.

## CI support matrix

Workflow status is **declared explicitly**, because "a nox session can run" does not mean CI has
the required hardware or dependencies. "Manual" means a maintainer must run the specified
fail-closed session on a configured machine before the interface may be claimed as verified --
**it must not be described as an automatic check**.

| Session | CI status | Runner and trigger |
| --- | --- | --- |
| `cuda` | automatic | Pushes and the weekly schedule use the declared RTX 4090 / CUDA 12.2 runner. Maintainers may add the `ci:cuda` label to a PR; the label, reopening and subsequent synchronize events all run the same gate. |
| `optional` | manual | No CUDA runner with complete dependencies is declared. Run `python -m nox -s optional` in the preset environment described below. |
| `rocm` | manual | No AMD/ROCm runner is declared. Run `python -m nox -s rocm` on a real supported AMD GPU. |
| `mpi` | manual | No multi-process MPI runner is declared. Run `python -m nox -s mpi` in an environment with a working launcher and compiler wrappers. |
| `nccl` | manual | The declared CUDA runner guarantees only one RTX 4090, while this gate needs two visible GPUs. Run `python -m nox -s nccl` on a multi-GPU host. |

**This matrix describes current scheduling, not the degree of backend support.** Manual results
must record the commit tested, the toolchain, the device topology, the command and the pytest
result; **an unavailable runner is not a pass.**

The maintained CUDA session runs the complete CUDA backend directory, dtype coverage, CPU/CUDA
device parity, the Torch TF32 controls and the strict CUDA OpInfo suite.

The `nccl` session needs two visible NVIDIA GPUs by default. It first warms an isolated JIT cache
serially for each rank, then uses `jittor.distributed.launch` to verify real flat FSDP2 parameter
sharding, NCCL all-gather and reduce-scatter, and sharded optimizer updates against independent
NumPy results. Setting `JITTOR_NCCL_WORLD_SIZE` and exposing at least that many devices runs more
than two ranks.

The `optional` session is a fail-closed offline CUDA gate against preset TorchMetrics,
mmcv-lite/MMEngine, PEFT, Safetensors, TensorDict and the deployed FlashAttention adapter. It
probes each package before pytest, enables the Jittor Torch shim explicitly, and treats **a PEFT
import failure as an error rather than an optional skip**. When `JITTOR_FLASH_ATTN_JITTOR_SRC`
points at the official FlashAttention checkout, the session runs in two phases: the ordinary
optional tests run with the deployed math adapter, then a phase **requiring the native
implementation** runs fused fp16 forward, dense/varlen/packed backward, dropout RNG replay, GQA
and the float32 opt-in tests. The native phase defaults to head dimension 32 and fp16; the
FlashAttention capability environment variables **extend** this base set rather than replacing
it. **The native phase cannot be satisfied by fallback.**

## Goals

- Compare forward behaviour against **independent references**;
- verify analytic gradients numerically, and do so across devices;
- keep structure, CPU and hardware requirements explicit;
- keep valuable historical boundary cases while replacing cross-test imports and ad-hoc
  discovery with shared helpers and registries;
- make unsupported behaviour visible through **specific skips or strict expected failures**;
- keep the collection phase free of compilation, downloads and hardware side effects.

## Acceptance

A test-system change is complete when collection has no device side effects, new harness logic
has its own targeted tests, the structural gates pass, at least one mandatory CPU case really
executes, and every hardware result distinguishes pass, framework failure and environment
unavailability.

**Quantity alone is not an acceptance criterion** -- the evidence must really exercise the
claimed semantics.
