# Test System

- Status: Accepted
- Last reviewed: 2026-08-22
- Baseline: `866914d4`
- Owner: test infrastructure maintainers
- Review when: collection roots, process-mode ownership, markers, OpInfo
  contracts, or backend gates change

Jittor uses pytest as the repository test runner while retaining compatible
`unittest.TestCase` tests. The suite lives under root `tests/` and is not part of
the installed package. [`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml)
is the authoritative collection and marker configuration;
[`noxfile.py`](https://github.com/Jittor/jittor/blob/master/noxfile.py)
is the reproducible command surface.

## Goals

- Compare forward behavior with an independent reference.
- Verify analytical gradients numerically and across devices.
- Keep structure, CPU, and hardware requirements explicit.
- Preserve useful legacy edge cases while replacing cross-test imports and
  ad-hoc discovery with shared helpers and registries.
- Make unsupported behavior visible through specific skips or strict expected
  failures.
- Keep collection free of compilation, downloads, and hardware side effects.

## Layout

```text
tests/
├── _fixtures/            # versioned test data
├── _helpers/             # explicit shared test utilities
├── opinfo/               # operator metadata, samples, references, skip policy
├── ops/                  # generic OpInfo forward and gradient batteries
├── backends/             # CPU/CUDA/ROCm/NPU and cross-device parity
├── compiler/             # JIT/compiler behavior and kernel traps
├── core/                 # tensor, dtype, graph, and autograd contracts
├── nn/                   # neural-network operations and modules
├── optim/                # optimizer contracts
├── distributed/          # MPI and distributed behavior
├── compat/               # compatibility surfaces and import behavior
├── integration/          # notebooks and cross-component workflows
├── structure/            # repository, packaging, and static contracts
├── models/               # maintained model-level tests
└── system/               # process/environment integration tests
```

Test modules may import `tests/_helpers` and `tests/opinfo` through the pytest
Python path configured for the suite. They must not import another test module as
an implicit helper API. Shared utilities need their own focused tests when they
contain nontrivial comparison or device logic.

## Process-mode isolation

Torch compatibility installation is process-global and changes public methods,
dtype promotion, reduction defaults, and lazy execution. Native and Torch-facing
tests therefore run in separate pytest processes.
`tests/_helpers/process_modes.py` owns the `TORCH_MODE_PATHS` list;
`tests/conftest.py` applies it by ignoring those paths during a broad native
collection and activating Torch mode when one of them is selected explicitly.
The shared OpInfo and device-parity suites use Torch-facing signatures and
therefore belong to the Torch process.

`tools/run_test_suite.py` is the complete-suite entry point. It runs native and
Torch sessions with separate state, caches, and mode variables, then reports a
combined result. A direct `python -m pytest tests` command is intentionally only
the native session and must not be reported as full-suite coverage. A test that
asserts `result_type`, Torch cast aliases, typed-tensor names, or Torch-specific
defaults belongs to the Torch session even if the file also carries a low-level
contract and remains under `tests/core/`.

An in-process independent PyTorch oracle requires `REAL_TORCH_SITE` to point to
a site-packages directory containing PyTorch's binary `_C` extension. Pytest
preloads that implementation before Jittor. Without this explicit oracle, tests
skip optional PyTorch comparisons even when a deployed Jittor-backed `torch`
stub is discoverable; the stub must never be accepted as an independent
reference.

## Three layers of operator evidence

### 1. Independent forward reference

Each `OpInfo` describes the callable, sample builder, dtypes, autograd support,
tolerances, and targeted skip/xfail policy. `tests/ops/test_ops.py` expands the
database across requested devices and compares outputs with independent NumPy or
mathematical references.

The reference must not call the Jittor operation under test. Sample builders
cover meaningful shapes, axes, broadcasting, optional arguments, and error-prone
dtypes rather than generating volume without semantic variety.

### 2. Numerical gradients

Differentiable OpInfo entries run float64 CPU `gradcheck`; operations that
support second-order differentiation also run `gradgradcheck`. Marking
`supports_autograd=False` or `supports_gradgrad=False` is a contract decision,
not a way to make a failing test green, and requires a reason in the definition
or known-issues ledger.

Numerical checks prove derivative formulas on CPU. They do not prove that an
accelerator executes the same backward kernel correctly.

### 3. Device parity

[`tests/backends/parity/test_device_parity.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/parity/test_device_parity.py)
runs identical inputs and cotangents on CPU and the available accelerator,
comparing forward outputs and gradients with both global and per-element error
metrics. This layer catches device-specific compile failures, dropped gradient
contributions, and silent kernel divergence.

CPU is a practical parity oracle only after its behavior is independently pinned
by the forward and numerical-gradient layers. Backend environment failures are
reported separately from framework defects.

## Test categories and markers

Markers are registered in
[`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml):

| Marker | Contract |
| --- | --- |
| `structure` | no device execution; layout, packaging, and static checks |
| `cpu` | maintained CPU behavior |
| `cuda` | requires an NVIDIA CUDA environment |
| `rocm` | requires an AMD ROCm environment |
| `npu` | requires an Ascend CANN environment |
| `mpi` | requires an MPI launcher or multiple processes |
| `slow` | excluded from the fast pull-request gate |
| `network` | requires external network access |
| `manual` | selected explicitly; never part of automatic default runs |

Apply the narrowest applicable marker. Hardware tests probe a real operation and
must not silently pass on CPU. Network and manual tests explain their external
requirements in the module docstring.

## Skips and known failures

- A skip represents an unavailable prerequisite or an intentionally unsupported
  contract. Its reason identifies the exact prerequisite or limitation.
- An expected failure represents a reproduced framework defect. Pytest uses
  strict xfail behavior, so a fix produces an XPASS and forces ledger cleanup.
- Do not catch arbitrary exceptions around a test body and convert them to a
  skip. Probe optional environments narrowly before execution.
- Every persistent expected failure is listed in the
  [known-issues ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)
  with an owner and an
  exit condition.

## Commands

```bash
# Complete two-process suite, native-only collection, or a focused module
python tools/run_test_suite.py
python -m pytest --collect-only -q tests
python -m pytest -v tests/ops/test_ops.py

# Select one operation or backend marker
JITTOR_TEST_DEVICES=cpu python -m pytest tests/ops/test_ops.py -k exp
python -m pytest -m structure tests/structure

# Reproducible gates
python -m nox -s structure
python -m nox -s cpu
python -m nox -s optional
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
```

The nox sessions create isolated state and caches. Direct concurrent runs must
also use distinct `JITTOR_HOME` or `cache_name` values. The first build of a new
JIT operation or extension should run serially.

The maintained CUDA session runs the complete CUDA backend directory, dtype
coverage, CPU/CUDA device parity, Torch TF32 controls, and the strict CUDA
OpInfo suite. Its accepted real-device baseline is recorded in the
[complete CUDA suite report](../../agent/results/2026-08-22-cuda-test-suite.md).

The `optional` session is a fail-closed, offline CUDA gate for pre-provisioned
TorchMetrics, mmcv-lite/MMEngine, PEFT, Safetensors, TensorDict, and the deployed
FlashAttention adapter. It probes every package before pytest, enables the
Jittor Torch shim explicitly, and treats PEFT import failures as errors instead
of optional skips. When `JITTOR_FLASH_ATTN_JITTOR_SRC` names an official
FlashAttention checkout, the session uses two phases: the normal optional tests
run with the deployed math adapter, then a native-required phase runs fused
fp16 forward, dense/varlen/packed backward, dropout RNG replay, GQA, and
float32 opt-in tests. The native phase defaults to head dimension 32 and fp16;
the FlashAttention capability environment variables extend that base set rather
than replacing it. The native phase cannot be satisfied by fallback.

## Adding coverage

Use an OpInfo definition when one operation can share the standard sample,
reference, dtype, gradient, and device-parity machinery. Use a focused test when
the contract concerns state, mutation, serialization, error behavior, module
lifecycle, import order, distributed coordination, or a specific regression not
expressible through OpInfo.

A new operation normally needs:

1. independent forward samples and reference;
2. dtype, shape, axis, empty, broadcast, and non-contiguous cases as applicable;
3. gradient and grad-gradient declarations backed by tests;
4. device parity for every advertised accelerator;
5. explicit error-contract tests;
6. an OpInfo report entry or focused test name that makes missing coverage
   discoverable.

## Acceptance

A test-system change is complete when collection succeeds without device side
effects, focused self-tests cover new harness logic, the structure gate passes,
at least one mandatory CPU case executes, and each hardware result distinguishes
pass, framework failure, and unavailable environment. Counts alone are not an
acceptance criterion; the evidence must exercise the claimed semantics.
