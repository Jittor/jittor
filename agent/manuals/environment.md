# Jittor Development Environment

- Status: Maintained
- Last reviewed: 2026-08-12
- Baseline: `dccca5b2`
- Owner: repository and CI maintainers
- Review when: supported Python, nox sessions, backend prerequisites, or cache
  behavior changes

This document describes portable environment contracts. Host addresses, personal
paths, device allocation, credentials, and one-off command output belong in
local run configuration or a dated result report, not here.

## Workspace roots

Run commands from the repository root. Keep experiments and mutable state in a
sibling lab directory:

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
mkdir -p "$JITTOR_LAB_ROOT/worktrees" "$JITTOR_LAB_ROOT/_state"
```

Use these boundaries:

- source, tests, durable docs, small fixtures: the Git checkout;
- parallel worktrees: `$JITTOR_LAB_ROOT/worktrees/`;
- downstream checkouts and experiments: `$JITTOR_LAB_ROOT/<topic>/`;
- caches, virtual homes, raw logs, generated builds, and temporary files:
  `$JITTOR_LAB_ROOT/_state/<topic>/<run>/`.

Do not write runtime state into the repository. A result report may describe an
external artifact path, but the result must remain understandable when that
artifact is unavailable.

## Isolated run template

```bash
run_root="$JITTOR_LAB_ROOT/_state/<topic>/<run>"
mkdir -p "$run_root"/{home,jittor-home,tmp,xdg-cache,logs}

export HOME="$run_root/home"
export JITTOR_HOME="$run_root/jittor-home"
export TMPDIR="$run_root/tmp"
export XDG_CACHE_HOME="$run_root/xdg-cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export cache_name="<topic>-<run>"
```

Each concurrent run needs a distinct `JITTOR_HOME` or `cache_name`. A new JIT
operation or extension compiles serially once before parallel validation.
Benchmarks and tests never share a cache.

[`noxfile.py`](../../noxfile.py) creates this isolation automatically under
`$JITTOR_LAB_ROOT/_state/nox/`.

## Python and tools

- Package runtime: Python 3.7 through Python 3.12, as declared by
  [`pyproject.toml`](../../pyproject.toml). Python 3.7 remains the lower syntax
  boundary; Python 3.12 is the maintained upper compatibility gate.
- Maintainer tooling: Python 3.11 with pinned versions from
  [`requirements/dev-tools.txt`](../../requirements/dev-tools.txt).
- Syntax compatibility: the `py37` nox session requires a real Python 3.7
  interpreter and compiles every repository Python file.
- Python 3.12 compatibility: the `py312` nox session requires a real Python
  3.12 interpreter, treats `SyntaxWarning` as an error, builds a wheel, installs
  that wheel outside the source tree, and runs `jittor.selftest` on CPU.

```bash
python -m pip install -e .
python -m pip install -r requirements/dev-tools.txt
python -m jittor.selftest
python -m nox -s structure
python -m nox -s py312
```

## Backend gates

### CPU

Use an empty `nvcc_path` when a test must prove CPU-only behavior. The maintained
gate sets `JITTOR_TEST_DEVICES=cpu` and runs a real tensor operation before tests.

```bash
python -m nox -s cpu
```

### CUDA

The CUDA session requires `nvcc_path` or `nvcc` on `PATH`, a working driver, and
a pre-provisioned Python environment. Set `JITTOR_CI_PYTHON` when the hardware
environment uses a different interpreter.

```bash
export nvcc_path="$(command -v nvcc)"
export JITTOR_CI_PYTHON="${JITTOR_CI_PYTHON:-$(command -v python)}"
python -m nox -s cuda
```

### Optional compatibility packages

The optional compatibility session uses the same pre-provisioned CUDA Python
environment and requires TorchMetrics, mmcv-lite, MMEngine, PEFT, Safetensors,
TensorDict, and the deployed FlashAttention adapter. It runs offline and fails
before pytest when a required package is not discoverable.

```bash
export nvcc_path="$(command -v nvcc)"
export JITTOR_CI_PYTHON="${JITTOR_CI_PYTHON:-$(command -v python)}"
python -m nox -s optional
```

To extend the same gate to the native fused FlashAttention backend, point the
session at an official source checkout. The session first runs the deployed
adapter tests with math fallback enabled, then runs a separate native-required
phase so the two contracts cannot mask or contradict each other. The native
phase defaults to head dimension 32 and fp16; set
`JITTOR_FLASH_ATTN_HEAD_DIMS` or `JITTOR_FLASH_ATTN_DTYPES` to expand it. These
values are unioned with the base `32/fp16` capability rather than replacing it.
It covers fused forward, dense/varlen/packed backward, dropout RNG replay, GQA,
and float32 opt-in cast. A native build or load failure cannot fall back to math
attention in that phase.

```bash
export JITTOR_FLASH_ATTN_JITTOR_SRC=/path/to/flash-attention
python -m nox -s optional
```

### Ascend NPU

The NPU session requires `CANN_SET_ENV` to name the vendor `set_env.sh`, plus a
Python environment containing the test requirements.

```bash
export CANN_SET_ENV=/path/to/cann/set_env.sh
export JITTOR_CI_PYTHON="${JITTOR_CI_PYTHON:-$(command -v python)}"
python -m nox -s npu
```

### ROCm and MPI

ROCm requires a working `rocminfo`; MPI requires `mpirun`. Both sessions use
`JITTOR_CI_PYTHON` when set.

```bash
python -m nox -s rocm
python -m nox -s mpi
```

An unavailable backend is an environment result. A reproduced kernel, dtype, or
gradient mismatch on a working backend is a framework result. Reports must not
conflate the two.

## Independent Torch oracle

Tests that compare with real Torch must prove that `torch` did not resolve to
Jittor's deployed shim. Use a clean environment, inspect `torch.__file__`, and
record the Torch version. If the oracle is absent or shim-resolved, report the
comparison as unavailable rather than using Jittor as its own reference.

## Result metadata

A durable verification report records:

- exact Git commit and dirty-state status;
- Python, compiler, dependency, backend, and device versions;
- relevant environment flags without secrets or host-specific addresses;
- cold/warm cache state and isolation choice;
- exact commands, pass/fail/skip counts, and known limitations;
- locations and hashes for unversioned artifacts when needed.

See [collaboration rules](collaboration.md) for report ownership and
[performance benchmarking](../../docs/performance/benchmarking.md) for timing
methodology.
