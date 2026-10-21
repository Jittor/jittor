---
name: torchquantum-readme-validation
description: Validate TorchQuantum README Basic Usage and Usage through `import jittor as torch`, including module identity, CUDA execution, measurement, and backward checks.
---

# TorchQuantum README validation

Use `run_readme_examples.py` against a fixed TorchQuantum checkout. The harness
keeps the operations from the upstream README and adds only:

- `import jittor as torch` before TorchQuantum imports;
- a selectable CPU/CUDA device for the Basic Usage comment's GPU path;
- a two-sample input that actually invokes the class defined by Usage;
- assertions and explicit success markers.

## Download upstream

Download the official repository from the network and pin the audited commit:

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"   # from the repository root
mkdir -p "$JITTOR_LAB_ROOT/torchquantum-validation"
git clone --depth 1 https://github.com/mit-han-lab/torchquantum.git \
  "$JITTOR_LAB_ROOT/torchquantum-validation/upstream"
git -C "$JITTOR_LAB_ROOT/torchquantum-validation/upstream" \
  checkout 8dc3255c51477dd4c28892049571df032c77e2ff
```

Confirm `git config --get remote.origin.url` is the official URL and compare
the local SHA with `git ls-remote ... HEAD refs/heads/main` before validation.

## Environment

TorchQuantum imports its Qiskit, TorchPack, and pulse dependencies eagerly. Put
those dependencies in an isolated directory and add it to `PYTHONPATH`; do not
install TorchQuantum with dependencies into the Jittor environment because pip
can replace the Jittor torch shim with real PyTorch.

Jittor's CUDA toolchain and cache must also be explicit. Example (`<cuda-root>` is a
CUDA toolkit root exposing `bin/nvcc` and `lib64`, for example the one Jittor's
auto-installer unpacks under `$JITTOR_HOME`; `<jittor-python>` is the interpreter that
runs Jittor):

```bash
export JT_ROOT="$PWD"                       # the Jittor repository root
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
export TQ_ROOT="$JITTOR_LAB_ROOT/torchquantum-validation/upstream"
export TQ_DEPS="$JITTOR_LAB_ROOT/torchquantum-validation/deps"
export JTCUDA=<cuda-root>
export PYTHONPATH="$JT_ROOT/python:$TQ_DEPS:$TQ_ROOT"
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/torchquantum-readme-validation/torchquantum_20260712"
export PATH="$JTCUDA/bin:$PATH"
export LD_LIBRARY_PATH="$JTCUDA/lib64"
export CUDA_HOME="$JTCUDA"
export nvcc_path="$JTCUDA/bin/nvcc"
export CUDA_VISIBLE_DEVICES=<gpu>
export use_cuda=1
export use_parallel_op_compiler=0
```

Run the first compilation serially:

```bash
<jittor-python> \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case basic --device cuda

<jittor-python> \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case usage --device cuda
```

Success requires `TORCH_IS_JITTOR=True` plus `BASIC_USAGE_OK` or
`USAGE_OK`. A dependency/import traceback is a failed validation, not a skip.
