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

## Environment

TorchQuantum imports its Qiskit, TorchPack, and pulse dependencies eagerly. Put
those dependencies in an isolated directory and add it to `PYTHONPATH`; do not
install TorchQuantum with dependencies into the Jittor environment because pip
can replace the Jittor torch shim with real PyTorch.

Jittor's CUDA toolchain and cache must also be explicit. Example:

```bash
export JT_ROOT=/home/zy/projects/jittor
export TQ_ROOT="$JT_ROOT/.work_tmp/torchquantum-validation/upstream"
export TQ_DEPS="$JT_ROOT/.work_tmp/torchquantum-validation/deps"
export JTCUDA=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux
export PYTHONPATH="$JT_ROOT/python:$TQ_DEPS:$TQ_ROOT"
export JITTOR_HOME="$JT_ROOT/.jittor_test_cache/torchquantum_20260712"
export PATH="$JTCUDA/bin:/home/zy/miniconda3/envs/jt311/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$JTCUDA/lib64"
export CUDA_HOME="$JTCUDA"
export nvcc_path="$JTCUDA/bin/nvcc"
export CUDA_VISIBLE_DEVICES=1
export use_cuda=1
export use_parallel_op_compiler=0
```

Run the first compilation serially:

```bash
/home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case basic --device cuda

/home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case usage --device cuda
```

Success requires `TORCH_IS_JITTOR=True` plus `BASIC_USAGE_OK` or
`USAGE_OK`. A dependency/import traceback is a failed validation, not a skip.
