# Performance benchmarking

Jittor uses ASV 0.6.6 to retain commit-indexed timing and memory results. The
suite deliberately uses the current Python environment (`--python=same`): a
fresh environment per historical commit would mostly measure Jittor's compile
and installation cost rather than steady-state runtime performance.

## Cache isolation

ASV and the unit-test suite must never compile into the same Jittor cache. Every
benchmark setup validates both variables before importing Jittor:

```bash
export JITTOR_HOME="${JITTOR_ASV_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/jittor-asv}"
export cache_name="asv-local"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

`cache_name` must begin with `asv-`, and the resolved `JITTOR_HOME` must contain
an `asv` path component. A missing or reused cache is a hard error, not an ASV
skip. Give parallel CPU, CUDA, and NPU jobs different `JITTOR_HOME` directories
or different `cache_name` values.

Generated ASV state defaults to `.asv/`. CI and nox should set
`JITTOR_ASV_HOME`/`JITTOR_HOME` to job-local storage outside the checkout; raw
results intended for publication should be uploaded as artifacts.

## Validate and smoke-test

Install the performance dependencies and check benchmark signatures without
building an ASV environment:

```bash
python -m pip install -e '.[perf]'
asv check --python=same
asv run --python=same --quick --dry-run \
  --bench 'operators.OperatorBenchmarks.time_operator'
```

The CPU parameters import and execute without CUDA. CUDA parameters raise an
explicit ASV skip when Jittor or real PyTorch cannot execute on CUDA. Real
PyTorch is an optional oracle and must be installed separately for the target
platform; an absent or shim-resolved `torch` is reported as skipped, never as a
zero-duration result. At least one mandatory Jittor CPU case must execute in CI,
so a run in which every parameter skips is not acceptable.

## Record selected revisions

Existing environments cannot ask ASV to build an arbitrary history. Run the
suite only from a checkout that matches the result label, and record that exact
commit:

```bash
commit=$(git rev-parse HEAD)
asv run --python=same --set-commit-hash "$commit"
```

For a comparison, create one dedicated benchmark worktree, switch that worktree
only to the two or three reviewed revisions, and run the command above after
each switch. Keep the same compiler, accelerator, dependency versions, cache
warmup policy, and ASV results directory. Do not run `ALL` or an unbounded
revision range. Use `asv compare <base> <candidate>` after both selected commits
have results.

## Initial benchmark set

| ASV module | Coverage | Parameters |
| --- | --- | --- |
| `operators` | matmul, softmax, LayerNorm, GELU | Jittor/optional torch; CPU/CUDA |
| `tiny_llama` | LlamaModel forward and forward+backward | Jittor/optional torch; CUDA |
| `optimizer_step` | gradient-ready SGD and AdamW step scaling | 32/128/512 tensors; CPU/CUDA |

Tiny Llama retains the established 2-layer, hidden-size 256, intermediate-size
768, 8 attention head/4 KV head, batch 2, sequence 128 configuration. Its setup
checks the output and every requested gradient for finite, nonzero values.
Optimizer setup fixes the total element count at 262,144, so the tensor-count
axis exposes per-tensor graph and launch overhead rather than larger workloads.
CUDA comparisons use float32 with TF32 enabled on both backends. Operator
benchmarks use inference/no-grad semantics; Tiny Llama enables gradients only
for its forward+backward parameter.

Every memory result is an ASV `track_*` benchmark with unit `bytes`. On CUDA,
`track_working_set_bytes` is the backend allocator's synchronized live working
set after one operation/model/step. On CPU, it is the process peak RSS reported
by the operating system. These are stable regression signals within one backend
and machine; they are not NVML peaks and should not be compared across devices.
