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
export ASV_PYTHONPATH="$PWD/python"
```

`cache_name` must begin with `asv-`, and the resolved `JITTOR_HOME` must contain
an `asv` path component. A missing or reused cache is a hard error, not an ASV
skip. Give parallel CPU, CUDA, and NPU jobs different `JITTOR_HOME` directories
or different `cache_name` values.

ASV removes ordinary `PYTHONPATH` when it starts an existing environment.
`ASV_PYTHONPATH` is therefore the explicit source-tree path passed to benchmark
processes; the nox sessions set it to `python/` automatically. The sessions also
put the Jittor compile cache and generated ASV state outside the checkout.
`ASV_RESULTS_DIR` and `ASV_HTML_DIR` may select durable external locations for
raw JSON and the published report.

## Canonical nox runs

Install the pinned development tools and record the maintained CPU selection:

```bash
python -m pip install -r requirements/dev-tools.txt
python -m nox -s benchmark -- \
  --bench '^(operators|optimizer_step)\.'
```

On a labeled CUDA host with the pinned CUDA benchmark dependencies installed,
record the accelerator selection with:

```bash
python -m nox -s benchmark_cuda -- \
  --bench '^(tiny_llama|optimizer_step)\.'
```

Both sessions perform a real ASV pipeline: `check`, register the machine, `run`
the exact current commit with samples, `compare` it with the nearest cached
ancestor (or an explicit `ASV_COMPARE_BASE`), then `publish` the HTML report.
`ASV_COMPARE_FACTOR` controls the regression factor and must be greater than
one. A first run bootstraps comparison against itself. A run is successful only
when result JSON and `index.html` both exist.

## Ecosystem comparisons

`tests/compat/torch/test_ecosystem_parity.py` and
`test_ecosystem_speed.py` compare separate Jittor and binary-PyTorch processes.
Both processes must claim their own `torch` namespace before loading one shared
downstream package site; dependency versions and origins are part of the result
contract. CUDA runs explicitly align matmul/cuDNN TF32, and optional cuDNN
autotuning is applied to both runtimes.

Correctness tensors are copied immediately so later optimizer/gradient writes
cannot mutate a NumPy view. Timed training preallocates multiple resident input
slots and one loss-weight tensor. Jittor retains every requested gradient and
explicitly synchronizes those Vars; neither runtime performs per-gradient D2H
inside the timing window. See the
[2026-08-23 ecosystem report](../../agent/results/2026-08-23-ecosystem-parity-performance.md).

The result label must match the checkout. The nox sessions reject a dirty tree
unless the caller deliberately sets `ASV_ALLOW_DIRTY=1`, which is intended only
for local investigation and not publishable evidence.

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

## CI retention and cadence

The CPU workflow runs `nox -s benchmark` in the baseline CPU container. It
restores prior result JSON when available, records the current commit, compares
and publishes, then saves the updated result cache. Raw JSON and generated HTML
are uploaded together as a retained CI artifact.

The CUDA benchmark is intentionally not a pull-request gate. The CUDA workflow
runs `nox -s benchmark_cuda` after its real-device test gate on the weekly
schedule and on manual dispatch. It uses the labeled CUDA 12.2 RTX 4090 runner,
keeps a separate result cache, and uploads a separate CUDA JSON/HTML artifact.
This cadence makes accelerator regressions visible without consuming dedicated
hardware for every source push.

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
