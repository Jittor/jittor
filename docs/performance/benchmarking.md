# Performance benchmarking

Jittor uses ASV 0.6.6 to retain commit-indexed timing and memory results. The
suite deliberately uses the current Python environment (`--python=same`): a
fresh environment per historical commit would mostly measure Jittor's compile
and installation cost rather than steady-state runtime performance.

## Reliable local timing

Use `jt.benchmark` for local microbenchmarks instead of placing
`perf_counter()` directly around a lazy operation:

```python
import jittor as jt

inputs = [jt.randn(1024, 1024) for _ in range(4)]
result = jt.benchmark(
    lambda value: (value * value).sum(),
    inputs,
    warmup=2,
    repeat=10,
)
print(result.median, result.samples)  # seconds
```

The input pool is snapshotted and materialized before timing, then selected in
round-robin order. Warmup is mandatory and excluded from the returned samples,
so first-use compilation is outside the timed region. The callable must return
every output that belongs to the measured work; nested tuples, lists, and
dictionaries are supported. Each round retains all returned Vars and performs
one target-specific device synchronization before stopping the clock.

These rules avoid three common false results:

- **CSE:** repeatedly building the same expression while an earlier lazy output
  is still live can reuse that graph. `jt.benchmark` materializes and releases
  one round before constructing the next; use multiple resident pool entries
  when the operation mutates inputs.
- **Dead-code elimination:** an unreferenced lazy output can be discarded
  without executing. The API rejects calls that return no `jt.Var` and retains
  every returned Var through synchronization.
- **Unmaterialized work:** timing only Python graph submission measures neither
  CPU execution nor asynchronous device completion. The per-round target sync
  is included in every sample.

The samples include Python graph construction plus execution. Use the ASV suite
below for retained cross-commit results, and keep correctness checks separate
from timing.

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
Both processes must claim their own `torch` namespace before loading the
downstream libraries. They share one package site only when their CPython ABIs
are compatible, normally the same major and minor version: a site directory
carries ABI-tagged extension modules, so a 3.12 reference cannot import a 3.11
site and otherwise fails on the first compiled dependency with an unrelated
error. When the ABIs differ,
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` supplies the independent oracle site
and each side imports its own copy. Shared sites must report equal versions and
origins; separate sites must report equal versions, while their origins are
expected to differ. CUDA runs explicitly align matmul/cuDNN TF32, and optional
cuDNN autotuning is applied to both runtimes.

Correctness tensors are copied immediately so later optimizer/gradient writes
cannot mutate a NumPy view. Timed training preallocates multiple resident input
slots and one loss-weight tensor. Jittor retains every requested gradient and
explicitly synchronizes those Vars; neither runtime performs per-gradient D2H
inside the timing window. See the
[2026-08-23 ecosystem report](../../agent/results/2026-08-23-ecosystem-parity-performance.md).

Masked SDPA performance must preserve fully-masked-row semantics. On CUDA, the
maintained attention path delegates explicit masks to the softmax kernel's
`zero_all_neg_inf` mode instead of building a separate row-valid reduction and
two ternary graphs. Pure causal attention does not need that mode because every
row contains its diagonal element. Profile comparisons must check graph rows as
well as wall time; see the
[CUDA masked SDPA report](../../agent/results/2026-08-23-cuda-masked-sdpa.md).

The result label must match the checkout. The nox sessions reject a dirty tree
unless the caller deliberately sets `ASV_ALLOW_DIRTY=1`, which is intended only
for local investigation and not publishable evidence.

The CPU parameters import and execute without CUDA. CUDA parameters raise an
explicit ASV skip when Jittor or real PyTorch cannot execute on CUDA. Real
PyTorch is an optional oracle and must be installed separately for the target
platform; an absent or shim-resolved `torch` is reported as skipped, never as a
zero-duration result. At least one mandatory Jittor CPU case must execute in CI,
so a run in which every parameter skips is not acceptable.

## CPU thread affinity

Pin threads for any CPU comparison. Neither Jittor nor PyTorch sets an OpenMP
affinity by default, and on a many-core host the scheduler migrates the threads
between measurements: the same Jittor ViT step measured
`0.5555`~`0.7074s` over six runs -- two distinct clusters `25%` apart, not a
spread. Setting

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

collapses that to `0.5165`~`0.5195s` across four runs, and it is worth `10`-`30%`
to *both* runtimes, so an unpinned comparison can invert a verdict: a diffusion
UNet reads `0.90x` unpinned and `1.16x` pinned, because PyTorch gains more from
pinning than Jittor does on that model.

Two corollaries for anyone reading a CPU number:

- A ratio measured without pinning is not evidence. Pin both sides, take a
  median over several runs, and say which configuration produced the number.
- "Cores busy" observed with `/proc/stat` is the parallelism a workload
  *achieved*, not a cap. Check `/sys/fs/cgroup/cpu.max` before concluding that
  a quota is limiting anything.

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
