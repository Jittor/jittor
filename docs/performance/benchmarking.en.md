# Performance benchmarking

Jittor uses ASV 0.6.6 to retain per-commit timing and memory results. The suite
**deliberately uses the current Python environment** (`--python=same`): creating a fresh
environment for every historical commit would mostly measure Jittor's compilation and
installation cost rather than steady-state runtime performance.

## Reliable local timing

Use `jt.benchmark` for local microbenchmarks; do not wrap `perf_counter()` directly
around a lazy operation:

```python
import jittor as jt

inputs = [jt.randn(1024, 1024) for _ in range(4)]
result = jt.benchmark(
    lambda value: (value * value).sum(),
    inputs,
    warmup=2,
    repeat=10,
)
print(result.median, result.samples)  # 单位：秒
```

The input pool is snapshotted and materialized before timing, then consumed in
round-robin order. **Warmup is mandatory** and is not counted in the returned samples,
so first-use compilation falls outside the timed region. The callable under test must
return **all** outputs that belong to the measured work; nested tuples, lists, and dicts
are supported. Every round holds all returned Vars and performs one targeted device
synchronization before stopping the clock.

These rules avoid three common false results:

- **Common subexpression elimination (CSE)**: repeatedly constructing the same
  expression while the previous lazy output is still alive reuses that graph.
  `jt.benchmark` materializes and releases the previous round before constructing the
  next one; when an operation mutates its input, use multiple resident pool entries.
- **Dead code elimination**: an unreferenced lazy output can be dropped without
  executing. The API **rejects** calls that return no `jt.Var`, and holds every returned
  Var until synchronization completes.
- **Unmaterialized work**: timing only Python graph submission measures neither CPU
  execution nor asynchronous device completion. Every sample includes that targeted
  synchronization.

Samples include Python graph construction plus execution. **Use the ASV suite below for
retained results across commits**, and keep correctness checks separate from timing.

## Cache isolation

**ASV and the unit test suites must never compile into the same Jittor cache.** Each
benchmark's setup validates these two variables before importing Jittor:

```bash
export JITTOR_HOME="${JITTOR_ASV_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/jittor-asv}"
export cache_name="asv-local"
export ASV_PYTHONPATH="$PWD/python"
```

`cache_name` must start with `asv-`, and the resolved `JITTOR_HOME` must contain an
`asv` path component. **A missing or reused cache is a hard error, not an ASV skip.**
Parallel CPU, CUDA, and NPU jobs must use different `JITTOR_HOME` directories or
different `cache_name` values.

When ASV starts an existing environment it clears the ordinary `PYTHONPATH`, so
`ASV_PYTHONPATH` is the source-tree path passed explicitly to the benchmark process; nox
sessions set it to `python/` automatically. These sessions also place the Jittor
compilation cache and generated ASV state outside the checkout. `ASV_RESULTS_DIR` and
`ASV_HTML_DIR` can select persistent external locations for the raw JSON and the
published report.

## The canonical nox run

Install the pinned development tools and record the maintained CPU selection:

```bash
python -m pip install -r requirements/dev-tools.txt
python -m nox -s benchmark -- \
  --bench '^(operators|optimizer_step|convolution)\.'
```

On a tagged CUDA host with the pinned CUDA benchmark dependencies installed:

```bash
python -m nox -s benchmark_cuda -- \
  --bench '^(operators|optimizer_step|convolution|reductions|transfer|tiny_llama)\.'
```

Both sessions run the full ASV workflow: `check`, register the machine, `run` and sample
**the exact current commit**, `compare` against the nearest cached ancestor (or an
explicit `ASV_COMPARE_BASE`), then `publish` the HTML report. `ASV_COMPARE_FACTOR`
controls the regression factor and must be greater than 1. The first run bootstraps
against itself as baseline.

Accelerate's wrapping overhead uses a separate process and cache, and is run on demand:

```bash
# JITTOR_CI_PYTHON 指向已安装 ASV、NumPy 和 Accelerate 的解释器
python -m nox -s benchmark_accelerate
# 使用 CUDA 编译环境时，额外要求真实设备及 nvcc
ASV_ACCELERATE_DEVICE=cuda python -m nox -s benchmark_accelerate
```

This entry point selects only `accelerate_training` by default, comparing direct training
through the same Jittor Torch frontend against `Accelerator.prepare/backward` wrapping; it
is not a speed comparison against real PyTorch and does not enter default CI. Missing
dependencies, frontend activation failure, or actual computation failure report an error
directly.
**A run counts as successful only when both the result JSON and `index.html` exist.**

## Ecosystem comparison

`tests/compat/torch/test_ecosystem_parity.py` and `test_ecosystem_speed.py` compare **two
independent processes** -- Jittor's and binary PyTorch's. Both processes must obtain their
own `torch` namespace before loading downstream libraries.

They share a single package site only when the CPython ABIs are compatible (usually the
same major and minor version): an extension module with an ABI tag in the site directory
means a 3.12 reference interpreter cannot import a 3.11 site, and would fail at the first
compiled dependency with an unrelated error. When the ABIs differ, use
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` to specify the site of the independent oracle,
and each side imports its own copy. **When sharing a site, both version and origin must
be equal; with separate sites the versions must be equal, and the origins are expected to
differ.** CUDA runs explicitly align matmul/cuDNN TF32, and optional cuDNN autotuning is
enabled for both runtimes.

Correctness tensors are **copied immediately** so that later optimizer/gradient writes do
not modify some NumPy view. Timed training preallocates multiple resident input slots and
a loss weight tensor. Jittor holds every requested gradient and explicitly synchronizes
those Vars; **neither runtime performs per-gradient D2H inside the timing window.**

Masked SDPA performance must preserve the "whole row masked out" semantics. The
maintained attention path on CUDA hands explicit masks to the softmax kernel's
`zero_all_neg_inf` mode instead of building a separate row-validity reduction plus two
ternary graphs. Pure causal attention does not need that mode because every row contains
its own diagonal element. **Profiling comparisons must check both the graph's row count
and wall time.**

Result labels must match the checkout. nox sessions reject a dirty working tree unless
the caller deliberately sets `ASV_ALLOW_DIRTY=1` -- that is for local investigation only
and **is not publishable evidence**.

CPU parameters import and execute without CUDA too. When Jittor or real PyTorch cannot
execute on CUDA, CUDA parameters raise an explicit ASV skip. Real PyTorch is an
**optional oracle** and must be installed separately for the target platform; a missing
`torch` or one that resolves to the shim is reported as a skip and **never as a
zero-time result**. CI must have at least one mandatory Jittor CPU case that actually
executes -- **a run where every parameter is skipped is unacceptable.**

## Manual comparison on Ascend NPU

The ASV suite does not cover Ascend. Speed and numerical comparisons between the ACL
backend and same-machine `torch + torch_npu` live in
[`benchmarks/acl/`](https://github.com/Jittor/jittor/blob/master/benchmarks/acl/README.md): `bench_jittor.py` / `bench_torch.py` run the same set of networks
(specifications in [`benchmarks/acl/models.md`](https://github.com/Jittor/jittor/blob/master/benchmarks/acl/models.md)), `benchmarks/acl/xval/` feeds Jittor weights and
inputs into torch to compare loss and gradients step by step, and `benchmarks/acl/probes/`
contains probes that call aclnn directly, bypassing Jittor. All of these need real devices
and are **run manually**; they do not enter CI and must use a different compilation cache
from ASV.

That README records three pitfalls measured on this machine: per-call `sync()` serializes
submission and execution, recompiling between A/B runs systematically favors whichever
ran first, and `jt.profiler`'s `TotalTime` is not additive for small operators. For
numerical drift on long trajectories, first run the comparison in
`benchmarks/acl/xval/control_th.py` before drawing conclusions.

## CPU thread pinning

Any CPU comparison must pin cores. Neither Jittor nor PyTorch sets OpenMP affinity by
default, and on a multi-core host the scheduler migrates threads between measurements:
the same Jittor ViT step measured `0.5555`~`0.7074s` across six runs -- that is **two
clusters differing by 25%, not one spread**. After setting

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

four runs converged to `0.5165`~`0.5195s`. Pinning is worth `10`–`30%` for **both
runtimes**, so an unpinned comparison can invert the conclusion: one diffusion UNet reads
as `0.90x` unpinned and `1.16x` pinned, because PyTorch benefits more from pinning than
Jittor on that model.

Readers of CPU numbers should remember two corollaries:

- **A ratio measured without pinning is not evidence.** Pin both sides, take the median of
  several runs, and state which configuration produced the number.
- The "cores busy" seen in `/proc/stat` is the parallelism that load **actually
  achieved**, not a ceiling. Before asserting what a quota limits, look at
  `/sys/fs/cgroup/cpu.max`.

## Recording the selected revision

An existing environment cannot make ASV build arbitrary history. Run the suite only in a
checkout that matches the result label, and record that exact commit:

```bash
commit=$(git rev-parse HEAD)
asv --config benchmarks/asv/asv.conf.json run --python=same --set-commit-hash "$commit"
```

To make a comparison, create **a dedicated benchmark worktree** and switch only that
worktree to the two or three revisions under review, running the command above after each
switch. Keep the same compiler, accelerator, dependency versions, cache warmup strategy,
and ASV results directory. **Do not run `ALL` or an unbounded revision range.** Once both
selected commits have results, run
`asv --config benchmarks/asv/asv.conf.json compare <base> <candidate>`.

These direct commands run from the repository root. The configuration is owned by
`benchmarks/asv/`; its `repo: "../.."` resolves relative to the configuration file, and
`benchmark_dir: "."` and the output directory also resolve relative to the config
directory. The source configuration puts default output in the repository-adjacent
`jittor-lab/_state/asv/`; with a custom `JITTOR_LAB_ROOT`, use the nox entry point, which
generates the corresponding absolute state paths. The maintained `nox -s benchmark` entry
generates a configuration using absolute repository, benchmark, and external state paths,
so moving the source configuration does not redirect CI artifacts into the checkout.

## CI retention and cadence

The CPU workflow runs `nox -s benchmark` in the baseline CPU container: it restores prior
result JSON when available, records the current commit, compares and publishes, then
saves the updated result cache. The raw JSON and generated HTML are uploaded together as
retained CI artifacts.

PRs and ordinary pushes select `operators`, `optimizer_step`, and `convolution`; scheduled
and manual runs also select `reductions`. Large-size reductions do not add per-PR wait
time.

**CUDA benchmarks are deliberately not a PR gate.** The CUDA workflow runs
`nox -s benchmark_cuda` on a weekly schedule and on manual trigger, after its real-device
test gate. It uses a tagged CUDA 12.2 RTX 4090 runner, a separate result cache, and
uploads separate CUDA JSON/HTML artifacts. This cadence keeps accelerator regressions
visible without occupying dedicated hardware for every source push.
The CUDA schedule selection includes operators, convolution, optimizer, reductions,
transfer, and Tiny Llama.

## Current benchmark set

| ASV module | Coverage | Parameters |
| --- | --- | --- |
| `operators` | matmul, softmax, LayerNorm, GELU | Jittor / optional torch; CPU/CUDA |
| `convolution` | standard 3×3, stride-2 1×1, depthwise 3×3; forward and forward+backward | Jittor / optional torch; CPU/CUDA; strict FP32 |
| `tiny_llama` | LlamaModel forward, forward+backward | Jittor / optional torch; CUDA |
| `optimizer_step` | scaling of SGD and AdamW steps with ready gradients | 32/128/512 tensors; CPU/CUDA |
| `reductions` | sum/max/min, including order-sensitive input patterns | 1M/16M/64M elements; CPU/CUDA |
| `transfer` | host↔device copies, device operations after reads, and peak memory | Jittor / optional torch; CUDA |
| `accelerate_training` | direct training and Accelerate wrapping, 8-step training | SGD/AdamW; CPU/CUDA; separate opt-in entry |

The convolution setup, outside timing, validates the entire forward tensor against an
independent NumPy float64 reference, as well as the input, weight, and bias gradients in
training mode; timing includes actual computation and device synchronization. Convolution
disables TF32, while the other suites continue to run under their existing precision
protocols. A missing device or an uninstalled optional PyTorch may skip; compilation,
registration, and execution errors on a device declared available, as well as a broken
PyTorch installation, are failures and must not be converted into skips.

Tiny Llama follows the established configuration: 2 layers, hidden 256, intermediate 768,
8 attention heads / 4 KV heads, batch 2, sequence 128. Its setup checks that the output
and every requested gradient are finite and nonzero. The optimizer setup fixes the total
element count at 262,144, so **the tensor-count axis exposes per-tensor graph
construction and launch overhead** rather than larger computation. CUDA comparisons use
float32 and enable TF32 on both backends. Operator benchmarks use inference/no-gradient
semantics; Tiny Llama enables gradients only for its "forward+backward" parameter.

Every memory result is an ASV `track_*` benchmark in `bytes`. On CUDA,
`track_working_set_bytes` is the backend allocator's **synchronized live working set**
after one operation/model/step; on CPU it is the process peak RSS reported by the
operating system. They are a stable regression signal **on the same backend and the same
machine**; they are not NVML peaks and **should not be compared across devices**.
