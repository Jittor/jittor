---
name: jittor-torch-diff
description: Differential ("对拍") testing and gradient debugging for jittor-as-torch vs real PyTorch. Use whenever verifying numerical parity (forward and/or backward) of a transformers/torch model running on jittor against real torch, or debugging why param.grad / loss.backward() / autograd behaves unexpectedly on jittor. Captures the env topology, the remote-fs gotcha, the net-scaled grad metric, and ready-to-run harness scripts.
---

# jittor ⇄ real-torch differential testing & grad debugging

Reusable harness for proving **G3 (逐层数值对齐)** and debugging autograd on the
jittor-as-torch stack. Built from repeatedly-rebuilt ad-hoc scripts — use these
instead of re-deriving them.

## Environments: two interpreters

Every tool here runs the same script twice, in two interpreters that must never be
the same one:

| role | variable | what it is |
|---|---|---|
| **JITTOR** | `$JT_PY` (`<jittor-python>`) | Python 3.11 whose `import torch` resolves to the Jittor shim built from the tree under test (`JITTOR_TORCH_SHIM=1`, or a deployed shim), plus the downstream packages the probe needs (transformers, safetensors) |
| **REAL TORCH** | `$RT_PY` (`<real-torch-python>`) | an independent real PyTorch (the oracle), same downstream package versions, no jittor. If its wheels need a newer `libstdc++` than the system one, set `RT_LIBSTDCXX` to that library and the scripts preload it |

Neither has a default: export `JT_PY` and `RT_PY` (and optionally `RT_LIBSTDCXX`)
before running anything below. A CPU build is enough for the model and op
batteries; the CUDA probes additionally need a CUDA build of the shim interpreter
and a CUDA build of real torch on a GPU host (select the card with
`CUDA_VISIBLE_DEVICES=<gpu>`).

## Non-negotiable gotchas (each cost real time before)

1. **Remote-fs**: in a split agent setup (file tools on one host, the shell on another)
   the Write tool writes to a *different* filesystem than the shell host's `/tmp`. Write throwaway scripts to the box via **Bash heredoc** (`cat > $TMPDIR/x.py <<'PY'`), NOT the Write tool. (Files **under the project tree** ARE shared — those are fine to Write.) Always use `$TMPDIR`, never `/tmp`.
2. **Offline**: set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in both envs.
3. **Confirm the oracle is real**: assert `not hasattr(torch, 'jittor')` and
   `torch.__version__` endswith `+cpu` on the RT side — never compare jittor to itself.
4. **Noise**: filter jittor's compile spam: `| grep -v -E "^\[i |^\[w |Compiling|cache_path|mpicc|addr2line|Total mem|Load cc|Writing model|Loading weights|Model config|DeprecationWarning"`.
5. **JIT warmup**: first iter compiles (gpt2 train 1st-iter ≈ 41s). Time *steady state* separately; bump Bash `timeout` to ≥420000ms for first runs.
6. **Python version**: verify on the maintainer interpreter (**py3.11**) first; an older
   1.3.11 build miscompiled JIT ops under py3.13, so a 3.13-only divergence needs a 3.11
   re-run before it is reported.

## The net-scaled grad metric (don't trust per-param rel-diff)

Per-parameter `max|g_jt - g_rt| / (max|g_rt|+eps)` is **misleading**: a param whose
own grad is ~1e-12 (8 orders below the dominant grad) shows a huge *relative* diff
from pure float32 roundoff while being numerically perfect. Normalize by the
**network-wide** grad scale: `gmax = max over all params of max|g_rt|`; report
`max|g_jt - g_rt| / gmax`. A backward pass is correct if net-scaled worst ≲ 1e-5.
(Real example: BERT per-param worst 2.35 → net-scaled 4.3e-7 = perfect.)

## Run it

```bash
bash agent/skills/jittor-torch-diff/run_parity.sh gpt2     # forward+backward parity vs real torch
bash agent/skills/jittor-torch-diff/run_parity.sh bert "$TMPDIR/p_bert"
```
`parity.py` has three subcommands (`jt` save side, `rt` oracle side, `cmp` compare);
`run_parity.sh` chains them across the two envs and prints the verdict table.
Configs live in `parity.py:make_config` — add an arch there. Tiny by default
(hidden 64, 2 layers, dropout 0) for fast, deterministic, JIT-cache-friendly runs.

## Debug a grad/autograd problem

```bash
$JT_PY agent/skills/jittor-torch-diff/grad_probe.py bert
```
Reports, for a fresh tiny model: which params have `.grad is None` after
`loss.backward()`, the leaf-registry length (`jt._torch_leaf_params`), whether
`jt.grad(loss, params)` returns correct nonzero grads (isolates *computation* vs
*exposure* bugs), and top/bottom grad magnitudes. This is how the "param.grad is
None" exposure bug (fixed in `jittor.compat.torch`) was isolated: grads computed fine,
only `.grad` exposure was broken.

## Op-level differential battery (finds *silent-wrong* op semantics)

```bash
OUT=$TMPDIR/op_parity
env PYTHONPATH=$PWD/python HF_HUB_OFFLINE=1 $JT_PY agent/skills/jittor-torch-diff/op_parity.py jt  $OUT
env ${RT_LIBSTDCXX:+LD_PRELOAD=$RT_LIBSTDCXX} $RT_PY agent/skills/jittor-torch-diff/op_parity.py rt  $OUT
                                    $JT_PY agent/skills/jittor-torch-diff/op_parity.py cmp $OUT
```
Runs ~38 tensor ops (the torch *public* API: where/gather/scatter/sort/topk/var/std/
unfold/diagonal/masked_select/...) on **identical seeded inputs** through jittor-as-torch
and real torch, and prints PASS/FAIL per op. Catches semantic divergences that model
probing only hits by luck — e.g. it pinned `Var.where` treating self as the condition,
`var`/`std` biased-vs-unbiased, and `.sort/.topk` method return-shape gaps.

Two harness rules learned the hard way:
- **Clone inputs per op** — some jittor ops are *in-place* (e.g. `Var.scatter` mutates
  self, unlike torch's out-of-place `scatter`); a shared input tensor gets poisoned and
  every later op on it spuriously "diverges". (`scatter` in-place was found this way.)
- **Test the torch-grade *path***: `.max(dim)/.min(dim)` METHOD form stays jittor-native
  (values-only) because core relies on it; test `torch.max(x,dim).values` (the function
  form, which IS correct) not `x.max(dim)[0]`.

Add an op as one line in `battery()`: `add("name", lambda T, lib: T["a"].op(...))`.

## Complex CUDA parity probe (`complex_cuda_parity.py`)

Use this when auditing native `complex64` CUDA support against real PyTorch CUDA.
It has the same three-phase shape as the other parity tools:

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"   # from the repository root
STATE="$JITTOR_LAB_ROOT/_state/jittor-torch-diff/complex_cuda_parity"
OUT="$JITTOR_LAB_ROOT/jittor-torch-diff/complex_cuda_parity_out"
CUDA_ROOT=<cuda-root>      # CUDA toolkit with bin/nvcc, e.g. the one Jittor auto-installs
mkdir -p "$STATE" "$OUT"
env HOME="$STATE/home" JITTOR_HOME="$STATE/jittor" \
    JTCUDA="$CUDA_ROOT" CUDA_HOME="$CUDA_ROOT" nvcc_path="$CUDA_ROOT/bin/nvcc" \
    PYTHONPATH="$PWD/python" cache_name=complex_audit_probe \
    use_parallel_op_compiler=0 CUDA_VISIBLE_DEVICES=<gpu> \
    "$JT_PY" agent/skills/jittor-torch-diff/complex_cuda_parity.py jt "$OUT"
CUDA_VISIBLE_DEVICES=<gpu> "$RT_PY" agent/skills/jittor-torch-diff/complex_cuda_parity.py rt "$OUT"
"$JT_PY" agent/skills/jittor-torch-diff/complex_cuda_parity.py cmp "$OUT"
```

It compares stable native complex64 items directly and records known CUDA gaps
(`prod`, general `linalg.eig`) as expected Jittor-side errors. Some aggregate
items (ComplexNumber and linalg residuals, `SEQUENCE_SENSITIVE_ITEMS`) are marked
`sequence_sensitive`: focused tests should be used for final linalg/ComplexNumber
conclusions. `rfft` and `irfft_rfft` are hard failures, not sequence-sensitive: the
earlier report of an rfft sequence risk after a complex forward/grad prelude was not
reproducible and was withdrawn; the deterministic regression is
`compat/tests/torch/test_torch_compat_fft_einsum.py::test_rfft_after_complex_forward_backward_sequence`.

## Op-level BACKWARD parity (`grad_ops.py`)

Same 3-subcommand shape (`jt`/`rt`/`cmp`) but compares **gradients**: differentiates
`(out**2).sum()` w.r.t. a float input on both stacks and compares `input.grad`
(jittor via `jt.grad`, real torch via `.backward()`). Catches scatter-add/gather/
reindex BACKWARD bugs that forward-only testing misses — e.g. it confirmed
as_strided/unfold/diagonal/scatter_reduce/index_add all have correct gradients
(jittor's autograd through reindex/getitem does the right scatter-add). Self-contained
pow2 loss = no external weight to size-match. Verified 18 ops, ALL MATCH, dual-card
(Ascend + CUDA). Tip: if jt AND rt error identically on an op, it's a malformed test
input, not a jittor bug (both raise the same shape error).

## Extending this skill

This is a living toolbox — when you build a new diff/debug probe (a new metric, a
new failure class, an N-card CUDA variant), add it here so the next run starts warm.
