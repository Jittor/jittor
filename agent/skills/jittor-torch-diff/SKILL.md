---
name: jittor-torch-diff
description: Differential ("对拍") testing and gradient debugging for jittor-as-torch vs real PyTorch. Use whenever verifying numerical parity (forward and/or backward) of a transformers/torch model running on jittor against real torch, or debugging why param.grad / loss.backward() / autograd behaves unexpectedly on jittor. Captures the env topology, the remote-fs gotcha, the net-scaled grad metric, and ready-to-run harness scripts.
---

# jittor ⇄ real-torch differential testing & grad debugging

Reusable harness for proving **G3 (逐层数值对齐)** and debugging autograd on the
jittor-as-torch stack. Built from repeatedly-rebuilt ad-hoc scripts — use these
instead of re-deriving them.

## Boxes & environments (cscg-hw00, the Ascend box — the Bash host)

| role | python | what it is |
|---|---|---|
| **JITTOR** (`$JT_PY`) | `/home/yizhang/miniconda3/envs/jt-torch/bin/python` | py3.11, `import torch` == jittor dev tree, transformers 5.12.1 + safetensors deployed, CPU build |
| **REAL TORCH** (`$RT_PY`) | `/home/yizhang/miniconda3/envs/rt/bin/python` | real torch 2.12.1+cpu (the oracle). **Needs** `export LD_PRELOAD=/home/yizhang/miniconda3/envs/rt/lib/libstdc++.so.6` |

N-card box (4090, CUDA) for G2: `ssh -p 20002 -o BatchMode=yes zy@116.177.253.46`,
dev tree `/home/zy/jittor_dev`, env `jt311` (`/home/zy/miniconda3/envs/jt311/bin/python`). No real torch there.

## Non-negotiable gotchas (each cost real time before)

1. **Remote-fs**: the Write tool writes to a *different* filesystem than the Bash
   host's `/tmp`. Write throwaway scripts to the box via **Bash heredoc** (`cat > $TMPDIR/x.py <<'PY'`), NOT the Write tool. (Files **under the project tree** ARE shared — those are fine to Write.) Always use `$TMPDIR`, never `/tmp`.
2. **Offline**: set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in both envs.
3. **Confirm the oracle is real**: assert `not hasattr(torch, 'jittor')` and
   `torch.__version__` endswith `+cpu` on the RT side — never compare jittor to itself.
4. **Noise**: filter jittor's compile spam: `| grep -v -E "^\[i |^\[w |Compiling|cache_path|mpicc|addr2line|Total mem|Load cc|Writing model|Loading weights|Model config|DeprecationWarning"`.
5. **JIT warmup**: first iter compiles (gpt2 train 1st-iter ≈ 41s). Time *steady state* separately; bump Bash `timeout` to ≥420000ms for first runs.
6. **py3.13**: jittor 1.3.11 miscompiles JIT ops under py3.13 — always verify on **py3.11**.

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
bash agent/skills/jittor-torch-diff/run_parity.sh bert /tmp/p_bert
```
`parity.py` has three subcommands (`jt` save side, `rt` oracle side, `cmp` compare);
`run_parity.sh` chains them across the two envs and prints the verdict table.
Configs live in `parity.py:make_config` — add an arch there. Tiny by default
(hidden 64, 2 layers, dropout 0) for fast, deterministic, JIT-cache-friendly runs.

## Debug a grad/autograd problem

```bash
$JT_PY .claude/skills/jittor-torch-diff/grad_probe.py bert
```
Reports, for a fresh tiny model: which params have `.grad is None` after
`loss.backward()`, the leaf-registry length (`jt._torch_leaf_params`), whether
`jt.grad(loss, params)` returns correct nonzero grads (isolates *computation* vs
*exposure* bugs), and top/bottom grad magnitudes. This is how the "param.grad is
None" exposure bug (fixed in `torch_compat.py`) was isolated: grads computed fine,
only `.grad` exposure was broken.

## Op-level differential battery (finds *silent-wrong* op semantics)

```bash
OUT=$TMPDIR/op_parity
env PYTHONPATH=$PWD/python HF_HUB_OFFLINE=1 $JT_PY .claude/skills/jittor-torch-diff/op_parity.py jt  $OUT
env LD_PRELOAD=$RT_LIBSTDCXX        $RT_PY .claude/skills/jittor-torch-diff/op_parity.py rt  $OUT
                                    $JT_PY .claude/skills/jittor-torch-diff/op_parity.py cmp $OUT
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
