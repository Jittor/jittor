# The two flatten nodes of batched Linear: removing them shrinks the graph by 13%

- Status: landed — the code and `tests/ops/test_matmul_higher_rank.py` are both on `2.0-refactor`
  (the commit at landing differs from the original commit below). The numbers in this document
  are measurements on the original branch; they were not re-run for this archival. Real-network
  parity and per-operator parity were still in progress at the time, so their conclusions are not
  attached.
- Date: 2026-09-12 (archived 2026-09-23)
- Baseline commit: `2cd9d6d0` (this round's change is `077e97ad`)
- Branch: `perf/matmul-rank` (merged; the branch itself can be discarded)
- Verification scope: decode/prefill (b1, d512, 8 layers) and large-batch MLP/CNN/transformer
  timings on a single CUDA card, plus both the CPU and CUDA passes of
  `tests/ops/test_matmul_higher_rank.py`; the card model was not recorded in the original, and the
  [metaop report](2026-09-12-cuda-metaop-launch-index-scalar.en.md) at the same baseline `d40a2e97`
  records it as H20 (sm_90). Not run: real-network forward/backward parity, the full
  `tools/run_test_suite.py` baseline, ROCm, NPU
- Maintainer: Jittor core maintainers
- Re-check conditions: a change to the shape contract of `cublas_matmul`, a change to the routing
  of `matmul`/`matmul_transpose`, or when the CPU side is also made to take the same path

## Problem

One transformer decode step (b1 s1 d512, 8 layers, with autograd enabled) builds **408** graph
nodes, while eager PyTorch with the same structure dispatches only about 318 times; the host path
is about 2.9x slower, and **96.9%** of that step is graph building. A node census shows that **80
of them are `reshape`**, and `reshape` is a pure view: `is_storage_view()` is true, it generates
no kernel, and it does almost nothing beyond `infer_shape`.

## 64 of these 80 come from one place

`matmul` and `matmul_transpose` get a rank>2 left operand into a 2-D cuBLAS kernel by flattening
first and restoring afterwards:

```python
aa = a.reshape((-1, a.shape[-1]))
cc = jt.nn.matmul_transpose(aa, b)
return cc.reshape(a.shape[:-1] + (-1,))
```

That is two nodes each time, and `nn.Linear` pays this pair on every forward. **`nn.Linear` goes
through `matmul_transpose`, not `matmul`** — this was initially looked for in the wrong place, and
changing `matmul` did not reduce the node count at all. In this test case 4 Linears × 8 layers × 2
= 64, plus the 2 × 8 = 16 written explicitly in the model, is exactly 80. The branch carries its
own comment `TODO:ugly implementation for tuner`.

## Change

A dense row-major `(d0,..,dn-1,m)` **is** its `(d0*..*dn-1,m)` flattening: the same pointer, the
same leading dimension, and all cuBLAS sees is the flattened extent. So `CublasMatmulOp` is made
to accept the rank it already has, with the contract:

- each operand collapses its leading dimensions into two (`rows = prod(shape[:-1])`,
  `cols = shape[-1]`) and applies the existing 2-D semantics;
- the output takes the leading dimensions from the side that contributes the rows: when `trans_a`
  is false the output is `a.shape[:-1] + [k]`; when true the rows come from the last dimension of
  `a` and the output falls back to rank 2, which is exactly the shape b's gradient
  `matmul(a, dout, 1, 0)` needs.

`a` must really be dense for the flattening to describe it, so strided views keep the original
reshape route. The rank-2 forward behavior is unchanged word for word (the flattening is the
identity).

**`rank 1` is the only trap here.** It also reaches the same branch of `matmul_transpose`, and its
flattening `(1,m)` is a row that reshape has to **add**, not a rank the kernel can read from the
buffer. The first version tripped on this by testing `len(a.shape) != 2`;
`tests/ops/test_matmul.py::test_linear1d` caught it, and it was changed to `> 2`.

### Why this is not a newly introduced device fork

I had previously rejected this myself on the grounds that it could only be changed on the CUDA
side and would make the graph shapes of CPU and CUDA differ. On review that reason does not hold:
the graph shape of `matmul` **already depends on the device** — `len_a == 2 and len_b == 2` goes to
`_matmul_2d_cublas` (CUDA-only), `len_a >= 3 and len_a == len_b` goes to
`select_kernel("batched_matmul", ...)` (selected per device), and only when neither matches does it
fall back to the generic broadcast+multiply+sum path. This change builds two fewer nodes on an
existing device-specific fast path; it does not add a new fork. The CPU side still goes through
reshape.

## Results

| | Before | After |
| --- | ---: | ---: |
| Nodes per step (autograd enabled) | 408 | **354** (−13.2%) |
| of which `reshape` | 80 | 16 |
| decode b1s1 d512 8 layers | 4.93 ms | **4.45 ms (1.10x)** |
| same, `jt.no_grad()` | 4.13 ms | **3.77 ms (1.09x)** |
| prefill b1s32 d512 8 layers | 5.39 ms | **4.91 ms (1.09x)** |
| mlp / cnn / transformer (infer and train, large batch) | — | **1.00x** |

decode and prefill were each measured for three rounds, and all three rounds reproduce.

**The full-network row is half of the conclusion, not noise**: MLP, CNN and large-batch transformer
are dominated by cublas/cudnn, and the 54 fewer nodes are on the host side, overlapped by device
execution, so there is no measurable change — and no regression. **This change only pays off when
the host is the critical path** (decode, prefill, small models, many small operators).

## Effect of all changes in this session stacked

Compared with the **unmodified baseline `d40a2e97`** (the two trees alternated on the same card, 4
rounds each):

| Form | Baseline | All changes | Speedup |
| --- | ---: | ---: | ---: |
| decode b1s1 d512 8 layers | 4.987 ms | **4.412 ms** | **1.13x** |
| prefill b1s32 d512 8 layers | 5.442 ms | **4.801 ms** | **1.13x** |

The four-round raw-value spread of these two rows is 2% and 1% respectively, and the two trees'
distributions **do not overlap**, so they can be cited.

**The large-batch full-network numbers on this machine cannot yield a citable cross-tree
comparison; do not report them.** With the same tree, the same case and the same configuration,
they jump within four rounds between a "fast mode" and a "slow mode" that differ by about a factor
of two: `mm`'s `mlp/infer` over three rounds is 0.937 / 0.941 / **3.250** ms, `cnn/train` is 11.25
/ 11.43 / **24.72** ms, while `basebench`'s same two items are stable at 2.17 and 26.7 (it **did
not draw the fast mode** in these rounds). So the practice of "take the minimum over several rounds
for each tree" produces fake results like **mlp 2.31x, cnn 2.63x** — the first draft of this report
was one step away from citing them.

The cause is consistent with the data: decode/prefill use small tensors and are host-bound, so they
are barely affected by other processes occupying the card on the same machine, hence tight and
unimodal; the full network uses large tensors and is GPU-bound, fully exposed to card contention,
hence bimodal.

**Criterion: look at the per-round raw values before any cross-process A/B.** When a metric is
bimodal, "take each minimum" is invalid — it compares "who was luckier to draw the fast mode". On
this machine GPU-bound forms can only be measured alternately by switching **within the same
process** (e.g. `cuda_thread_num` in `jt.flags.compile_options`); that convention gives mlp/cnn
1.00x, transformer 1.13-1.15x, and 1.41x for the d=256 transformer.

## Where the remaining nodes are

After the change a step has 354 nodes:

| Type | Count | Share | What it is |
| --- | ---: | ---: | --- |
| `binary` | 88 | 24.9% | real elementwise operations |
| `broadcast_to` | 64 | 18.1% | 32 paired scalar constants, 32 are Linear's bias broadcast |
| `array` | 32 | 9.0% | scalar constants |
| `cublas_matmul` | 32 | 9.0% | real GEMM |
| `tape` + `tapes` | 55 | 15.5% | autograd bookkeeping; absent with `jt.no_grad()` in inference |
| `getitem` | 24 | 6.8% | qkv's three splits, views |
| `reshape` | 16 | 4.5% | written explicitly in the model |
| `fuse_transpose` | 16 | 4.5% | attention's two transposes |

Candidates for the next step, ordered by measurement:

1. **A scalar constant is still two nodes** (`array` + `broadcast_to`, 64 in total = 18%).
   Measured piecewise: binary itself 5.32 µs, the expand node +1.99 µs, creating the array node
   +2.91 µs. A single operator that "produces a constant by shape" could bring it down to one node,
   but it requires a new op (registration, codegen, grad), is expected to be worth only 1.5-2%, and
   **must not be used for `jt.zeros/ones/full` as a side effect** — that would reintroduce the
   silent bug where their buffer is not their own shape (see
   `2026-09-12-cuda-metaop-launch-index-scalar.en.md`).
2. **Pure views still each pay a full graph node** (`getitem` 24 + `reshape` 16 + `fuse_transpose`
   16 = 15.8%). Wiping them out entirely would require making a view a property of Var rather than
   an Op, and Op nodes carry the autograd edges — a fundamental change to the graph model.
3. **Per-node cost itself**: 354 nodes per step against 4.588 ms is about 13 µs per node, while an
   isolated measurement of one binary's graph building is 5.31 µs. The remainder is in the Python
   layer (`nn.Module` calls, functional wrappers, `_runtime/dispatch.py`). After the change the
   cProfile total time is 0.500 → 0.387 s (−23%), but the remaining cost is already flattened out
   with no single dominant term: the largest is `nn/_bindings.py`'s binary wrapper at 19%, and most
   of its tottime is the C++ graph building it calls.

## Verification

- `tests/ops/test_matmul_higher_rank.py` (added with this change): forward for each rank,
  `matmul_transpose` (the spelling Linear uses), **rank 1**, the two forms of a strided left
  operand (slice and transposed view), two gradients, batched `nn.Linear` forward and three
  gradients, float32/float64 dtypes, once each on CPU and CUDA — **15 passed / 30 subtests**, with
  float64 bit-identical to numpy.
- `tests/ops/test_matmul.py` and `tests/nn/test_bmm.py`: the failing set is **identical nodeid by
  nodeid before and after the change (7 each)**. The rank-1 bug was caught here.
- Per-operator CPU/CUDA parity (about 227 generated cases): being re-run on a clean cache, with the
  conclusion attached separately. **The first conclusion is void because of an environment problem
  I created myself; it is recorded here to avoid repeating it**: see the next section.
- **The real-network forward/backward parity was not run** (`tests/models/test_network_parity.py`
  and `test_network_training_parity.py`, ResNet/ViT/GPT-2/diffusion UNet). Two traps: without
  `REAL_TORCH_SITE`, all 16 cases **skip while the gate still reports green**; with it set, this
  round still failed with `REAL_TORCH_SITE did not provide independent binary PyTorch` — `torch`
  in that session was resolved by the deployed shim first, and the oracle needs an environment
  where the real torch outranks the shim, which was not configured successfully this round.

## A "large-scale failure" I created myself, and the correct order

The first time I ran this gate on the mm tree I got `671 failed / 104 passed` (baseline
`121 failed / 654 passed`), with 1304 `cudaErrorIllegalAddress` — one illegal memory access poisoned the CUDA
context, after which 549 cases failed in a cascade; the first "new" failure was `test_inner`.

**This is not a regression.** The elimination process (each item is worth recording, because each
is also a method):

- Running the 105-125 window containing `test_inner` on both trees with a fresh cache: **the
  failing set is exactly identical nodeid by nodeid** (7 each; 422 versus 429 seconds).
- `test_inner` is `jt.matmul(a, b.transpose(-1,-2))` with both operands at rank 2; checking step by
  step, the `n/m/k` and output shape that `infer_shape` and `jit_run` give at rank 2 are
  **equivalent word for word** to before the change, and the gradients are all rank 2 as well.
- I suspected that a rank>2 `dout` produced by `CublasMatmulOp::grad` might be a strided view that
  misaligns the flattening. A hard check does not trigger: `CublasMatmulOp` does not declare
  `accepts_storage_strides`, so the framework materializes strided inputs as contiguous **at
  construction time**. The check was kept — it writes the contract on the op itself rather than
  scattering it across call sites.
- The `compute-sanitizer` lead is **fake**: all three trees (including an unmodified baseline
  `d40a2e97`) report "errors", but the body is
  `CUDA_ERROR_INVALID_HANDLE (error 400) on cuKernelGetFunction`, a known side effect of the sanitizer with jittor's dynamically loaded JIT
  `.so`, not a memory error. **The count only reflects how many JIT modules were probed** (baseline
  1, scalar 2, mm 3); do not treat it as regression evidence.

**Cause**: I `rm -rf`'d the `JITTOR_HOME` of the first gate process **while it was still alive**,
and killed it only afterwards. The plan document's rule is "do not kill -9 a gate; if it must be
killed, delete its JIT cache first and re-run" — "delete the cache first and re-run" means delete
**after killing and before the next run**, not while it is still running. The correct order:

```bash
pkill -TERM -f '<这道门禁的选择器>'     # 1. 先停
pgrep -f '<同上>' || echo stopped       # 2. 确认真的停了
rm -rf "$JITTOR_HOME" && mkdir -p "$JITTOR_HOME"   # 3. 再清缓存
# 4. 然后才重跑
```
- **Not run**: the full maintenance baseline of `tools/run_test_suite.py`, ROCm, NPU. This change
  only affects CUDA's `cublas_matmul` and the routing of `matmul`/`matmul_transpose`; the CPU takes
  the original route.

## Reproduction

```bash
export JITTOR_LAB_ROOT=<仓库同级的 jittor-lab>
source $JITTOR_LAB_ROOT/_state/metaop/env-mm.sh
cd $WT
CUDA_VISIBLE_DEVICES=<一张卡> $PY -m pytest -q tests/ops/test_matmul_higher_rank.py
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/bench_decode.py
```

The node census uses the difference of `jittor_core.number_of_lived_ops()` plus
`jt.dump_all_graphs()`, building a step's graph **without synchronization**; under `jt.no_grad()`
this count is unavailable (the graph is released promptly and the dump only sees fragments), so
there only timings can be compared.
