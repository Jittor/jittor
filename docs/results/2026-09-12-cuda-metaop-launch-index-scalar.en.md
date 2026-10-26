# CUDA metaops: launch configuration, strided indexing and scalar-constant fusion

- Status: partially complete — three changes are implemented and measured on H20; the third also fixes a silent miscalculation; the full CPU/CUDA gate has not finished running
- Date: 2026-09-12
- Baseline commit: `d40a2e97`
- Branch: `perf/metaop-speed` (first two items), `perf/scalar-fusion` (third item)
- Verification scope: a single H20 (sm_90), CUDA 12.9, g++ 12.3.1, Python 3.12; metaop microbenchmarks and same-process
  full-network A/B, `tests/backends/cuda`, per-operator CPU/CUDA parity (about 227 generated cases) and the native CPU
  suite compared nodeid by nodeid against the baseline. Not run: the full `tools/run_test_suite.py` baseline, ROCm, NPU;
  the launch-configuration defaults are unverified on sm_89 and earlier cards
- Maintainer: Jittor core maintainers
- Re-check conditions: a new CUDA architecture, a change to `broadcast_to`'s storage semantics, or another adjustment to the launch configuration

## Problem

How far the metaops (unary/binary/ternary/reduce/broadcast/reindex) on CUDA still are from the hardware limit, where the gap is, and how much speedup closing that gap would bring to a full network.

## Environment and measurement convention

H20 (sm_90, nominal about 4 TB/s), CUDA 12.9, g++ 12.3.1, Python 3.12. The work tree and run state are under `$JITTOR_LAB_ROOT` and are not version-controlled.

**Other processes on this machine occupy all 8 cards for long periods**, so any synchronization pays an extra fixed stall of about 2.3 ms — real PyTorch measures the same stall on the same card, so this is not a Jittor problem, but it destroys the "one synchronization per operator" measurement method. The convention is therefore:

- each case launches R **mutually independent** operators of the same kind, synchronizes once, and reports the **slope** `(t(R2)-t(R1))/(R2-R1)`;
- R1 and R2 are measured **alternately within the same round** and the **median** of the paired differences is taken. Taking the minimum of each window separately pairs a quiet R2 with a busy R1, and a negative slope was observed in practice;
- creating "mutual independence" **must not be done by adding an operator**. The first attempt used `a+b+float(i)` to distinguish instances, and the profiler showed that scalar compiled into an `array` kernel of its own, accounting for 44% of a run's records. The fix was to pre-allocate R different inputs;
- full-network A/B switches `cuda_thread_num` **within the same process** and measures alternately, avoiding placement bias caused by recompilation.

## Changes

### 1. `cuda_thread_num` default 1024 → 256

`ParallelPass` writes this value into both the kernel's `__launch_bounds__` and the launch shape. 1024 is the largest block the hardware allows, and it suppresses both occupancy and the per-thread register limit.

Sweeping 1024/512/256/128/64 over thirteen metaop shapes: **256 is optimal or tied with the optimum on every one, and no shape gets slower**; 512 is next; 64 and 128 get worse on axis reductions. `cuda_block_num` stays at 2048 (a larger thread budget measured worse).

One mechanistic point must be stated clearly: **the initial suspicion was that the 1024 `__launch_bounds__` caused register spilling, and ptxas refuted that hypothesis** — ternary and the five-operator fused chain have the same register count under both bounds (28/32) with zero spills. The gain comes from occupancy and wave quantization, plus each thread processing more elements once the thread budget drops from 2M to 512K. Both factors contribute: with the budget fixed at 2M, narrowing only the block is 1.11x, and lowering the budget to 512K as well is 1.33x.

The 1024 hard-coded in `SharedReducePass` is connected to the same accessor (`cuda_block_width`) as well. It only takes effect at `para_opt_level >= 4` (the default is 3), but a block wider than `__launch_bounds__` cannot be launched at all, so it could not be left as is.

### 2. Elementwise metaops no longer do an integer division per axis for strided indexing

The generated kernel used to flatten the loop coordinates into a linear index and then decode them back with one `%` and one `/` per axis — even though the coordinates were already at hand. After encoding "which axes actually move the physical index" per operand into the JIT key:

- axes outside the mask have their term removed entirely;
- an axis's division is kept only when there are still lower axes that need the remainder;
- axis 0 does not need a modulo (the linear index is already smaller than that axis's extent at that point).

Thus row broadcast keeps only one modulo, column broadcast only one division, and a rank-1 view neither (each used to be two). ternary used to share one `STRIDED` flag across its three operands, so one strided condition forced the full decoding onto the two value operands that are usually contiguous; now they are separated per operand and share one remainder chain.

**Note**: measured on its own, this item's gain is small. In an independent CUDA microbenchmark, replacing two div/mod with one modulo is only **1.13x** — this kernel is memory-bound, so most of the integer division is hidden. Its real value is the stability of the broadcast cases when stacked with item 1 (see below).

### 3. Keep the fusible form for one-element expand (a scalar constant no longer becomes its own kernel)

In this branch's storage-boundary refactor (`7e83d6da`), `broadcast_to` became a **storage descriptor**: the output shares storage with the input, broadcast axes have stride 0, and `set_type(OpType::other)`. That is correct for real broadcasts and no longer materializes them.

But it also cut off scalar-constant fusion. The graph of `x * 0.5` is `array(1元素) → broadcast_to → binary`, and `count_fuse`'s `edge_fusable` rejects any edge with `OpType::other` on either end **before looking at `_force_fuse`**. `ArrayOp` explicitly sets `_force_fuse` for one-element outputs and notes "Fused scalar values are emitted inside generated kernels" — exactly the fusion that is wanted.

Measured against the pre-refactor commit `845dd6c6`:

| | Generated kernels |
| --- | --- |
| `845dd6c6` (before the refactor) | 1: `array + broadcast_to + binary` in one kernel, the scalar is a **kernel parameter** |
| `d40a2e97` (after the refactor) | 2: a `kernel<<<1,1>>>` writing 4 bytes, then a binary reading it back elementwise |

The cost counted in launches (one Transformer training step, using the profiler's counting convention):

| Category | Before fix/step | After fix/step |
| --- | ---: | ---: |
| **One-element constant as its own kernel** | **130.0** | **1.0** |
| Fused elementwise | 115.9 | 164.9 |
| cublas GEMM | 72.0 | 72.0 |
| Other | 36.0 | 36.0 |
| Fusion with reduction | 16.0 | 16.0 |
| **Total** | **369.9** | **289.9** |

**130 of the 370 launches per step just write one scalar to device memory.** The fix: a one-element source does not go through the storage descriptor (sharing 4 bytes has no benefit), keeps the computational form, and is folded into the consumer by the fuser. `is_storage_view()` is changed to return a member decided by `infer_shape`, the same pattern as `GetitemOp`'s `storage_view`.

#### The same place was also a silent miscalculation

This was originally treated as a performance-only change, but measurement showed it fixes a **silent wrong result**. When a one-element expand goes through the storage descriptor, the tensor returned by `jt.zeros/ones/full` has that constant's four bytes as its storage and all-zero strides — its shape is `(6,4)` but its buffer is not. Jittor's own operators read the strides and cannot see the problem; **any external library that takes its pointer and writes by shape will be wrong**, and no error is reported.

`cusparse_spmmcsr` uses it exactly that way: `cusparseCreateDnMat(&matC, os[0], os[1], os[1], outputVar->ptr<T>(), ...)` writes it as a dense matrix. On the baseline, running `tests/backends/cuda/test_cusparse_{op,dtype}.py` alone gives **11 failed / 4 passed**; after the change, **15 passed**. Minimal reproduction (baseline tree):

| Source of `output` | Result |
| --- | --- |
| `jt.zeros((3,3))` | **MISMATCH**, reads back all 0 |
| `jt.array(np.zeros((3,3)))` | ok |

The Python-visible criterion is the numpy strides of `.data`: on the baseline `jt.zeros((4,3),'float32')` reports `(0, 0)` and `C_CONTIGUOUS=False`; after the change it is `(12, 4)` and contiguous. The regression test `tests/core/test_constant_tensor_storage.py` asserts this invariant directly (fails before the fix, passes after, in the same commit as the fix) and at the same time pins "a multi-element expand is still a view".

## Results

### Metaops (n=2^22 float32, µs/operator, lower is better)

Measured back to back on the same card. `jt 最终` = all three changes in effect.

| Case | jt baseline | jt final | Speedup | eager torch | torch.compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| unary_abs | 25.11 | 14.06 | 1.79x | 12.61 | 12.50 |
| unary_exp | 25.22 | 14.10 | 1.79x | 12.62 | 12.61 |
| binary_add | 27.71 | 17.92 | 1.55x | 17.29 | 17.13 |
| binary_mul | 27.72 | 18.05 | 1.54x | 17.17 | 17.20 |
| ternary | 82.85 | 21.74 | **3.81x** | 18.44 | 18.24 |
| fused_mul_add | 27.93 | 18.07 | 1.55x | 29.25 | 17.23 |
| fused_chain5 | 30.09 | 19.28 | 1.56x | 106.11 | 17.19 |
| reduce_all | 13.81 | 13.42 | 1.03x | 12.33 | 51.21 |
| reduce_lastaxis | 13.76 | 10.93 | 1.26x | 17.85 | 7.49 |
| reduce_firstaxis | 13.11 | 11.06 | 1.19x | 12.33 | 53.51 |
| bcast_row | 76.05 | 15.28 | **4.98x** | 23.35 | 40.89 |
| bcast_col | 75.42 | 15.40 | **4.90x** | 23.41 | 12.72 |
| strided_slice | 23.46 | 11.55 | 2.03x | 13.76 | 11.52 |

**The baseline's broadcast cases are bimodal**: across multiple runs of the same configuration `bcast_row` reported 32.8 µs and also 76.1 µs. The table takes the run from the same round as the other columns. Counted at 32.8, the speedups for these two rows are about 2.1x rather than 4.9x; after the change the bimodality no longer appears. Cite these two rows together with the convention.

**You must compare against `torch.compile`, not only against eager.** A large part of `fused_chain5`'s 5.5x over eager torch is just "Jittor fuses into 1 kernel while eager launches 5"; inductor fuses as well, and against it the numbers are 19.28 versus 17.19, with Jittor slightly slower. The comparison that really holds: after the change Jittor is no worse than eager torch on 7 of the 13 cases and within 15% on the rest; against torch.compile it wins some and loses some — torch.compile is faster on unary and last-axis reduction, while Jittor is faster on full reduction, first-axis reduction and row broadcast.

### Full network

Metaops are 1.5–3.8x faster, but **the full network is only 1.00–1.41x**, because MLP/CNN are dominated by cublas/cudnn. The Transformer gain grows as the model narrows (same-process A/B, `cuda_thread_num` 1024 versus 256):

| Model/mode | tn=1024 | tn=256 | Speedup |
| --- | ---: | ---: | ---: |
| mlp/infer | 2.141 ms | 2.139 | 1.00x |
| mlp/train | 6.617 | 6.562 | 1.01x |
| cnn/infer | 9.691 | 9.671 | 1.00x |
| cnn/train | 11.233 | 11.206 | 1.00x |
| transformer/infer | 8.884 | 7.848 | 1.13x |
| transformer/train | 27.231 | 23.636 | 1.15x |

| transformer width (4 layers, seq 256, training) | tn=1024 | tn=256 | Speedup |
| --- | ---: | ---: | ---: |
| d=256 | 13.714 ms | 9.759 | **1.41x** |
| d=512 | 27.234 | 23.630 | 1.15x |
| d=1024 | 31.544 | 30.629 | 1.03x |
| d=2048 (2 layers) | 61.257 | 58.041 | 1.06x |

The full-network gain of scalar fusion alone (the two trees alternating on the same card, taking each minimum) is **1.01–1.05x** (transformer training at 1.05x, the highest), and the launch-bound decode form (b1 s1 d512, 8 layers) goes 5.09 → 4.92 ms, **1.035x**. Launching 22% fewer kernels buys only a single-digit percentage, because launches are asynchronous and most are not on the critical path — this should be stated honestly.

## Why this was not found earlier: the 4090 conclusion does not transfer to H20

The existing skill `cuda-elementwise-bandwidth-roofline` concluded on an **RTX 4090** that "the elementwise class as a whole is already against the roofline (1086 GB/s, ratio 0.84), and the only way to go faster is to move fewer bytes". That conclusion is correct on its machine: the 4090 measures a copy bandwidth of 916.7 GB/s, memory access hits the limit first, and the launch configuration is not the bottleneck.

H20's bandwidth is about 4x that, yet the same kernel shapes reach only 1300–1800 GB/s, so **the launch configuration becomes the bottleneck that arrives first instead**. The statement "metaops are already against the roofline" is therefore **machine-dependent**, and changing a hardware generation requires re-measuring.

## Verification

- `tests/backends/cuda` + `tests/backends/parity/test_dtype_coverage.py`: the failing set is **identical nodeid by nodeid** before and after the patch (31 each, none added, none gone). The baseline and the patch tree ran independently, each with its own compile cache.
- Index specialization: a parity suite covering ranks 1–5, per-axis broadcast, 15 multi-axis broadcast masks, the all-zero mask, non-broadcast slices, transposes, ternary with each of its three operands strided, and backward; **the pure-index cases are bit-identical** (`exp` and summation order use a 1 ulp tolerance).
- The silent miscalculation of one-element expand: `tests/core/test_constant_tensor_storage.py` fails 11 subtests on the baseline tree and fully passes on the patch tree; `tests/backends/cuda/test_cusparse_{op,dtype}.py` run alone goes from 11 failed/4 passed to 15 passed.
- `tests/backends/cuda` (patch tree 21 failed / 284 passed) compared nodeid by nodeid with the baseline (31 failed / 279 passed): **no new failures**, and the 11 cusparse cases above went from failing to passing.
- One-element expand: all 33 targeted parity cases pass, covering scalar constants in each arithmetic position, computed one-element sources (reduction feedback, softmax form), consumers that are not elementwise operators (reduce/matmul/getitem/transpose/direct read), multi-element expand still being a view, four dtypes, ranks 1–5, and three kinds of gradient.
- **Per-operator CPU/CUDA parity** (`tests/backends/parity/test_device_parity.py` and `tests/ops/test_ops.py`, about 227 generated cases, Torch compatibility mode): the patch tree for changes 1 and 2 and the baseline tree ran independently, each with its own compile cache, and the results are **exactly identical at `121 failed, 654 passed, 9 skipped`, with zero nodeid-level difference across the 121 failures** (patch 7629 s, baseline 7504 s). Both crashed at the same point on session exit (`corrupted double-linked list`), identical before and after the patch; it is a pre-existing problem and out of scope for this round. The same gate for change 3 is running, with the conclusion attached separately.
- **Per-operator parity, change 3**: likewise `121 failed, 654 passed, 9 skipped`, with **zero nodeid-level difference** from the baseline. Each of the three changes ran this gate independently, and all results match the baseline.
- **Native CPU suite** (`tests/core` `ops` `nn` `autograd` `mem` `type` `linalg`, excluding three torch-mode files): patch tree `1499 failed / 978 passed / 45 errors`, baseline `1506 failed / 968 passed / 47 errors`. **The absolute numbers are not a gate conclusion** — this is a directory combination I picked by hand, not the `tools/run_test_suite.py` maintenance baseline, and both sides have many failures; the only meaningful thing is the nodeid-level difference set (both sides ran in parallel on the same machine under the same load):

  - **1 new failure**: `tests/ops/test_broadcast_index.py::test_a_pure_broadcast_is_a_strided_view`. Deserved — see below.
  - **6 went from failing to passing**: `test_scalar_fuse_unary` (including cuda), `test_fuse_reduce2`, `test_setitem_`, `test_fuse_transpose1`/`2`. Re-running them separately confirms: baseline 5 failed, this tree 5 passed. **These five were pre-existing red lights left when the refactor broke scalar fusion**, and this round fixes them back.

  Together with the 11 cusparse cases on the CUDA side, that is **16 pre-existing failures turned passing and 1 new failure**.

### That one new failure is deserved, and how it was handled

`tests/ops/test_broadcast_index.py` guards KI-OPS-009 (a strided indexing Var must be read by stride, otherwise it reads out of bounds into its neighbors). It constructs that strided index with `jt.zeros(shape)` — which is **built on top of the very defect this round fixes**: the buffer is not its shape. After the change this construction no longer produces a view, and every case in the file would pass without triggering anything. Its own docstring says "this is the premise. If it no longer holds, this file no longer tests anything" — the guard worked.

The handling was to replace the premise with an expand that has a **reason** to be described as a view: expand a row of zeros rather than a single zero. The strides change from `[0,0]` to `[0,1]`, and the dense walk's out-of-bounds amount changes from nineteen twentieths to three quarters; it still goes out of bounds and the logic of each assertion is unchanged. `test_a_broadcast_index_reads_its_own_element` is changed to expand `base[0:5]`, so the dense walk gives each row a different pattern while the correct read is the same for every row, still independent of heap state. One more case, `test_a_one_element_expand_is_not_a_view`, is added to pin the new behavior.

**The rebuilt file on the baseline tree gives 12 passed / 2 failed**; the failures are exactly the newly added guard, and the remaining KI-OPS-009 cases pass as usual — showing that they do not depend on this round's change and the file's purpose is preserved.

- **Not run**: the full CPU maintenance baseline of `tools/run_test_suite.py`, ROCm, NPU. All performance conclusions in this round were obtained only on H20 (sm_90); the launch-configuration defaults are **unverified** on sm_89 and earlier cards (see the "Why this was not found earlier" section: on a 4090 this class of kernel is already against the memory limit, so approximately no change is expected, but it was not measured).

### A false failure I caused myself, recorded to avoid repeating it

In the first round the `tests/backends/cuda` patch tree had 2 more failures than the baseline, both in `test_softmax_cuda_grad.py`, reporting `Jit var YSMASK not found`. The cause was not the change: **I modified `src/ops/binary_op.cc` while the gate was still running**. `op_compiler` reads the `.cc` source file directly at JIT time, so the already-loaded core (old `jit_prepare`, with no YSMASK in the key) was paired with the new template. Section 0 of the plan document states "do not modify src while a gate is running"; this rule exists for exactly that. Later changes were made in **another work tree**, and the two failures disappeared after re-running.

## Located but not implemented

By the **graph-building** cost of an operator (no synchronization, only nodes, µs/operator):

| Operator | jittor | eager torch | Ratio |
| --- | ---: | ---: | ---: |
| binary Var+Var | 5.31 | 4.33 | 1.23x |
| **binary Var×scalar** | **10.31** | 5.16 | **2.00x** |
| unary abs | 4.75 | 4.80 | 0.99x |
| **reshape** | 3.38 | 0.72 | **4.7x** |
| **reduce sum** | 11.86 | 5.06 | **2.34x** |
| ternary (with comparison) | 7.35 | 9.55 | 0.77x |

1. **`Var × 标量` builds three graph nodes** (array + broadcast_to + binary), while torch dispatches once. Item 3 of this report removed the redundant *kernel*, but the nodes remain. Measured piecewise: binary itself 5.32 µs, the expand node +1.99 µs, creating the array node +2.91 µs. Removing it requires giving BinaryOp an immediate operand, which touches the JIT key, backward (the scalar's gradient is a full reduction, currently handled by expand's backward) and every backend — an independent large change, not done this round. **One rejection recorded along the way**: I considered caching a one-element constant Var to save the array node, but that would make it a "batch input", and `count_fuse`'s first rule is that batch inputs are never fused — which exactly cancels item 3.
2. **The launch-bound form's host path is about 2.9x slower, and the cause is the node count rather than per-node cost.** b1 s1 d512, 8 layers: jittor's `jt.no_grad()` step is **4.13 ms** (4.934 ms without it), and eager torch with the same structure under `torch.no_grad()` is **1.43 ms**. (The previously written 3.5x compared jittor with autograd enabled against torch under no_grad — a mismatched convention, now corrected.) Graph building accounts for **96.9%** of a step. Broken down item by item:

   | | jittor | eager torch |
   | --- | ---: | ---: |
   | Nodes/dispatches per step | **408** | about 318 |
   | Cost per node in context | about 12 µs | about 4.5 µs |
   | Cost of a single operator measured in isolation | 5.31 µs (binary) | 4.33 µs |

   Composition of the 408 nodes of one step (`jt.dump_all_graphs()`, including autograd):

   | Type | Count | Share | | Type | Count | Share |
   | --- | ---: | ---: | --- | --- | ---: | ---: |
   | binary | 88 | 21.6% | | array | 32 | 7.8% |
   | **reshape** | **80** | **19.6%** | | tape/tapes | 48 | 11.8% |
   | broadcast_to | 64 | 15.7% | | getitem | 24 | 5.9% |
   | cublas_matmul | 32 | 7.8% | | fuse_transpose | 16 | 3.9% |

   Two actionable conclusions:

   - **A pure view also pays a full graph node.** `reshape` accounts for 19.6%, yet it `is_storage_view()`, has no kernel, and does almost nothing beyond `infer_shape`; building it still costs 3.38 µs (torch 0.72 µs), and the cost is entirely in the generic node-creation machinery (Op node, Var node, edges, VarHolder). Wiping it out entirely would require making a view a property of Var rather than an Op — a fundamental change to the graph model.

     But **64 of these 80 come from one place** and can be handled separately: `nn/functional/matrix.py`'s `matmul` takes `len_b == 2 and len_a > 2` (that is, every `nn.Linear` with a 3-D input) through `aa = a.reshape((-1, m))` → matmul → `cc.reshape(a.shape[:-1] + [k])`, which is **two reshape nodes** per Linear. In this test case 4 Linears × 8 layers = 64, plus the 2 × 8 = 16 written explicitly in the model, is exactly 80. The branch carries its own comment `TODO:ugly implementation for tuner`. Since the input is already contiguous, this flattening is a pure view; letting `cublas_matmul` take `prod(shape[:-1])` as the row count itself would remove both reshapes — about 15.7% of a step's nodes (about 218 µs / 4130 µs = 5.3% at 3.38 µs each). Not done: matmul is one of the hottest paths and the one that can least afford errors, and changing it would require a full parity suite for shapes and backward, beyond what this round could verify.
   - The gap between 12 µs in context and 5.31 µs in isolation comes from the Python dispatch layer: cProfile shows `_runtime/dispatch.py`'s `select_kernel` is called about 104 times per step and accounts for about 14% of graph building cumulatively. That file was already optimized in task 3.21 with measurements recorded, so it was not touched this round.
3. The graph-building cost of `reduce sum` is 11.86 µs versus torch's 5.06 µs: **located, and the fix was rejected.** `a.sum()` builds only one node (the same as `a+a`), and the extra ~6.5 µs is entirely in the Python dispatch layer — cProfile shows `select_kernel` plus the `supports` callback at about 9.7 µs, while `_supports_full_reduce`'s criterion only looks at `x` (fewer elements than one block should not take the fast path) yet is ordered **after** `_dispatch_placement` and dtype-name construction.

   Moving the criterion earlier and writing it cheaply at the same time (the original builds a tuple, runs an `any()` generator, then loops to compute the product; a single pass suffices, and a negative dynamic dimension still needs separate rejection) measured:

   | | As-is | Criterion moved earlier |
   | --- | ---: | ---: |
   | Small tensor (512, criterion rejects) | 12.20 µs | 8.36 µs (**1.46x**) |
   | Large tensor (131072, criterion accepts) | 35.64 µs | 37.70 µs (**0.95x**) |

   **Not adopted**, for two reasons: on the large-tensor side the criterion gets computed twice and becomes slower, and a regression cannot be introduced for one gain; more importantly, `reduce_entry` sends any reduction with an axis (`_is_full_reduction` false) straight down the native path without entering dispatch at all, so the only ones that actually pay this cost are **full reductions** — softmax and LayerNorm both use axis reductions, which almost never occur in a transformer step. The gain targets a rare call.

4. **Reduction is about 6x off bandwidth, and there is another small located defect.** A transformer step launches only one kind of generated reduction kernel (`DIM=3 REDUCE=3 add`, i.e. Linear's bias gradient; LayerNorm and softmax go through hand-written CUDA), and it reaches only **386 GB/s**, versus 2800 GB/s for elementwise on the same card. The 6x gap is **not located**.

   A small one was located along the way: the generated warp folding only applies when every lane of a warp holds the same output index, and when **the kept axis is the last dimension** that condition is always false, so the kernel falls back to one global atomic per thread and reports no exception. The way to tell is to turn on `no_warp_reduce` and look at the ratio. Same extent, same reduction amount, changing only which dimension is kept:

   | Shape | Reduced dims | warp on | warp off | off/on |
   | --- | --- | ---: | ---: | ---: |
   | `[8,256,512]` keep last dim | 0,1 | 10.86 µs | 10.78 | **0.99** |
   | `[8,512,256]` keep middle dim | 0,2 | 9.59 | 34.13 | 3.56 |
   | `[512,8,256]` keep first dim | 1,2 | 9.64 | 35.87 | 3.72 |

   The cause is in `AtomicTunerPass`: it identifies "the atomic dimension" by the largest loop id, and when the kept axis is the last dimension that id is the largest and is moved to the front after reordering, so `max_pos == 0` causes the whole thread-count correction to be skipped; taking `loop_orders[fi].back()` by the post-reorder position matches what its own comment says. **But this is worth only about 12%** (10.86 versus 9.6 µs), it is not the cause of the 6x, and fixing it will not close the main gap. The discriminating method and the measurements have been written into the `cuda-reduction-strategy-comparison` skill.
5. **`para_opt_level=4` (in-block shared-memory reduction) is likewise worse on H20**, consistent with that skill's conclusion on the 4090 — this item **does** transfer across architectures: the transformer shape goes 29.78 → 49.42 µs (1.66x slower) and the last-dimension shape 44.34 → 193.18 µs (4.36x slower); accuracy is indeed better (1.4e-7 versus 6.8e-7). The default needs no change. (Note that the skill's `reduce_ab.py` reads the profiler's MinTime, which on a card-occupied machine reports every line as that 2.3 ms stall; this round re-measured with the slope method.)

Also, vectorized memory access (float4) is 1.31–1.52x over scalar access in an independent microbenchmark, but after the changes Jittor's elementwise already reaches 2800 GB/s (eager torch 2911, torch.compile 2938), so the headroom is small and it was not implemented.

## Reproduction

```bash
export JITTOR_LAB_ROOT=<lab-root>
source $JITTOR_LAB_ROOT/_state/metaop/env-scalar.sh
cd $WT
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/bench3.py --n $((1<<22))
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/check_strided.py
CUDA_VISIBLE_DEVICES=<一张卡> $PY $JITTOR_LAB_ROOT/metaop-perf/check_scalar.py
```

Comparison convention: `bench3_torch.py` (eager) and `bench3_torch_compile.py` (inductor) run the same set of cases with the same slope method in an interpreter with real PyTorch installed. The raw logs, compile caches and work trees are under `$JITTOR_LAB_ROOT/_state/metaop/` and are not version-controlled.
