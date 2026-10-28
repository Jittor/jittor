# Host-bound step: removing per-operator Python overhead from graph construction

- Status: the Python path is complete; the C++ side's per-node cost and `perf/matmul-rank` are still under investigation; this stack of changes has an unfixed device-level fault under a cold reference cache (see "Unresolved" at the end)
- Date: 2026-09-13
- Baseline commit: `48055a75` (branch `perf/host-path`); the baseline underneath it is `d40a2e97`
- Verification scope: a single CUDA card, the card model was not recorded in the original text (the same baseline's [meta-operator report](2026-09-12-cuda-metaop-launch-index-scalar.en.md) records H20); graph-construction microbenchmarks are measured with `auto_flush_ops=0`, and end-to-end decode/prefill/b8s256 with the default `auto_flush_ops=128`; selected files of `tests/runtime`, `tests/ops`, `tests/nn`, plus the full `tests/autograd`, are compared nodeid by nodeid against the branch starting point
- Maintainer: Jittor core maintainers (meta-operator/computation-graph performance)
- Revisit when: another layer of dispatch is added to the frontend, or the default of `auto_flush_ops` changes

## Problem

Decode at batch 1 is **host-bound**: constructing 354 nodes in one step takes 2.99 ms, while a bare binary meta-operator takes only 1.60 us. 354 × 1.60 us = 0.57 ms, which is to say that 2.4 ms of one step is not building nodes but Python on top of the nodes.

First confirm this is not caused by the graph getting larger: chain K binaries into one unexecuted chain, and measure the construction time per operator for each K; from K=1 to K=1024 it stays stable at 1.57–1.63 us, with a release time of 0.12 us/operator. **The cost of building a node is independent of the size of the graph to be executed**, so the entire 8x gap is in the frontend.

## Measurement method (including a trap that would completely invalidate the conclusion)

`jt.flags.auto_flush_ops` defaults to 128 on CUDA: every time that many operators have been created, the executor **walks all live VarHolders** and launches everything pending. So

- a timing segment that "only builds the graph and does not synchronize" actually mixes in kernel launches;
- and that walk is O(number of live holders); in a microbenchmark like `out = [fn() for _ in range(N)]` that keeps N results in a list, the cost grows with N — the measured "microseconds per call" changes with N.

Measured in the same batch, with `auto_flush_ops` on, Linear measures 34.7 us and LayerNorm 55.5 us; with it off, 20.8 us and 39.2 us. All graph-construction numbers in this document are measured with `auto_flush_ops=0` and `use_cuda=1` (dispatch follows the Var's placement, so it must be CUDA), and end-to-end numbers are measured with the default value.

Noise control: running the baseline twice in a row, the micro rows differ by ≤0.5% and the step/block rows by ≤3%.

## Where one block's host time goes (before the changes, 370 us)

| Item | Per call | Count | Share |
| --- | --- | --- | --- |
| Linear(512→512/1536) | 21.7 us | 4 | 23% |
| LayerNorm | 36.9 us | 2 | 20% |
| gelu | 27.0 us | 1 | 7% |
| softmax | 23.1 us | 1 | 6% |
| batched matmul | 10.2 us | 2 | 6% |
| transpose 5d | 10.4 us | 2 | 6% |
| getitem | 4.2 us | 3 | 3% |
| Other (reshape, residual, scaling) | | | 4% |
| Not counted (the block's own Python + cache warming from isolated measurement) | | | 25% |

Break the 20.8 us of Linear down further (use_cuda=1, auto_flush off):

| Component | us |
| --- | --- |
| the cublas operator itself | 1.24 |
| select_kernel and its surrounding frames | 4.14 |
| the reshape pair (3d→2d→3d) | 5.79 |
| bias addition (two nodes: broadcast + binary) | 5.31 |
| `_check_matmul_shapes` | 1.64 |
| `Module.__call__` | 2.64 |

That is: the real operator accounts for 6%, and the remaining 94% is frontend and extra nodes.

## What changed

See commit `Host path: removing per-operator Python overhead from graph construction in ten places`. Ten places, ordered by benefit:

1. **backend_libraries memo**: each matrix multiply asks `cublas` twice (once to decide whether dispatch is possible, once for the actual call), and the dispatch layer asks once more per candidate; this used to mean three function frames, one RLock, and three dictionary lookups each time. Names with an `enabled` policy (only mkl, which reads module globals and environment variables) never enter the memo, and any registration clears the memo.
2. **transpose's integer-permutation fast path**: `numbers.Integral` is an ABC, so every axis used to go through `ABCMeta.__instancecheck__`, and then a dict was built to check for duplicates. 10.4 → 6.2 us.
3. **dispatch**: fetch `jittor.core` / `core.Var` / `core.dispatch_context` only once; inline `_canonical_backend` and the read of the resolved candidate table; container arguments (shape, permutation, dim list) go through a flat traversal that does not build a cycle-detection set; dtype filtering uses frozenset's `issuperset`; the `optional` wrapper inlines `try_dispatch`.
4. **`_check_matmul_shapes`**: when any operand has ≤2 dimensions the batch-dimension loop can never execute, but the old code still built two slices, two reversed, one zip, and one enumerate before discovering this. `nn.Linear` is exactly this shape.
5. **Function**: a Function is not a parameter container (`dfs` is an empty implementation here), so the parameter classification `Module.__setattr__` does for every Var attribute is dead work, and `self.saved = x, mean, rstd` is exactly how fused layer_norm/softmax saves backward inputs; `Sequence` is an ABC, so the two isinstance checks look at the exact type first; with no keyword arguments, `_reject_var_keywords` is not entered.
6. **indexing**: the `jittor` module is bound only once (a single `x[i]` used to run four IMPORT_NAMEs), and `_is_plain_int` returns immediately for an exact int. 4.20 → 3.24 us.
7. **per-call rebuilding in layer_norm / gelu**: `normalized_shape` is no longer rebuilt once for each of the two relays, and `dims` is built only when the generic path is actually taken; `_jittor_dtype_name` used to be called three times, two of which operated on the *string* it itself returns; gelu's three numpy scalar constants keep only two sets according to the compute dtype.

## Results

Graph construction (us/call, use_cuda=1, auto_flush_ops=0):

| Row | Before | After | Speedup |
| --- | --- | --- | --- |
| one decode step (8 layers) | 2991.4 | 2516.8 | **1.19** |
| one block | 370.4 | 311.0 | **1.19** |
| transpose 5d | 10.38 | 6.20 | 1.67 |
| matmul_transpose (2d) | 7.18 | 5.03 | 1.43 |
| getitem | 4.20 | 3.24 | 1.30 |
| softmax | 23.05 | 19.12 | 1.21 |
| layer_norm | 34.56 | 29.33 | 1.18 |
| Linear | 21.68 | 19.54 | 1.11 |
| gelu | 26.97 | 25.34 | 1.06 |
| *reshape 5d (control)* | 2.30 | 2.34 | 0.98 |
| *x + x (control)* | 1.599 | 1.607 | 1.00 |
| *x \* 0.5 (control)* | 5.358 | 5.360 | 1.00 |

The last three rows are controls: these paths were not touched, and their not moving is what shows the changes above are not noise.

End-to-end (GPU, ms/step, including execution, default `auto_flush_ops=128`). The first round was measured while the card was occupied by someone else, and the two device-bound rows are completely unusable (the same configuration measured 15.1 and 22.5 ms in two runs); later, after the card freed up, everything was measured again, all four rows are usable, and `metaop-cpp` was measured twice in a row to confirm reproducibility (3.512 / 3.514, a 0.06% difference):

| Scenario | Starting point 48055a75 | + Python changes | + one-sided broadcast | Cumulative |
| --- | --- | --- | --- | --- |
| decode b1s1 d512 L8 inference | 4.128 | 3.641 | **3.512** | **1.18x** |
| decode b1s1 d512 L8 training | 10.589 | 10.034 | 9.805 | 1.08x |
| prefill b1s128 inference | 4.325 | 3.960 | 3.708 | **1.17x** |
| *batch b8s256 inference (control)* | 15.126 | 15.106 | 15.060 | **1.00x** |

The last row is a control, and also the row most worth looking at this round: **the device-bound scenario has not changed at all**. Changes to the host path should only help when the host is the bottleneck, and this row proves that elsewhere it is neither faster nor slower. The training row is only 1.08x, because the time of `jt.grad` and the optimizer within a step is not on this path.

## Verification

`tests/runtime/{python_dispatch,dispatch_context,domain_dispatch,native_op_dispatch,capability_queries,native_backend_registry}`, `tests/ops/{matmul,matmul_dispatch,transpose_op,slice,index_op,index_bounds,gather_index_contract}`, `tests/nn/{linear,norm,normalization_dispatch,acl_registry_routing}`, and the full `tests/autograd`, compared nodeid by nodeid against the branch starting point `48055a75`: both sides are 22 failed / 300 passed / 81 subtests, and **the failure sets are exactly the same**. Those 22 were already red at the branch starting point.

The first time this comparison was run, the baseline side collected 0 tests and exited — `nvcc` could not write to a `TMPDIR` that nobody had created, and the script still printed a "regression" table as usual, listing in fact all 22 failures of the tested side. **A diff with one side empty looks exactly like "everything regressed"**, so the summary now prints the passed/failed counts of both sides.

## Remaining headroom

By the same measurement, after the changes a block is still 311 us / about 44 nodes = 7.1 us/node, while a bare operator is 1.6 us. The remaining items, ordered by size:

1. **The fixed 1.6 us per node**. `x * 0.5` is three nodes (array + broadcast + binary), and `x + bias` is two; `BinaryOp`'s broadcast branch also builds a `BroadcastToOp` for each of the two operands, one of which merely forwards the input unchanged, then builds a second `BinaryOp` and forwards to it — a single scalar multiplication constructs 5 operator objects to leave 3 nodes. To go faster, the only way is to **build fewer nodes**, not to make node construction cheaper.
2. **matmul's reshape pair** (5.8 us per batched `nn.Linear`, 23 us per block, 7%). `perf/matmul-rank` lets cublas take rank>2 directly, which is exactly removing this pair; see below.
3. **Function's tape recording**: in one layer_norm, tape recording accounts for 11.8 us (one tape node each for 3 inputs + 1 output, plus `tape_together`), 35% of its 33.5 us. On the inference side `jt.no_grad()` already eliminates this part entirely (19.9 → 8.1 us).
4. **Beyond that are "replay"-level changes**: for steps of the same shape that repeatedly run the same piece of Python, record the operator sequence once and replay it on the C++ side, which is the only way to approach the 0.57 ms floor. This is the amount of work in the torch.compile / CUDA Graph league, and was not done this round.

## Building one fewer node: BinaryOp's one-sided broadcast

The cheapest item in "Remaining headroom" item 1 has been done. `BinaryOp`'s broadcast branch used to build a `BroadcastToOp` for **each** of the two operands, while `BroadcastToOp(x, y, {})` itself asks `need_broadcast`, and if the answer is no it forwards the input unchanged — that is, for every `x * 0.5`, every `x + bias`, every `x * mask`, there was always one operand whose `BroadcastToOp` was purely built and thrown away. Changed to first ask the same predicate and build only for the side that actually needs it:

```cpp
VarPtr xh, yh;
Var* xp = x;
Var* yp = y;
if (y->num < 0 || BroadcastToOp::need_broadcast(x, y->shape)) { xh = ...; xp = xh; }
if (x->num < 0 || BroadcastToOp::need_broadcast(y, x->shape)) { yh = ...; yp = yh; }
```

The `y->num < 0` half cannot be omitted: for an operand whose shape is undetermined, the `BroadcastToOp` constructor would not forward anyway, so here it must also be built as before.

This is not a new shape: the `TernaryOp` next to it in the same file is already written this way — `if (bx2) cc = make_broadcast(...)`, `if (bx) xx = make_broadcast(...)`, judging each of the four directions once and building only the one that is needed. `BinaryOp` is the only one of these three meta-operators that unconditionally builds two.

Measured (same machine, same card, measured once before and once after):

| Row | Without this change | With this change | Speedup |
| --- | --- | --- | --- |
| `x + bias` (broadcast) | 4.073 | 3.470 | **1.17** |
| `x * 0.5` (scalar) | 5.350 | 4.697 | **1.14** |
| gelu(2048) (four scalar binaries) | 25.33 | 22.74 | 1.11 |
| one decode step (8 layers) | 2532.6 | 2377.5 | **1.07** |
| one block | 310.4 | 299.0 | 1.04 |
| *`x + x` (control)* | 1.595 | 1.622 | 0.98 |
| *getitem (control)* | 3.175 | 3.251 | 0.98 |
| *reshape (control)* | 2.307 | 2.237 | 1.03 |

Together with the earlier Python changes, graph construction for one decode step drops from 2991 us to 2378 us, **1.26x**.

## The state of `perf/matmul-rank`: 671 failures have narrowed down to one

It is no longer "six hundred-odd mysterious failures". Tracing back from the first **extra** failure:

- That one is `test_device_parity.py::TestDeviceParity::test_inner`, **which reproduces when run alone**, failing 8 out of 8 times (13 out of the earliest 15; the first two passed because the relevant kernel had not been compiled yet at that point). The six hundred-odd that follow are collateral: one failure corrupts the CUDA context, and everything else goes red along with it.
- **It blows up in backward, not forward**: `jt.grad(loss, diff)` → `sfrl_allocator.cc:305: mem_ptr does not belong to allocation`. The forward values are correct.
- **It is unrelated to the rank>2 fast path**: `sample_inner` feeds A:(3,4), B:(2,4), both rank 2, so `len_b == 2 and len_a > 2` simply does not hold.
- A three-way comparison pins it down:

  | Tree | Contents | `test_inner` |
  | --- | --- | --- |
  | `d40a2e97` | unchanged baseline | pass |
  | `perf/matmul-on-baseline` (`a03c88ef`) | baseline + matmul changes | pass (the whole set 121, same as baseline nodeid by nodeid) |
  | `perf/host-path` (`6ae82f2b`) | baseline + launch configuration / strided indices / scalar fusion | pass (it is test number 115, within the 172 reached in that run, character-for-character identical to the baseline) |
  | `perf/matmul-rank` (`e426abf8`) | has **both** of the above stacks | **fails 8/8** |

  That is to say: **each of the two stacks of changes is clean on its own, and only the combination blows up**, and the trigger is a rank-2 matrix multiply.
- What remains unresolved: writing the same shapes, same gradients, same cotangent into a script without the test framework, **it passes** (forward and backward for rank2/rank3/rank4, `matmul_transpose`, `matmul(a, b.transpose(-1,-2))` all correct). So the trigger still needs some state inside the framework, and the next step is to instrument inside the framework rather than continue assembling reproductions from outside.

**Conclusion: this does not land.** It is worth 7% of a block (removing a reshape pair per batched `nn.Linear`), but it cannot be merged with a reproducible allocator-invariant crash. The table above and the minimal entry point `test_inner` are the few hours saved for whoever picks this up next.


## Unresolved: this stack of changes itself has a device-level fault under a cold reference cache

**This one matters more than all the speedups above, so it is written down first.**

Symptom: when the **reference cache is cold**, test number 6, `test_affine_grid`, in `tests/backends/parity/test_device_parity.py` dies on a device-level `cudaErrorIllegalAddress`, after which the whole file goes red (241 failed / 5 passed, about 11 minutes). The unchanged baseline does not.

Why it only triggers with a cold reference cache: this file's CPU reference values are computed on the spot by `_cpu_oracle` in **the same process** using `_run(op, sample, use_cuda=0)`, and once computed they are written to disk. So in a round with a cold cache, every test first builds the graph on CPU and then builds it on CUDA; once the cache is warm, the CPU half does not run at all. And **`source_fingerprint()` covers the entire `python/jittor`** — any one-line Python change invalidates all references, so the first run after every change is necessarily cold.

Control (all fresh caches, cold reference cache, six tests are enough to see it):

| Tree | Card | Test number 6 `test_affine_grid` |
| --- | --- | --- |
| `d40a2e97` unchanged baseline | card A | pass (the whole file 11 failed, a known red light) |
| `d40a2e97` unchanged baseline | card B | pass (`......FF..FFFF` is exactly the baseline pattern) |
| any tree with this stack (`perf/host-path`, `perf/metaop-broadcast`) | card A / card B | **fails**, everything afterwards collateral |

So it is neither the card nor the gate framework, **it is the tree**. And it predates tonight's work: tonight's two commits are each clean nodeid by nodeid relative to their own branch starting point, but **the branch starting point itself is not clean**.

Ruled out:
- Not JIT cache corruption — it still reproduces after a fresh cache and serial warmup;
- Not the matmul changes in `perf/matmul-rank` — a tree without that change blows up the same way, with the same `cudaErrorIllegalAddress` and the same string of launch candidates (`setitem id=30`, `cublas_batched_matmul id=43`, `fused_op fused_ids=[10,2]`);
- Not the fact of "CPU first, then CUDA, in one process" itself — writing broadcast binary, strided reads, reduce, and matmul into a script that switches back and forth for 40 rounds, both trees are clean. So the trigger still needs some specific operator among the first six tests.

**It has been narrowed down to a commit.** A four-point comparison, each with a fresh cache, cold reference cache, running only the first six tests:

| Point | Commit | Test number 6 `test_affine_grid` |
| --- | --- | --- |
| base | `d40a2e97` unchanged baseline | 6 passed |
| launch | `9a60c3e9` launch configuration + strided index specialization | 6 passed |
| **scalar** | `9e17002d` (including `2fc3837d` one-element expand) | **signal 11, `(no mapping)`** |

**The cause is in `2fc3837d`** — the one "one-element expand keeps the fusible form": it changes `broadcast_to`'s `is_storage_view()` from true to false when `x->num == 1`. What `test_affine_grid` builds is exactly `jt.zeros` / index-grid style one-element expands. The launch-configuration and strided-index commit is clean.

**Next step**: after `2fc3837d`, which place still assumes "a broadcast is always a storage view" (or the converse). `test_affine_grid` is the minimal entry point, and six tests reproduce it in ten minutes.

**Impact on tonight's results**: the performance numbers are unaffected (they are all graph-construction and end-to-end timings that do not depend on this path), and the gating of the two commits relative to their own starting points is also unaffected. But **this stack should not be merged until this fault is located and fixed**.
