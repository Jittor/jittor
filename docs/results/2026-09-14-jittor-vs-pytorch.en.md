# Jittor vs real PyTorch 2.9.1: where the gap is now

- Status: a snapshot of 2026-09-14, not updated on this page since. Native Jittor vs eager PyTorch: 6 wins, 5 ties, 1 loss (at this round's starting point it was 2 wins, 1 tie, 9 losses); the only loss is prefill training (alternating-pair median ratio 1.021), caused by the pipeline not being filled, not yet understood. Against the best `torch.compile` mode, after correction it is 2 wins, 2 ties, 5 losses (9 of 12 rows had both arms in a consistent state and are judgeable). CUDA Graph replay of a full training step was only manually verified this round
- Date: 2026-09-14; compressed and archived 2026-10
- Baseline commit: branch `perf/metaop-broadcast` (based on `d40a2e97`); the conclusion of the last update is `743fbe09`
- Verification scope: a single CUDA card, the card model was not recorded in the original text; the same `metaop` lab, same baseline's [meta-operator report](2026-09-12-cuda-metaop-launch-index-scalar.en.md) records H20 (sm_90). The comparison is PyTorch 2.9.1 + CUDA 12.9, neither side uses TF32 (`allow_tf32=False`). Six transformer / MLP / CNN models and shapes × inference/training; other tenants on the same machine occupy the entire GPU, so only ratios measured back to back are quoted. Not covered: CPU, ROCm, NPU, multi-card
- Maintainer: Jittor core maintainers
- Revisit when: graph construction becomes replayable, the criterion of `auto_flush_bytes` changes, or the executor launch path or CUDA stream semantics change

This page is compressed from about 900 lines of daily records: the intermediate methods, retracted conclusions, and raw readings are all in the Git history (`git log -p -- docs/results/2026-09-14-jittor-vs-pytorch.md`). The lab scripts (`alt_pairs.sh`, `run_gate2.sh`, `vs_torch_pt.py`) are in `$JITTOR_LAB_ROOT/metaop-perf/` and do not enter the main repository.

## How it was measured

- **Each case in its own process, both sides alternating on the same card, taking the minimum over multiple rounds.** When running all cases in one process, allocator state left by earlier cases makes the same b8s256 training measure 44.6 and 20.8 ms.
- **Warm up by whole segments, and warm up enough.** b8s256 training takes about 7 segments (35 steps) to reach steady state; warming only 4 segments leaves it on the 42 ms plateau; both sides were changed to warm 10 segments first. torch is nearly stable from the start, which is itself a user-perceivable difference (cause not investigated).
- **The synchronization discipline is the same on both sides**: inference is always under `no_grad`, and both synchronize only once at the end of a segment of `steps`.
- **Labels are matched as whole words**: `"b1s1"` is a substring of `"b1s128"`, which once made prefill use decode's step count.
- **Boundary cases use the median ratio of five alternating pairs**: this machine's absolute values can jump 15% between two runs.

## Results

### Against eager PyTorch (ms, ratio = jittor / torch, less than 1 means jittor is faster)

| case | jittor | pt eager | pt compile | vs eager |
| --- | --- | --- | --- | --- |
| tf b8s256 training | 19.57 | 40.14 | 39.26 | **0.49** |
| cnn-6conv inference | 8.93 | 14.12 | 13.92 | **0.63** |
| tf d1024 decode inference | 0.44 | 0.68 | 0.58 | **0.65** |
| tf decode b1s1 inference | 1.19 | 1.43 | 0.83 | **0.83** |
| cnn-6conv training | 23.99 | 28.23 | 27.74 | **0.85** |
| mlp-1024x4 inference | 0.37 | 0.40 | 0.38 | **0.93** |
| mlp-1024x4 training | 1.17 | 1.20 | 1.23 | 0.97 tie |
| tf b8s256 inference | 13.17 | 13.41 | 13.00 | 0.98 tie |
| tf decode b1s1 training | 6.64 | 6.59 | 4.65 | 1.01 tie |
| tf d1024 decode training | 3.53 | 3.49 | 3.55 | 1.01 tie |
| tf prefill b1s128 inference | 2.86 | 2.82 | 1.78 | 1.02 tie |
| tf prefill b1s128 training | 8.02 | 7.58 | 7.57 | 1.06 |

b8s256 training jumps between two states, 19.6 / 42.9 (insufficient warmup, see above), and this round landed on the fast half. On the torch side, prefill training swings between 7.26–8.64 (19%), so a single table cannot give a verdict; defer to the pairs below.

The six boundary rows, the median ratio of five rounds of torch/jittor alternating pairs (`alt_pairs.sh`):

| case | Median ratio | Range | Verdict |
| --- | --- | --- | --- |
| tf-d1024-L4 decode inference | 0.766 | 0.756–0.768 | win |
| tf-d1024-L4 decode training | 1.012 | 0.892–1.180 | tie |
| tf-d512-L8 decode inference | 0.393 | 0.389–0.403 | win |
| tf-d512-L8 decode training | 0.997 | 0.991–1.134 | tie |
| tf-d512-L8 prefill inference | 0.975 | 0.968–0.986 | win |
| tf-d512-L8 prefill training | 1.021 | 1.016–1.106 | **loss** |

The two decode training rows were previously 1.17 / 1.15; the general benefit of per-thread streams (per-operator floor 7.89 → 4.31 us) pulled them level; the advantage on the inference side comes from CUDA Graph replay.

### Against `torch.compile` (after correction)

The earlier compile comparison had two harness errors, both favoring jittor: the compile side was written as `zero_grad(set_to_none=False)` (which is also the real cause of "`reduce-overhead` fails on all training cases", and the claim that "jittor replay does not have this failure mode" has been retracted), and the compile numbers were measured in a separate pass spanning two of the machine's performance states. After re-measuring the four arms alternately within the same case:

| case | vs the best compile |
| --- | --- |
| cnn-6conv inference / training | **0.67 / 0.92 win** |
| mlp-1024x4 inference / training | 0.97 / 0.99 tie |
| tf decode b1s1 inference | 1.42 loss |
| tf prefill b1s128 training | 1.42 loss |
| tf d1024 decode training | 1.58 loss |
| tf prefill b1s128 inference | 1.61 loss |
| tf decode b1s1 training | 3.01 loss (6.62 vs 2.20) |

compile is strong at host-bound transformer small steps; jittor is strong where the work is device-bound and compile cannot help torch.

## Changes landed this round (all general changes; users do not need to change code)

| Change | Measured |
| --- | --- |
| `zero_grad` reuses the zero per gradient instead of rebuilding it every step | 1.031 → 0.037 ms |
| CUDA fused SGD (one kernel over a batch of parameters, also used when `momentum == 0`) | decode b1s1 training step 8.49 → 6.51 ms (torch 6.61) |
| automatic flush gains a size criterion `auto_flush_bytes` (default 8 MB) | host-bound 1.09–1.16x; Resnet50 training pipeline unchanged (146.1 vs 145.9 ms) |
| broadcast-shaped permute of rank ≥5 hands the whole loop nest to threads | qkv permute 165.9 → 62.3 us; b8s256 one-step device time 8.161 → 7.706 ms |
| remove the Python wrapper on `+ - * /` that exists only for complex scalars, moving the conversion into `py_converter.h` | decode one step saves 0.160 ms (5.9%) |
| small H2D blocks that need `ordered` become stream-order copies (the driver charges 2.3 ms for a ≤64 KB blocking copy, and torch pays it too) | 2344 us → 1.5 us to issue |
| `jt.graph_replay`: record an inference call once, then replay with different inputs | d512-L8 decode 0.87x, prefill 0.79x (vs torch); d1024 decode 1.10x |
| the layer_norm kernel reads a row into registers and reuses it | 44.1 → 27.4 us (torch 29.7) |
| move all CUDA work to `cudaStreamPerThread` (nvcc `--default-stream per-thread`, library handles bound to the stream) and add capture primitives | per-operator floor 7.89 → 4.25 us; replay is recorded into a device graph from the third run on |

The last item (`28e8e1a7`) is also the prerequisite for CUDA Graph: the legacy default stream cannot be captured. After inference replay was wired into the automatic path, tf-d1024-L4 decode inference 0.852 → 0.538 ms, tf-d512-L8 decode inference 0.575 ms. A full training step (forward+backward+update), after manual capture, is bit-identical to eager over 30 steps, 2.186 ms/step vs torch eager's 3.01–3.05 (1.39x), with host cost dropping from 3.5 ms to a single 2.24 us launch.

## Diagnostics measured and recorded

- **decode is slow in host graph construction.** Of 3.08 ms per step, Python graph construction is 2.382 ms (70%), launch 0.536 ms, and planning only 0.117 ms; one step makes 6426 Python/C calls to build about 100 operators. Removing the entire frontend saves only 38% of construction (251.5 → 156.3 us/block), so winning requires no longer rebuilding the graph every step — this is the basis for graph replay.
- **Per-operator floor.** Replaying a chain of 200 operators, CUDA costs 7.89 us per operator (CPU backend 1.05, bare `cudaLaunchKernel` 2.0–2.6), independent of graph size and operator kind; the bulk is the generated code `entry(this)` at 5.38 us. dlopen, legacy-stream implicit synchronization, queue saturation, scalar constants, and profiler hooks were all measured and are not it.
- **That 1.12x on b8s256 (15.06 vs 13.40 ms per step)** is half kernels (layernorm 1.49x, softmax 1.65x; gemm, transpose, elementwise level; the earlier "level per operator" conclusion was measured with a single wall clock and has been retracted), half launch gaps (9% vs torch's 4%).
- **The 0.50 ms of d1024 decode training**: the host cost of the training-mode layer_norm and softmax is 4.2–5.6x torch's (`jt.Function` mechanism accounts for about 61%), but even catching up would only reach 1.04–1.06 of torch; autodiff bookkeeping itself costs nothing. Adding a backward pass to cuBLASLt with `jt.Function` is a net loss and has been retracted.
- **Full-step training replay (without CUDA Graph) is worth only 1.08x**: `run_exec_plan` with a retained graph is 2.6x eager. A retained training graph rewrites its own leaves and is not idempotent — any unrelated weak sync that sweeps it silently takes one extra optimizer step.
- **That 2% in prefill training**: host 7.016 ms/step, device about 5.35 ms, pipelined measured 8.145 ms, with about 1.1 ms not overlapped; a full `auto_flush` scan and in-step synchronous copies were both ruled out.

## Two corrections to the gate criteria

- **"1615 failed" was manufactured by the harness.** When `tests/backends/parity/test_device_parity.py` and `tests/ops/test_ops.py` are written into the same pytest invocation, the process-global Torch-mode state left by the former turns the latter red en masse (parity fails in the first 99%, ops in the first 20%). Run one file per process. The "98 device-parity tests turned green" measured on this basis is void.
- **`test_ops.py` deterministically crashes at 73%**: `bilinear + align_corners=False` upsampling, forward already executed, the second `jt.grad(..., retain_graph=True)` segfaults, after which about 400 tests never ran. Logged and fixed: KI-EXEC-006. After working around it, `test_ops.py` drops from 286 to 56 failed (28 of which are the environment lacking cupy): the stale `np.atleast_1d` in the references (189), six reduce wrappers disagreeing on the `keepdim`/`keepdims` criterion (21), the harness only accepting namedtuple multi-output unpacking (20).

## Reproduction

```bash
# 一个 case 一个进程、两边交替；lab 脚本不进主仓库
bash $JITTOR_LAB_ROOT/metaop-perf/alt_pairs.sh
# 门禁：一个文件一个进程
bash $JITTOR_LAB_ROOT/metaop-perf/run_gate2.sh
# 仓库内的回放与捕获回归
python -m pytest -q tests/core/test_graph_replay.py tests/core/test_graph_capture.py
```

## Open items (as of this snapshot)

- The 1.1 ms of non-overlap in prefill training; the cause was not found.
- Training-step capture was not automated this round; it was later implemented by `StepCapture` (`2f794e28`, see the [real-model comparison report](2026-09-24-torch-compat-real-models.en.md)).
- Changing the training-mode layer_norm/softmax `jt.Function` to `cuda_grad_src` could save about 72–128 us/step; not done; the host cost of `no_grad` layer_norm is also 2.5x torch's.
- The remaining 28 in `test_ops.py`: `reinterpret_view_op.cc:58` byte count mismatch 16, `'float' object has no attribute 'astype'` 20 (test side), `cross_entropy_loss` not accepting `label_smoothing` 10, `norm_p2` default dimension 8, `median` reference 8, `rms_norm` second-order gradient 4, fused operator execution failure 2 (the original's category counts, whose total does not match 28, not re-checked).
- jittor training takes about 35 steps to reach steady state; torch is nearly stable immediately; cause not investigated.
