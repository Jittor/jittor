# Torch Compatibility Layer on Real Models vs. PyTorch: Gap Table, Memory and Performance Fixes

- Status: third edition (after the memory, fused attention, fused AdamW and host-overhead fixes). The suite
  ([`benchmarks/torch_compat/`](https://github.com/Jittor/jittor/blob/master/benchmarks/torch_compat/README.md)) is reusable. The fixes are submitted together with this
  report; the "after" column in the table comes from `f1e3dfa9` before submission plus the workspace with
  these changes (result files marked dirty), consistent with the submitted content.
- Date: 2026-09-24
- Base commit: `f1e3dfa9`
- Verification scope: RTX 4090 24 GB (driver 580.178.04), AMD EPYC 9554
- Reference: PyTorch 2.11.0+cu128 (cuDNN 9.19), Transformers 4.56.2, Diffusers 0.35.1,
  NumPy 1.26.4 on both sides; Jittor side nvcc 12.2.140 + `jittor[cuda12]` components (including
  `nvidia-cudnn-frontend` 1.30), the torch API declared by the shim is 2.11.0
- Maintainer: Jittor core maintainers
- Review trigger: changes to executor release timing, backward graph construction, graph replay, SDPA
  dispatch, optimizer, placement/type promotion; changing the GPU or the reference versions

## What is measured

The same ordinary PyTorch code (HF Transformers / Diffusers / hand-written ResNet-50) is run once in each of two
interpreters: in one, `torch` is standalone binary PyTorch; in the other, `torch` is Jittor after installing
`jittor-torch`. **Ratio = Jittor time / PyTorch time**; less than 1 means Jittor is faster, consistent with the
[downstream library performance ledger](../performance/library-standing.en.md).

Weights are randomly initialized according to published configurations (parameter counts match the official
ones: Qwen3-0.6B 596M, SD1.5 UNet 860M, VAE 84M); generative tasks disable EOS and use a fixed length. fp32
tasks enable TF32 on both sides. Each (task, runtime) is a separate process; time is the median of timed steps;
memory is the **actual peak process usage** read from NVML (the same ruler on both sides, including the CUDA
context and allocator cache).

```bash
python benchmarks/torch_compat/run.py --torch-python $TORCH_REF/bin/python --gpu 1
python benchmarks/torch_compat/run.py --runtimes jittor --workloads qwen3_train,vit_b16_train
```

## Results

Default specification. PyTorch and "Jittor before" come from `full-2`, "Jittor after" comes from `round2`; there
was no other load on the machine during measurement.

| Task | Spec | PyTorch | Jittor before | Jittor after | Ratio | Memory PT / JT before → after |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3_prefill` | bf16, 1×2048 | 36.2 ms | 52.6 ms | 40.4 ms | 1.12x | 1.7 / 3.3 → 1.6 GB |
| `qwen3_decode` | bf16, prompt 128, generate 128 | 2.15 s | 5.88 s | 4.73 s | **2.20x** | 1.6 / 6.0 → 1.6 GB |
| `qwen3_train` | fp32, 4×512, AdamW | 267.6 ms | OOM | 237.2 ms | 0.89x | 20.0 / >22 → 17.4 GB |
| `sd15_sample` | fp16, 512², DDIM 20 steps, CFG | 438 ms | 1588 ms | 1211 ms | **2.76x** | 2.4 / 5.7 → 3.1 GB |
| `sd15_vae_decode` | fp16, 64²→512² | 34.0 ms | 36.0 ms | 37.5 ms | 1.10x | 1.7 / 7.0 → 2.4 GB |
| `sd15_unet_train` | fp32, batch 2, AdamW | 270.6 ms | OOM | 273.3 ms | 1.01x | 17.1 / >22 → 15.9 GB |
| `ddpm_unet_train` | fp32, 64², batch 16, AdamW | 66.6 ms | 199.9 ms | 104.0 ms | **1.56x** | 4.4 / 6.9 → 4.4 GB |
| `resnet50_infer` | fp16, batch 64 | 14.3 ms | 12.7 ms | 12.3 ms | 0.86x | 1.1 / 1.7 → 1.2 GB |
| `resnet50_train` | fp32, batch 64, SGD | 91.5 ms | 98.5 ms | 98.3 ms | 1.07x | 6.5 / 10.4 → 7.1 GB |
| `vit_b16_train` | fp32, batch 64, AdamW | 159.2 ms | OOM | 182.2 ms | 1.14x | 9.0 / >22 → 9.4 GB |
| `bert_base_train` | fp32, 32×128, AdamW | 54.8 ms | 84.8 ms | 62.9 ms | 1.15x | 4.5 / 10.0 → 5.7 GB |

All 11 tasks ran successfully, and the geometric mean of the ratios is **1.26x** (1.69x for the 8 tasks that ran
before the fixes). Jittor's first step (cold compile) takes between 30 seconds and 7 minutes; once the cache is
warm it is 1.1–11.5 s. It is not included in the ratio, but users do pay it.

## How to read

- **Memory**: before the fixes, three training tasks did not fit on the 24 GB card; after the fixes all fit,
  with per-process peaks at 0.87–1.45× that of PyTorch, and the three training tasks (`qwen3_train`,
  `sd15_unet_train`, `ddpm_unet_train`) no higher than PyTorch.
- **On par or faster**: `qwen3_train` 0.89x, `resnet50_infer` 0.86x, `sd15_unet_train` 1.01x.
- **The remaining speed gaps are all on the host side**: the device time of the 20 `sd15_sample` steps is only
  about 350 ms (less than PyTorch's entire 438 ms), while the wall clock is 1211 ms; `qwen3_decode` is the same.
  The gap comes from rebuilding the graph in Python every step (a few microseconds per operator through the torch
  frontend) and from the executor's per-operator planning and launch. See "Host overhead" below.

## Memory: where the gaps come from and how they were fixed

Each item first had its gap measured with a minimal reproduction; the fix comes with a regression test, and the
test was confirmed to fail before the fix and pass after.

| Symptom | Root cause | Fix location | Regression test |
| --- | --- | --- | --- |
| The backward peak is about the sum of all intermediates: one step of an 8-layer Qwen3 is 13.07 GB vs. PyTorch's 9.17 GB; the backward of a 16-layer stack holds 23.8 more activations | The `batch_hold` added in `ceae1910` to fix a loader race pins every var in the batch until the whole batch ends; one backward is one batch | `src/core/executor.cc` (`schedule_hold_release`), `exec_plan.h`, `exec_runner.cc`: release the hold at the segment position of each var's last use within the batch | `tests/backends/cuda/test_batch_releases_memory.py` (23.8 → 4.0 activations) |
| Under the default `auto_flush_ops`, the backward peak is 5.5× that of lazy execution (2534 vs. 460 MB) | When constructing the backward, ops built by Python `Function.grad` trigger an automatic flush, so the backward is split into 16 batches and the previous segment's gradients are forced to materialize before the next segment begins | `src/core/grad.cc`, `src/runtime/submission_pipeline.h`: no automatic flush during `GradConstructionScope` | `tests/backends/cuda/test_auto_flush_graph_split.py::TestNoFlushWhileBuildingTheBackward`, `compat/tests/torch/test_backward_peak_memory.py` |
| The cross-entropy peak is 4.25× the logits, especially large on Qwen3's 150k vocabulary | The composite implementation retains multiple `[N, C]` intermediates through one-hot and autodiff | `python/jittor/nn/functional/loss.py`: closed-form backward in `_CrossEntropyRows`, using reindex to gather the target column | `tests/nn/test_cross_entropy_memory.py` (≤ 2.5×, numerically matches NumPy) |
| SD1.5 UNet training OOM: one attention forward-to-backward at 64² resolution retains 522 MB, PyTorch 5 MB | The math-path SDPA retains the `[..., Lq, Lk]` softmax for the backward, and autodiff additionally retains the scores | `python/jittor/nn/functional/attention.py`: use `_MemoryEfficientAttention` when gradients are needed and there is no dropout, saving only q/k/v/output and recomputing the probabilities in the backward | `tests/nn/test_sdpa_memory_efficient.py` (closed-form numerics, inference path unchanged, retained bytes) |
| SD1.5 VAE decode process peak is 7.0 GB vs. PyTorch 1.7 GB; decode itself peaks at only 0.5 GB | Automatic graph replay (`auto_graph_replay`) uses `keep_graph` to keep the whole graph replayable while also making every intermediate's memory resident: from the second call on it uses 6.2 GB extra | `src/core/exec_runner.cc`: `keep_graph=2` keeps the nodes but returns intermediate memory at last use within the batch and reallocates it on the next run (storage views restore their alias to the parent buffer); capture and replay in `python/jittor/_runtime/graph_replay.py` use it. Only recording into a CUDA graph needs fixed addresses and retains all buffers; the automatic policy records only graphs whose total allocation is ≤ `auto_graph_replay_retain_bytes` (64 MiB); `src/mem/`: the pool gains a cumulative allocation counter | `tests/core/test_graph_replay_retention.py` (before the fix both retained and peak were 128 MiB; checked with changing inputs each time, including storage views) |
| SD1.5 sampling uses 0.9 GB more activations than PyTorch | Inference (no-gradient) attention constructs the entire `[..., Lq, Lk]` score matrix and softmax at once, 1 GiB per call at 64² resolution | `python/jittor/nn/functional/attention.py`: when the scores exceed 384 MiB, chunk by query (non-causal) and pass Kᵀ as a cuBLAS flag | `tests/nn/test_sdpa_memory_efficient.py::TestInferenceAttentionInBlocks` and others |
| `torch.cuda.max_memory_allocated` reports only 0.1–1.3 GB on a 22 GB training step | The peak is sampled only at Python calls, so peaks inside a batch are invisible | `src/mem/allocator/sfrl_allocator.*`, `src/mem/mem_info.*`: record the peak of each allocation per device; `compat/torch/installers/cuda/api.py` reads it | `compat/tests/torch/test_backward_peak_memory.py::TestMaxMemoryAllocatedSeesInsideABatch` |

Trade-offs of memory-efficient attention:

- **Cost of recomputation**: compared with the composite implementation that retains probabilities, it does one
  extra QKᵀ and one extra softmax.
  - A single 4096-token attention forward+backward is 16.1 vs. 12.2 ms.
  - On the full model `vit_b16_train` is about 10% slower (179.7 vs. 162.6 ms), in exchange for 1 GB less.
  - `sd15_unet_train` goes from not fitting to runnable.
- **Two implementation details matter noticeably**:
  - The transpose must be passed to the batched matmul as a cuBLAS flag, not by calling `.transpose()` on an
    intermediate. The latter copies an entire `Lq×Lk` matrix, 2.5 ms per call at 4096 tokens.
  - The softmax must be composed from max/exp/sum, not a fused softmax kernel. The latter is opaque to fusion,
    so both input and output materialize, raising the SD1.5 UNet peak from 20.2 GB to 21.1 GB and OOMing again.

Trade-offs on the inference side:

- **Graph replay**: VAE decode before the fix is 36.0 ms / 7.0 GB (replay, everything resident); turning replay
  off alone gives 45.5 ms; replaying while returning intermediates at last use gives 37.5 ms / 2.4 GB. The decode
  step of d512 L8 allocates only 0.3–3 MiB per call and is still recorded as a CUDA graph, with each step
  unchanged at 0.3–0.4 ms.
- **Chunk size** (the math path when there is no fused kernel): chunking on the device side is not slower, but it
  affects the pipelined full step. In a 20-step sampling loop, 2–3 chunks per call is on par with no chunking
  (1528–1534 vs. 1535 ms), while 4 and 8 chunks slow down to 1649–1685 ms. The reason was not isolated (the
  automatic-flush cut points changed), so the chunks are kept large: no split under 256 MiB, and 1 GiB of scores
  is split into three chunks.

## Fused attention

The math path writes the `[..., Lq, Lk]` score matrix and softmax to memory and reads them back; PyTorch uses
FlashAttention for fp16/bf16 and memory-efficient kernels for fp32, both computing in tiles on chip and never
materializing the scores. `nn.scaled_dot_product_attention` now tries `nn.fused_attention` first:

| kernel | Coverage | 4096 tokens, 8 heads (SD1.5 64²) | Regression test |
| --- | --- | --- | --- |
| cuDNN fused SDPA (`backends/cuda/kernels/nn/cudnn_attention_cuda.py` + `cudnn_sdpa.cc`) | fp16/bf16, forward and backward, causal; bool/floating masks inference only (fully masked rows give 0, consistent with the composite implementation) | Inference 2.92 → 0.53 ms, +401 → +20 MiB; one training step 15.3 → 2.2 ms, +2079 → +50 MiB | `tests/backends/cuda/test_cudnn_fused_attention.py` |
| fp32 memory-efficient (`fused_attention_f32_cuda.py`, a SIMT implementation of the FlashAttention-2 algorithm) | fp32, forward and backward, no mask or causal, head_dim ≤ 128 | Inference 6.0 → 2.3 ms, +802 → +40 MiB; one training step 25.4 → 10.0 ms, +4147 → +110 MiB | `tests/backends/cuda/test_fused_attention_f32.py` |

The cuDNN frontend is header-only C++17 while Jittor's JIT is C++14, so the graph-construction code is compiled
separately into a small library with a C interface, and the `jt.code` operator just calls it; it requires
`nvidia-cudnn-frontend` (already added to `jittor[cuda12]`), and when absent the kernel declines and falls back to
the original path. Full model: `sd15_unet_train` 426 → 377 ms, 20.2 → 17.0 GB (together with the fused AdamW below,
273 ms, 15.9 GB); `qwen3_prefill` (bool causal mask) 52.5 → 43.2 ms (with the host-overhead fixes, 40.4 ms).

## Host overhead

After the fixes the gap is mainly on the host side: `ddpm_unet_train` has about 40 ms of device time per step,
and `sd15_sample` has about 17 ms of device time per UNet step but about 60 ms of wall clock. This round fixed
three places:

| Symptom | Root cause | Fix | Regression test |
| --- | --- | --- | --- |
| Host time of one AdamW step exceeds the entire forward | The torch frontend builds about ten graph nodes per parameter, i.e. 4500 for the 450 parameter tensors of a diffusers UNet; PyTorch defaults to foreach multi-tensor kernels | The CUDA implementation in `src/ops/composite/fused_adamw_op.cc` (a parameter table of up to 36 tensors per launch) + `backends/cuda/kernels/optim/fused_adamw_cuda.py`; the torch frontend's AdamW uses it by default unless `fused`/`foreach` is explicitly disabled | `tests/backends/cuda/test_fused_adamw_cuda.py`, `compat/tests/torch/test_adamw_fused_default.py` (differs from the per-parameter path by < 1e-5) |
| Every operator construction walks the Python stack | Async error location records the Python line number of every op: it fetches `f_globals` by name per frame, decodes the module name, and computes the line number, taking 5–8% of the training step's host time | `src/bindings/pybind/py_var_tracer.cc`: cache "whether Jittor/torch internal" per code object, and cache the origin per (code, instruction offset) | `tests/backends/cuda/test_async_error_location.py::test_launch_history_survives_graph_release` |
| The torch frontend's `a * 2.0` is 18 µs (native 4 µs) | Every binary operator converts both sides' dtypes into torch dtype objects and back, and scalars also go through the full `result_type` | `compat/torch/installers/tensor/method_api.py`: read the native dtype name directly; a floating scalar with a floating tensor and an integer scalar with a non-bool tensor take the tensor's dtype as the result dtype | `compat/tests/torch/test_torch_compat_promotion.py` |

Effect: `ddpm_unet_train` 211 → 104 ms, `bert_base_train` 84 → 63 ms, `sd15_sample` 1438 → 1211 ms,
`qwen3_decode` 5451 → 4735 ms; the torch frontend's `a + b` 7.8 → 5.8 µs, `Linear` 35 → 28 µs. The remaining host
time is spread across the torch frontend's Python wrappers, the executor's planning and launch, and host-side
execution of cuDNN convolution plans, with no single hotspot.

## Bugs fixed to make them run

When the suite first ran on the Jittor side, only 1 of the 11 tasks ran. Each item below first had a minimal
reproduction, and the fix comes with a regression test:

| Symptom | Root cause | Fix location | Regression test |
| --- | --- | --- | --- |
| First host compile under conda's nvcc cannot find `crt/host_config.h` | Only `<cuda_home>/include` is searched; conda's headers are in `targets/<target>/include` | `python/jittor/build/compiler.py` | `tests/build/test_cuda_library_search_dirs.py` |
| No CUDA kernel can compile because of `npp.h` | An optional component, NPP, is included unconditionally for 8 integer constants | `src/type/cuda_limits.h` | (covered by any CUDA kernel compile) |
| fp16/bf16 `MaxPool2d` on CUDA makes nvcc report an ambiguous `min` | half's `jittor::min/max` shadows the global integer overloads | `src/type/fp16_compute.h` | `tests/type/test_half_precision_parity.py::TestHalfPooling` |
| All HF/diffusers models die on GPU forward at `setitem` with mixed devices | `7b0f188f` makes every module call enter an environment placement of "default device = CPU", so buffers that ops allocate for results in the forward also land on the CPU | `compat/torch/frontend.py`, `nn_frontend.py` | `compat/tests/torch/test_default_device_is_cpu.py::TestTheDefaultIsOnlyForConstructors` |
| Qwen3 `generate()` on CUDA gives `TypeError: enable_gqa`; `test_torch_compat_attention.py` has 9 failures on HEAD | The flash SDPA registered on CUDA by `31fc3b2b` also answers the Torch frontend's ACL shortcut hook: its signature lacks `enable_gqa`, and it bypasses the frontend's own flash gate (short training goes through the math path, backend capability check and cache, compact GQA heads) | `backends/cuda/kernels/nn/flash_attention_cuda.py`: accept `enable_gqa`, reject short training itself, and mark `torch_frontend_loads_directly`; `compat/torch/installers/nn/attention.py` accordingly skips the hook; the threshold is provided uniformly by `compat/shim/backends/flash_attention.training_min_scores()` | `tests/nn/test_sdpa_cuda_flash_kernel.py`, `compat/tests/torch/test_torch_compat_attention.py` (9 → 0 failures) |
| An fp16 SD UNet silently becomes fp32, after which SDPA reports inconsistent q/k/v dtypes | `half / 1.0` and `half * 0-dim fp32` promote as two tensors; in torch, a Python scalar and a 0-dim tensor participate in promotion only from the higher category | `compat/torch/installers/core.py`, `installers/tensor/method_api.py` | `compat/tests/torch/test_torch_compat_promotion.py` |

Corresponding test-side adjustments:
- **Test infrastructure:**
  - The CPU session of `tools/run_test_suite.py` clears only the old name `nvcc_path`, not the canonical name
    `JT_BUILD_NVCC_PATH`. So in a shell that exports the latter, the CPU gate builds CUDA and is then rejected by
    its own readiness probe (`tests/structure/test_gate_scope.py`).
  - The result report added in `f1e3dfa9` did not regenerate `MANIFEST.in`, so the structural test
    `test_manifest_covers_runtime_trees_without_cache_payloads` is already red on HEAD.
- **Tests changed for the new default-device semantics:**
  - Two CUDA assertions in `test_independent_frontend.py` left over from `7b0f188f`.
  - Two `MultiheadAttention` cases in `test_torch_compat_attention.py`. Their "cuda" run originally had the whole
    forward squeezed onto the CPU; now, like torch, they first `.to(dev)` the module and inputs and actually
    execute on CUDA.

Verification:
- CPU/CUDA `--tier core` all green (CUDA: native 128 passed, torch 215 passed).
- `tests/structure` + `compat/tests/structure` 1494 passed, 0 failed.
- On CUDA, `tests/backends/cuda`, `tests/core`, `tests/nn`, `tests/optim`: 1163 passed, 12 failed; the 12 also fail
  on a clean `f1e3dfa9` worktree (see the next section).
- `compat/tests/torch/test_torch_compat_attention.py` has 0 failures on CUDA (9 on HEAD).

## Incidental findings, not fixed

These also reproduce on a clean `f1e3dfa9` worktree and were not introduced this round:

- **In-process `JITTOR_TORCH_SHIM=1` does not publish `torch`'s distribution metadata**: Transformers/Diffusers
  therefore conclude that PyTorch is absent. So `test_torch_hf_models.py` and `test_diffusers.py` in the test
  suite all 10 get `ImportError` in an environment where these libraries are installed (when not installed they
  are skipped, so it was never exposed). The user path (direct `import torch`) does not have this problem, and
  this suite takes the user path.
- **`torch.save` crashes on a CUDA build with zero visible GPUs** (`backend_streams.cc: Invalid stream
  device 0`): reproduced under `CUDA_VISIBLE_DEVICES=""`.
- The smoke-tier torch session additionally has 5 failures already on HEAD on this machine: two `trapz` cases call
  `numpy.trapezoid`, which only exists in NumPy 2 (the dev group pins 1.26.4); autocast conv2d with bias returns
  float32; the Generator pickle identity assertion; and the C++ extension cannot determine the CUDA architecture
  in a session without a GPU.
- `Tensor.true_divide` is missing.
- The 12 already failing on HEAD on CUDA (rechecked item by item on a clean worktree this round):
  - Two in `test_async_error_location.py`: the injected illegal-address write does not report, or does not carry
    the launch history.
  - Three in `test_cublas_test_op.py`, two in `test_cudnn_op.py`, two in `test_cuda_op_capabilities.py`, one in
    `test_backend_teardown.py`.
  - `tests/optim/test_optim_core.py::...test_low_precision_parameter_and_state_dtypes_are_preserved`
    (the executor topological-order invariant `queue.size() == roots.size()` fails).
  - `tests/nn/test_conv_parity.py::...test_the_generic_paths_do_reproduce_bit_for_bit`.
- Two torch frontend optimizer tests already failing on HEAD: the Adan step mixing devices in
  `test_torch_compat_optim.py`, and `normal_` initialization leaving results off CUDA.
- Points where memory can still be reduced further:
  - Training attention not covered by a fused kernel (masked, head_dim > 128, CPU) still uses the
    probability-recomputing Function, whose backward holds about 4 `Lq×Lk` fp32 temporaries at once.
  - The gradient of tied word embeddings retains 3 copies of the vocabulary size.
  - The allocator cache and context portion of the SD inference process peak (`sd15_sample` uses 3.5 GB of which
    only 2.0 GB is live).
  - Causal attention inference is not chunked (the mask is laid out over the whole query segment); long-sequence
    causal prefill still constructs the full score matrix.

## `torch.compile(mode="reduce-overhead")`: whole-step replay

Added 2026-09-25, based on the workspace of `cb93ba04`. What remains of the host overhead is that every step
rebuilds the graph in Python and then the executor plans and launches each operator; reducing it operator by
operator yields limited gains. PyTorch's answer to the same problem is to record once and replay many times, and
users express this with `torch.compile(..., mode="reduce-overhead")`; here it is mapped onto generic graph replay:

- Modules (inference): `OptimizedModule`, replaying
  [`GraphReplay`](https://github.com/Jittor/jittor/blob/master/python/jittor/_runtime/graph_replay.py) under `no_grad` (supports keyword arguments and
  tuple/dict/dataclass/`ModelOutput` structured outputs);
- Functions (the whole training step, forward + backward + optimizer):
  [`StepCapture`](https://github.com/Jittor/jittor/blob/master/python/jittor/_runtime/step_capture.py), where state updates come from runtime-recorded
  holder rebindings without needing to know the specific module and optimizer; the step and lr of CUDA fused
  AdamW are read from the device, and lr changes by the scheduler take effect as usual.
- Once stable it is recorded into a CUDA graph, one launch per step.

| Task | Jittor eager | `torch.compile` | PyTorch eager (table above) | Result |
| --- | --- | --- | --- | --- |
| `sd15_sample` (fp16, 20 steps, compile UNet only) | 1030 ms | 476 ms | 438 ms | The difference from eager is within eager's own run-to-run variation (max 0.20 / 0.25) |
| `ddpm_unet_train` (fp32, batch 16, AdamW) | 92 ms | 61 ms | 66.6 ms | The 8-step loss matches eager step by step to 1e-6, parameter difference 1.7e-4 (eager vs. eager 2e-4) |

The methodology differs from the table above: single process, `sync_all(True)` every step, taking the minimum
after stabilization, not through `benchmarks/torch_compat/run.py`. Memory is the allocator count
(`device_memory_peak`), not NVML: SD peak 1.89 GB (eager 1.81), DDPM peak 4.68 GB (eager 3.70). Recording holds
one step's working set; temporary workspaces are reused within the recording.

Executor bugs fixed while doing this (all of them make replay silently answer wrong; automatic replay is on by
default, so they also affect inference that is not compiled): `keep_graph=2` released a view's base while the
view (especially a reshape used as a result) still pointed at the old memory—CUDA inference at the end of
`jt.stack`, the hidden of `nn.RNN`, and the loss computed by the full-reduce fast path all stayed at their first
values; pass-through operators such as `tape` that launch no kernel and share storage with their input, once
released, got an unwritten new buffer—GroupNorm's backward therefore went wrong; during recording, an operator's
temporary workspace was freed and returned to the pool or the driver while the recorded kernel still used the old
address; reading back a parameter migrated it to the host while the recorded graph kept writing to the already
returned device block. Regression tests are in `tests/core/test_graph_replay.py`, `tests/core/test_step_capture.py`,
`compat/tests/torch/test_torch_compile_replay.py`.

Unsupported and explicitly declined, still executing normally (each call still executes exactly one step): steps
that draw random numbers, reading values back within a step, optimizers other than fused AdamW (SGD and others do
their bookkeeping in Python), and AdamW on CPU (the per-parameter path bakes the step count into the graph). See
also KI-COMPAT-006–008.

### Full-size suite (2026-09-26)

`benchmarks/torch_compat/run.py --compile none,reduce-overhead`, based on `75aa9977` plus the fixes in this
section (results marked dirty), consistent with the submitted content. Only compiled rows are listed; "speedup"
is the ratio of each side's eager to compiled.

| Task | PyTorch compiled | Jittor compiled | Jittor speedup | Jittor memory (eager) | Notes |
| --- | --- | --- | --- | --- | --- |
| `resnet50_infer_b1` | 0.6 ms | 0.7 ms | 9.40x | 0.6 GB (0.6) | host-overhead-dominated regime |
| `bert_base_infer` | 1.5 ms | 1.7 ms | 4.49x | 1.1 GB (1.0) | |
| `qwen3_decode_static` | 687.6 ms | 835.6 ms | 5.62x | 2.2 GB (2.2) | 1268 replays from a single capture |
| `sd15_sample` | 374.6 ms | 473.8 ms | 2.10x | 3.4 GB (3.3) | device-bound; the remaining gap is cuDNN layout conversion |
| `qwen3_train` | 187.0 ms | 246.0 ms | 0.99x | 17.4 GB (17.4) | OOM before the fix (21 GB) |
| `sd15_unet_train` | 208.7 ms | 253.4 ms | 1.02x | 16.0 GB (15.9) | 19.7 GB before the fix |
| `resnet50_train` | 75.3 ms | 99.0 ms | 1.00x | 7.3 GB (7.2) | 10.7 GB before the fix |

The numerical difference between compiled and eager is within 1e-5 everywhere (except bf16 decoding: greedy
decoding diverges under bf16, and PyTorch itself differs by 1.4e-1 before and after compilation). On device-bound
training tasks Jittor replay no longer has host overhead to save; the gap to PyTorch compiled comes from
inductor's kernel fusion and is outside this section.

The problems fixed this round are all general mechanisms rather than per-model patches:

- **Static KV cache decoding re-captures on every `generate()`**: `StaticCache.reset()` rebinds the cache tensors
  outside the step. An external rebind of the same shape and dtype is now copied into the original leaf
  (`_adopt`) instead of invalidating the capture.
- **Memory of a captured training step**, from three compounding places:
  - The release rule of `keep_graph=2` required everything in an alias group except the base to be a view, so
    `jt.Function`'s input and tape, and in-place `setitem` with its buffer (the embedding layer's scatter
    gradient), were never released. Now the whole group is released after the last member is used, and the alias
    reattaches to the base.
  - During capture, CUDA's `auto_flush_ops` and the `self.sync()` inside the compatibility layer's `backward()`
    executed the forward early; `keep_graph=2` released as soon as it was used, and it was computed again when
    the whole step synchronized—a step ran about two and a half times. Both are turned off during capture.
  - Blocks whose release was deferred during recording were neither split nor merged and could only be reused at
    approximately the same size. Now they are returned to the pool as usual during recording, and at the end the
    ranges the recording touched that are free at that moment are taken from the pool and held by the graph
    (equivalent to PyTorch's graph-private pool), so recording no longer costs extra memory.
- **A CPU scalar waits for the entire device**: the executor synchronized all devices on encountering any CPU
  operator after a GPU operator, and so did `numpy()`. diffusers' scheduler `prev_timestep >= 0` therefore waited
  for a whole UNet (22 ms) every step. Now it waits only when a CPU operator reads device or managed memory;
  reading back device memory already waits for its producing stream.

### Device-side kernels (2026-09-27)

After whole-step replay brought host overhead down, the gap in compiled rows is mainly device time. Using CUPTI
to compare kernel by kernel against PyTorch (`torch.profiler`, same workload, same compile mode), the following
items were fixed in order of gap size; all are general kernels, not specific to a model. Based on `97b9aab4` plus
the changes in this section (results marked dirty), consistent with the submission.

| Change | Reason (found by comparison) | Effect |
| --- | --- | --- |
| BatchNorm training/inference kernels changed to (channel × segment) reduction + whole-tensor elementwise, with per-element Welford statistics | One block per channel meant ResNet-50's first few layers had only 64 blocks; running stats were computed again with a separate generic reduction | BN-related time in one ResNet-50 training step 34 → 23 ms; variance precision is higher than the original two-pass method |
| fused AdamW: bias correction computed once per block, fp32 uses float4 | Every thread computed two `pow`s in double, and double runs at 1/64 rate on GeForce | Qwen3-0.6B step 46.3 → 21.6 ms (about 720 GB/s) |
| RMSNorm gamma gradient changed to 32-channel × 8-row tiling with segmented sums | One block per channel walking along rows left warp accesses uncoalesced | 62 µs per call → drops off the list |
| float32 fused attention supports bool / additive `attn_mask`, skips fully hidden blocks and does not read the mask for fully visible blocks; small grids use narrow query blocks; with no backward and a score matrix < 8 MiB it hands off to matmul | Transformers passes an explicit causal mask under `is_compiling()`; the original kernel gave up on seeing a mask, so compiled Qwen3 wrote out [L, L] scores per layer | Backward with an explicit causal mask 65 → 39 ms (`is_causal` gives 36 ms), and the score matrix is no longer written |
| GroupNorm rewritten the way BN is, and accepts float16 / bfloat16 | One block per (sample, group) meant SD at batch 2 had only 64 blocks; half precision did not take the fast path at all | GroupNorm over 20 SD1.5 sampling steps about 31 → 15 ms |
| Half-precision tensor division by a Python float changed to compute in float32 (PyTorch's opmath) | The float64 path set up for bit-exactness in float32 was also used for float16 | removes a 15 ms double kernel from SD1.5 sampling |

Full-size suite (Jittor side, ms; the PyTorch column is from the same round):

| Task | Compile | Before | After | PyTorch |
| --- | --- | --- | --- | --- |
| `qwen3_train` | reduce-overhead | 246.0 | 214.7 | 186.8 |
| `qwen3_train` | none | 242.7 | 210.8 | 271.2 |
| `sd15_sample` | reduce-overhead | 473.8 | 426.1 | 375.0 |
| `sd15_sample` | none | 993.9 | 830.9 | 448.3 |
| `sd15_unet_train` | reduce-overhead | 253.4 | 212.6 | 208.2 |
| `resnet50_train` | reduce-overhead | 99.0 | 88.8 | 75.4 |
| `ddpm_unet_train` | reduce-overhead | 64.7 | 58.0 | 52.2 |
| `sd15_vae_decode` | reduce-overhead | 36.7 | 28.8 | 24.6 |

The numerical difference of `resnet50_train` from PyTorch goes from 6e-4 to 3e-3: the first step (pure forward)
is closer (7.27325 vs. 7.27373; 7.27492 before the fix), and the difference appears in the second step; this
task's loss rises step by step, TF32 convolution itself carries error on the order of 1e-3, and the two
trajectories diverge at the second step. Against a float64 reference, the new BN's forward and backward errors are
both smaller than the old kernel's.

Follow-up (2026-09-27, `e8a4c49b`, `1957f9b6`):

- cuDNN attention reads and writes each of q/k/v in its own layout (BSHD or BHSD), passing
  `x.view(b, s, h, d).transpose(1, 2)` straight to cuDNN; `transpose` composes once when it meets a transposed
  view and returns the source tensor directly when the two are inverses, and view records are decided along the
  computation graph when the base holder is invalidated. SD1.5 sampling over 20 steps drops about 10 ms of
  transposes.
- float32 fused attention forward and backward changed to register tiling (transposed shared-memory storage +
  vectorized loads). One Qwen3 training step: forward 16.2 → 9.6 ms, backward 38.8 → 25.6 ms; PyTorch
  mem-efficient is 10.3 / 32.3 ms. `qwen3_train` compiled 214.7 → 194.0 ms (PyTorch 186.8), eager 210.3 → 187.9 ms;
  `vit_b16_train` compiled 176.0 → 161.6 ms.

Fusing BN / GroupNorm with the following elementwise operations was tried in three forms, none yielded a gain,
and none was adopted:

| Approach | Result |
| --- | --- |
| All BN with ordinary operators (reduction + elementwise, autodiff) | BN + residual add + ReLU forward plus backward is 20–30% slower |
| Welford kernel for the statistics, ordinary operators for normalization, with the statistics' gradient supplied by a small `Function` | Same as above, about 30% slower; the extra reduction and recomputation in the backward cancel the fusion |
| GroupNorm inference: the kernel produces per-(sample, channel) scale / shift, and normalization uses broadcast operators | It fuses with SiLU into one kernel but saves only 0.5 ms / 20 steps; half-precision elementwise is cheap anyway, and `transformer_2d`'s permute is a non-fused `transpose` operator that was not merged in |

The output of `jt.Function` passes through a tape operator (sharing storage with the input), where the fuser
breaks; a view reshape also truncates the fusion chain. Getting inductor-like gains requires the fuser to cross
these two boundaries, which is a change to the fuser itself.

Convolution filters (2026-09-27): for half precision cuDNN selects an NHWC kernel, and given an OIHW filter it
converts on every call; of the 54 ms of conversion in 20 SD1.5 sampling steps, 51 ms is the weights. When there is
no backward (neither input nor weights need gradients), it now supplies an OHWI filter instead: a copy is made
once per weight version (the Var held by the parameter), attached to the weight, and freed with the parameter; a
1×1 filter has the same bytes in both layouts and is not copied; training is unchanged. `sd15_sample` compiled
426.1 → 368.7 ms (PyTorch 375.0). It was then changed to keep no copy: the weight itself is moved into OHWI
storage and read out with OIHW strides (a `transpose_storage_view` view, with values and shape unchanged), and
cuDNN uses the same bytes as OHWI; SD one-step process memory 3622 → 2666 MB with no speed change. It can be
disabled with `jittor.nn.backends.cudnn.channels_last_filters = False`. ResNet-50 inference and VAE decode are
unaffected (the kernels they select do not convert weights in the first place).

Gaps that remain open, and why:

- The NCHW → NHWC conversion of convolution activations (after removing the weight conversion, SD sampling still
  has about 5.5 ms / 20 steps) and the memory of the filter copy: storing parameters in channels_last requires
  layout propagation.
- The NCHW → NHWC `permute` of `transformer_2d` before attention (SD sampling about 12 ms / 20 steps): PyTorch
  fuses it with GroupNorm's output.
- Fusion of BN / GroupNorm with adjacent elementwise operations: see the table above; it requires a fuser change.

## Boundaries

- Each configuration was measured for only one round; the after-fix numbers were measured on an idle machine. The
  same code's `sd15_sample` has shown about ±5% variation across processes, so the last digit of a ratio is
  unreliable. Host-overhead-sensitive tasks are very sensitive to co-located load: one round measured concurrently
  with unit tests put `ddpm_unet_train` at 426 ms, while a separate re-measurement gives 212 ms.
- Random weights: speed is independent of the weight values, but this report contains no numerical comparison;
  correctness is covered by the respective unit tests and the ecosystem comparison gates.
- The raw result JSON and logs are not versioned and are located at `$JITTOR_LAB_ROOT/_state/bench-torch-compat/`.
  `round2/` is present; `full-1/`, `full-2/` and the intermediate rounds were lost with the RAM disk when the
  machine restarted, so the PyTorch and "before" columns in the table are taken from the numbers recorded in the
  first edition of this report.
