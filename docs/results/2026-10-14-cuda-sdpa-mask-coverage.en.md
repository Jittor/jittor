# CUDA SDPA fusion support matrix and mask training

- Status: the eager matrix and boundaries below are verified on real CUDA; fusion is not equivalent to a performance conclusion.
- Date: 2026-10-14.
- Synchronized baseline: `3d6aac9db75fb7018fa51ae71b9647bf130929dd`, including the earlier AMP fix and remote `7ed9b3b52b658d2a3c1ac390eba09087c5e80f95`; the fix is determined by the commit this report belongs to.
- Maintainer: CUDA attention and Torch compatibility layer maintainers.
- Platform: RTX 4090 / SM89, driver 580.178.04, CUDA compiler 12.2.140, GCC 11.5, Python 3.11; actual cuDNN 9.24.0, frontend 1.30.0.
- Review when: cuDNN/frontend versions, flash source configuration, attention dispatch, mask data domain, AMP or capture/replay rules change. Other GPUs, ROCm, ACL/NPU not verified.

## Eager support matrix

Each input group fixes batch=1, heads=2, with shapes as `(Lq,Lk,head_dim)`:

| Name | Shape |
| --- | --- |
| short_self | `(32,32,32)` |
| self | `(128,128,64)` |
| cross | `(33,47,64)` |
| wide_self | `(70,70,128)` |
| head160 | `(32,32,160)` |

Each group enumerates FP32/FP16/bf16 × no mask/bool padding/additive dense × causal on/off × inference/training, **180 execution cells** in total. bool padding is `[1,1,1,Lk]` with the last quarter of keys invisible; additive dense is a finite FP32 bias of `[Lq,Lk]`. Inputs are deterministic NumPy normal samples, Q/K/V and upstream gradient std 0.5, additive bias std 0.2.

The table below records the actual provider after synchronous output, executing backward and checking Q/K/V gradients. Returning `None` after selecting a function is not counted as fusion; composite implementations must also execute on CUDA.

| dtype / head_dim | mask | causal | inference | training |
| --- | --- | --- | --- | --- |
| FP32 / 32, 64, 128 | three kinds | two kinds | float32 tile fused | float32 tile fused |
| FP32 / 160 | three kinds | two kinds | composite | composite |
| FP16, bf16 / 32, 64, 128 | three kinds | two kinds | cuDNN fused | cuDNN fused |
| FP16, bf16 / 160 | three kinds | two kinds | cuDNN fused | composite |

Baseline is 48 cells float32 tile fused, 56 cells cuDNN, 76 cells composite; after the fix it is **48, 108, 24**, with **52 cells moving from composite to actual cuDNN fusion**. The remaining 24 cells come from the float32 tile head_dim≤128 limit and this platform's cuDNN backward rejecting head_dim=160. The latter's real probe returns `hidden_dim should be less than or equal to 128`, not a static registration inference.

This run did not configure the official FlashAttention source, and the native flash selector actually returned no result; this matrix does not claim to verify the flash extension's fusion capability. No new adapter or dependency was added to the repository.

## Changes and other execution scenarios

The cuDNN backward interface gains a constant bias consistent with forward; bias keeps FP32 to preserve the compute domain, avoiding finite values like `-1e9` becoming Inf when converted to FP16. Supports 1D and query-axis-broadcast padding masks. Mask preparation is pinned in the FP32 AMP domain; non-view mask updates invalidate the cache.

All `-inf` rows, and rows with no visible key after causal and mask intersect, are first given finite internal statistics to cuDNN, then the outer layer zeroes that row's output and upstream gradient; output and dQ are exactly zero, dK/dV match the independent closed-form reference. Additional regressions also verify constant masks for grouped heads.

| Scenario | Actual path and evidence |
| --- | --- |
| regular eager mask | the 180 cells above and their Q/K/V gradients; does not include capture state |
| full-row huge finite bias | explicit softmax composite. The new fusion candidate's dQ for a `-1e9` full row once showed about 31× peak error; on GPU, detection of FP32 row offset absorbing `log(Lk)` rejects and caches that result, and FP16/bf16 re-verification against real Torch recovers to about `5.6e-4` / `4.5e-3` |
| same row with part of the columns at `-1e9`, but still visible keys | padding mask continues cuDNN fusion; not rejected wholesale by the maximum absolute mask value |
| capture/replay tracing of FP16/bf16 cuDNN mask | keeps graph-only composite, avoiding host scalar read for mask-domain admission; four different inputs/different padding masks measured `captured=1,replayed=2`, output and all gradients pass against the closed-form reference |
| learned mask | cuDNN bridge does not return mask gradient, explicitly rejected, keeps the original path |

Capture scenarios and eager ratios were accepted separately; the 180-cell ratio does not represent the fusion rate in capture state. The replay evidence above corresponds to framework graph replay, not a claim of CUDA Graph recording. The existing cuDNN forward/backward path without mask keeps using the original selection logic.

## Independent reference and verification

Real PyTorch **2.11.0+cu128** executed MATH SDPA in a separate process on the same CUDA card; confirmed `torch._C` is a real binary extension with no shim installation marker. Torch MATH rejects passing both a mask and `is_causal=True`, so for these combinations the oracle explicitly merges top-left causal into the mask and calls `is_causal=False`; this is the mathematical reference for Jittor's current combination semantics, not a claim that the direct Torch API supports that parameter combination.

Jittor runs enable `forbid_backend_fallbacks()` and synchronously check the actual `location()==device` of the output and three gradients. Error is `max(abs(got-want))/max(abs(want))`:

| dtype | max output error | max error of the three gradients |
| --- | --- | --- |
| FP32 | `7.66e-7` | `7.80e-7` |
| FP16 | `7.73e-4` | `9.46e-4` |
| bf16 | `6.37e-3` | `7.64e-3` |

- CUDA cuDNN targeted: [test_cudnn_fused_attention.py](../../tests/backends/cuda/test_cudnn_fused_attention.py), **20 passed, 0 skipped, 13.83 seconds**; includes the new mask boundaries, AMP, changed-input replay and the existing long-sequence memory check.
- Native public math and flash decline: `test_sdpa_memory_efficient.py` and `test_sdpa_cuda_flash_kernel.py`, **26 passed, 0 skipped, 74.65 seconds**, CPU/CUDA.
- Torch public no-mask, causal, FP16 mask training, zero masked rows targeted: **4 passed, 0 skipped, 40.50 seconds**, CPU/CUDA. Also verified `autograd.grad`, four golden arrays, actual cuDNN provider and residency from the `import torch` CUDA FP16/bf16 bool-mask path, with runtime default `jt.flags.use_cuda=0`, device determined by actual operands.
- `bash tools/check_repo_layout.sh`, `git diff --check` and import layering passed, in-cycle modules stay 165.
- Full `tests/structure`: **1427 passed, 4 skipped, 3 failed, 234.42 seconds**. One is this baseline's stale `DropoutACL` gradient registration; the other two require wiring the new result report into the results toctree and generating `MANIFEST.in`. These shared files are maintained by the main task during integration. Two skips are resolution checks because the private venv lacks the core distribution; the other two are declared skips. Not used to claim the complete structure gate passed.
- Core gate run uniformly by the main task after integration; this task did not run full.

Reproduction commands use an independent environment, `JT_CUDA_KERNEL_MATH=strict`, single card `CUDA_VISIBLE_DEVICES=<gpu>`, isolated HOME/JITTOR_HOME/TMPDIR, first JIT serial:

```bash
JITTOR_TORCH_SHIM=0 JITTOR_TEST_DEVICES=cuda PYTHONPATH=python python -m pytest -q \
  tests/backends/cuda/test_cudnn_fused_attention.py
JITTOR_TORCH_SHIM=0 JITTOR_TEST_DEVICES=cpu,cuda PYTHONPATH=python python -m pytest -q \
  tests/nn/test_sdpa_memory_efficient.py tests/nn/test_sdpa_cuda_flash_kernel.py
```

Unversioned raw scripts, 90 Torch golden arrays, JSON and logs are in `$JITTOR_LAB_ROOT/_state/refactor-goal/sdpa/`. The matrix script runs in separate processes via the `torch` / `jittor` runtime arguments, with `--root` pointing to the golden directory and Jittor's `--output` to the result JSON. SHA-256:

| File | SHA-256 |
| --- | --- |
| matrix.py | `0557a06a8573f8da3edbe99b738a40db43bf5943c7ef23d91ae66730d311a4e7` |
| baseline.json | `4bcaee31e0b24178b7adc60e6b4f7bbf5214791a24eedd6c1b463ea0a43f538e` |
| final.json | `0e9ab7b41ae58f481f754b0842a49265da26ee75d68be59bb80799f71855bcdf` |

This is a support matrix over finite samples, dtypes and devices. padding mask is currently converted to an FP32 bias of `[bias_batch,bias_heads,Lq,Lk]`, not yet using a compressed sequence-length mask. This report does not do latency comparison or infer memory ratios, keeps the existing `APPROXIMATE` fidelity; nor does it claim the unmodified float32 extreme finite mask or known elementwise parity issues are resolved.
