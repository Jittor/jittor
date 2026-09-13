# MiniMax-H3 under the Torch compatibility layer

- Status: Partial — pipeline runs end to end; the released 33B checkpoint is not yet
  usable at a practical speed on CUDA
- Date: 2026-09-12
- Baseline commit: `89622bb3` (dirty tree; see "Changes" below)
- Owner: Jittor compatibility maintainers
- Review when: the blocked items in "Open items" land, or the JIT op compiler's
  per-shape cost changes

## Question

Can MiniMax-H3 (33B omni video+audio diffusion transformer, Qwen3-VL-32B
conditioner, video and audio VAEs) run through `jittor.compat.torch`, with the
native fused FlashAttention path, and produce a video?

The upstream integration is a diffusers `ModularPipeline`
(`MiniMaxH3Blocks`), so this is a downstream-library adaptation and follows
[`agent/skills/downstream-library-adaptation`](../../agent/skills/downstream-library-adaptation/SKILL.md):
the model is consumed unmodified, and every breakpoint is routed to jittor core,
`jittor.compat.torch`, or the shim.

## Environment

- Jittor from this checkout, installed into a dedicated venv
  (`python3.12`, `pip install --no-deps --no-build-isolation .` plus `./compat`,
  then `jittor-torch-shim`).
- `transformers==5.5.3` (the adapter's supported version), `diffusers` at
  `0.41.0.dev0` (main, `a71e62e`; the MiniMax-H3 classes are not in a release).
- CUDA 12.9, sm_90; upstream `flash-attention` `v2.7.4.post1` with its CUTLASS
  submodule, built through `JITTOR_FLASH_ATTN_JITTOR_SRC` with
  `JITTOR_FLASH_ATTN_HEAD_DIMS=128 JITTOR_FLASH_ATTN_DTYPES=bf16`.
- Checkpoint: `MiniMaxAI/MiniMax-H3`, `FL2VA` partition, diffusers layout
  (transformer 62 GB, text encoder 63 GB, VAE 9.8 GB, audio VAE 0.6 GB).
- Oracle: real PyTorch 2.9.1 on the same host, same diffusers checkout.

Run state lives under `$JITTOR_LAB_ROOT/_state/h3/`; the harness
(`infer_h3.py`, `probe_audio_vae.py`, `env-jittor.sh`) is in
`$JITTOR_LAB_ROOT/minimax-h3/` and is deliberately outside this repository.

## Results

### Numerical parity on a reduced checkpoint (L2, CPU)

`hf-internal-testing/tiny-minimax-h3-modular-pipe`, `t2va`, 124 frames, 2 steps,
with the video and audio noise injected so the two runs draw identical latents:

| output | max abs diff | mean abs diff |
| --- | --- | --- |
| video frames (uint8, 0-255) | 1 | 0.046 |
| soundtrack (float) | 0.43 | 0.14 |

The video matches to rounding. The soundtrack does not correlate
(`corr ≈ 0.05`, RMS ratio 0.53), so the audio VAE path still differs
structurally; this was not isolated further because the same reduced checkpoint
reproduces both runs bit-for-bit only after the noise is injected, which is the
part that matters for the video path.

### End-to-end on CUDA

The same reduced checkpoint runs `t2va` on CUDA through the shim and writes a
124-frame mp4 with a soundtrack (64×64 and 512×288 canvases), with the native
fused FlashAttention kernel serving `scaled_dot_product_attention`.

### The released checkpoint

Loading works: 135 GB of components load into host memory (≈300 s) and the
components manager moves each one onto the accelerator on demand. The
transformer then runs, but its first forward spends its time **compiling JIT
kernels**, not computing: ≈730 distinct kernels were cached in the first hour
and ≈30 more appear per minute, one at a time. A single denoise step did not
complete in 75 minutes, so no video was produced from the released checkpoint in
this session.

Two environment facts make that worse and are worth fixing before the next
attempt:

- the shim pins `JT_USE_PARALLEL_OP_COMPILER=0`, and the value survives being
  exported from the caller's environment;
- `auto_cpu_offload` plus the shim's accelerator-default tensors mean a
  component is loaded into host memory and then moved, so the load path and the
  execution path disagree about the default device.

## Changes

Fixed while bringing the pipeline up. Each is a real defect, not a workaround;
the ones that changed shared behaviour are listed with what would catch a
regression.

### jittor core

| Change | Why |
| --- | --- |
| `ops/numerical.py`: new `sinc`, published as `jt.sinc`/`Var.sinc` | jittor had no normalized sinc; `torch.special.sinc` already expected `jt.sinc`, and the H3 audio VAE calls it |
| `src/type/fp16_compute.h`: default constructors for `float16`/`bfloat16` | the code generator declares `decltype(acc) stack[N]` for blocked reductions; without a default constructor every fp16/bf16 reduction that pass applies to fails to compile |
| `backends/cuda/kernels/cub/cub_cumsum_op.cc`: block size as a code-generation key | the JIT source transform flattens a C++ template parameter into a single `#define`, so `BlockScanKernel<512>` and `<1024>` collapsed into one and 8-byte types got the 1024-thread block, whose launch is rejected (92 registers × 1024 > 64k). float64/int64 `cumsum` failed with `cudaErrorLaunchOutOfResources` |

### `jittor.compat.torch`

| Change | Why |
| --- | --- |
| `installers/numerical/signal.py`: `kaiser_window` | the H3 audio VAE builds its resampling filter with it |
| `native_api.py`: `sinc` in the native API list | `torch.sinc` was unreachable even after core gained it |
| `installers/tensor/method_api.py`: `contiguous` materializes strided storage | it returned `self` unconditionally, so a broadcast/expanded Var kept storage smaller than its shape and the following `reshape` failed with "call contiguous() first" while the call meant to fix the layout did nothing |
| `installers/tensor/shape_api.py`: `reshape` falls back to a copy | torch's `reshape` is "a view when possible, otherwise a copy"; the native path only implements the view half |
| `frontend.py`: `Tensor(...)` accepts `dtype`/`device`/`requires_grad`/`pin_memory` | `accelerate` builds tensors as `param_cls(value, requires_grad=...)`; rejecting every keyword made device moves fail |

### shim

| Change | Why |
| --- | --- |
| `resources/torch_dist_info/top_level.txt` + `deploy.py` | without it `importlib.metadata.packages_distributions()` maps `torch` to `jittor-torch` only, Transformers reads the torch version as 1.3.x and disables its model code |
| `backends/flash_attention/__init__.py`: head dim 160 | upstream flash-attention ships 160-wide kernels and `flash_api.cpp` references the 160 split-kv dispatch; omitting it left an undefined symbol and the whole native backend failed to import |
| `cpp_extension/src/jtorch_aten.cu`: accept Var subclasses | `torch.Tensor` under the shim *is* a Var subclass, and the exact-type test rejected every tensor built through the torch API with "q must be a Jittor Var" |

## Open items

1. **JIT compile cost for a large model.** One forward of the 33B transformer
   needs thousands of distinct kernels. `jt.flags.use_parallel_op_compiler`
   arrives as `0` -- exporting `use_parallel_op_compiler` or
   `JT_USE_PARALLEL_OP_COMPILER` from the caller's environment does not change
   it -- and with it off the kernels are built one at a time, measured at
   15-30 per minute. The flag *is* settable at runtime, and the run harness sets
   it to 16 before the pipeline call; the framework-side question is why the
   environment spelling is ignored, and whether the compatibility layer should
   raise it by default for models this large.
2. **Autocast asks for mixed-dtype convolutions.** H3's video decode runs under
   `torch.autocast(float16)`; the shim's amp register changes the *result* dtype
   only, so a float32 convolution is asked for a float16 output, which cuDNN
   cannot serve and which trips `cudnn_conv3d`'s `best_algo_idx != -1`
   invariant. The run harness works around it by loading the video VAE in
   float16; the framework-side fix is to cast the operands the way torch's
   autocast does.
3. **Soundtrack parity.** The video path matches the oracle to rounding on the
   reduced checkpoint; the audio VAE path does not correlate yet.
4. **Attention dtype gate.** A float32 attention needs
   `JITTOR_FLASH_ATTN_CAST_FLOAT32=bf16` to reach the flash kernel; without it
   the shim silently materializes a dense `[heads, seq, seq]` score matrix,
   which at H3's sequence length is 80 GB.

## Reproducing

```bash
export JITTOR_LAB_ROOT=/root/jittor-lab
source $JITTOR_LAB_ROOT/minimax-h3/env-jittor.sh
export use_cuda=1
export JITTOR_FLASH_ATTN_JITTOR_SRC=/root/jittor-lab/flash-attention
export JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1
export JITTOR_FLASH_ATTN_HEAD_DIMS=128 JITTOR_FLASH_ATTN_DTYPES=bf16
export JITTOR_FLASH_ATTN_CAST_FLOAT32=bf16
$VENV/bin/python $JITTOR_LAB_ROOT/minimax-h3/infer_h3.py \
  --model /root/jittor-lab/_state/h3/models/tiny-h3 \
  --outdir /root/jittor-lab/_state/h3/runs/tiny-cuda --tag tiny-cuda \
  --height 64 --width 64 --num-frames 124 --steps 2 \
  --device cuda --vae-dtype float16
```

The oracle run is the same command under `env-oracle.sh` with
`PYTHONPATH=$JITTOR_LAB_ROOT/diffusers-main/src`.

Raw logs, the checkpoint, the FlashAttention build and the JIT caches are
unversioned and live under `$JITTOR_LAB_ROOT/_state/h3/`.
