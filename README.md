# jittor-trellis

Optional TRELLIS.2 runtime integration for Jittor. This distribution owns the
project-specific behavior that should not live in the Jittor package itself.
Importing `jittor_trellis` is side-effect free; Jittor loads the registrations
through standard entry points, or an application can call
`jittor_trellis.install()` explicitly.

## What it registers

The `jittor.module_patches` entry point installs only TRELLIS-related patches:

- TRELLIS.2 dense and sparse attention fast paths, sampler-scoped cross-KV
  caching, sparse topology reuse, flexible-grid finalization, fused normalization,
  and DINOv3 rotary embedding.
- FlexGEMM/Triton argument compatibility and the validated sparse-convolution
  algorithm selection.
- DINOv3 extractor layout compatibility and lazy BiRefNet loading.
- An opt-in pure-Jittor submanifold convolution backend.
- Explicit extension-boundary policies for FlexGEMM, o-voxel, CuMesh, and
  nvdiffrast. Read-only, scratch-borrow, argument-specific, and forced-copy cases
  are kept separate.

The `jittor.external_backends` entry point extends the generic `flash-attn`
resolver. It searches from `TRELLIS2_ROOT` and `TRELLIS_ROOT`, including common
top-level, `third_party`, and `extensions` source directory names. It does not
install a project-specific import finder.

## Configuration

`FLEX_GEMM_AUTOTUNE_CACHE_PATH` always wins when set. Otherwise the adapter uses
`TRELLIS2_ROOT/.cache/jittor_trellis/flex_gemm/autotune_cache.json` (falling back
to `TRELLIS_ROOT`, `JITTOR_TORCH_RUNTIME_ROOT`, then the user cache directory).
This keeps autotuning results scoped to the active TRELLIS checkout when its root
is known.

Set `JITTOR_TRELLIS_SPARSE_BACKEND=jittor` to select the pure-Jittor
submanifold-convolution backend. Without that opt-in, FlexGEMM remains selected;
`JITTOR_TRELLIS_FLEXGEMM_ALGORITHM` can override its default
`IMPLICIT_GEMM` algorithm.

Runtime optimizations can be disabled together with
`JITTOR_TRELLIS_RUNTIME_PATCHES=0`. Individual controls include
`JITTOR_TRELLIS_CROSS_KV_CACHE`, `JITTOR_TRELLIS_PROCESSED_KV_CACHE`,
`JITTOR_TRELLIS_C2S_TOPOLOGY_CACHE`, `JITTOR_TRELLIS_FUSED_MESH`,
`JITTOR_TRELLIS_FUSED_RMS_NORM`, `JITTOR_TRELLIS_FUSED_SPARSE_RMS_NORM`,
`JITTOR_TRELLIS_FP16_LAYERNORM`, `JITTOR_TRELLIS_BF16_LAYERNORM`, and
`JITTOR_DINOV3_FUSED_ROPE`.

`TRELLIS_DINOV3_PATH` redirects the TRELLIS extractor to a local DINOv3 model.
The adapter does not contain general Transformers, PEFT, TRL, ms-swift, or FSDP
compatibility code.
