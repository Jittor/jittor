# CUDA float32 attention row term and module zero_grad: two training-path fixes

- Status: verified on a real CUDA device, including before/after numbers. The
  equivalent Ascend change is recorded separately.
- Date: 2026-10-29
- Baseline commit: `008cae211`; the two changes in this report are `b0dcb366d`
  (attention row term) and `d2bd0de56` (batched `zero_grad` sync).
- Verification scope: RTX 4090 24 GB, CUDA 12.2.140 with the `jittor[cuda12]`
  component wheels (`pipcu122`), against PyTorch 2.11.0+cu128; same card, same
  Transformers site and same TF32 policy on both sides.
- Maintainer: Jittor core maintainers
- Review conditions: the row term in
  `backends/cuda/kernels/nn/fused_attention_f32_cuda.py`, `zero_grad` in
  `compat/torch/installers/nn/module_methods.py`, or the float32 SDPA dispatch
  changing; a new card or PyTorch version

## Symptom

In the CUDA large-model cases of `compat/tests/torch/test_ecosystem_speed.py`,
BERT-large (12 layers, hidden 768, 8x256) took `torch 27.4 ms / jittor 179.2 ms`
(6.55x) for one forward+backward step, while convnet/llama sat near 1.0x.
Phase timing put it in the backward:

| Phase | Jittor | PyTorch |
| --- | ---: | ---: |
| forward build + execute | 10.7 + 5.3 ms | 12.3 ms |
| backward build + execute | 15.5 + **148.1 ms** | 25.2 ms |
| `zero_grad(set_to_none=False)` | **11.9 ms** | 1.5 ms |

A device-side profile (CUPTI through `JITTOR_CUPTI_LIB` pointing at the 12.8
`libcupti.so.12`) showed `mea::row_delta` at 11.3 ms per call, 12 calls: the
float32 memory-efficient attention backward walked the whole `lq x lk x d`
score matrix **once per layer** only to rebuild a quantity the forward had
already computed.

## Root cause

For every query row, `sum_k p_k * dP_k` is the same number as `dO . O`, where
`O` is a small `[b, h, lq, d]` tensor. Recomputing it matters only when a row's
mass sits almost entirely on one key: there `O`'s float32 rounding swallows the
Q/K gradient, which is why `4c722fee0` introduced the exact path. For ordinary
distributions the reconstruction costs `2 * lq * lk * d` reads and writes per
layer -- gigabytes of traffic at 2048 tokens.

`Module.zero_grad(set_to_none=False)` had the same shape of problem: every
parameter's `zero_` first ran a full `sync_all` before writing, so 199
parameters meant 199 full syncs that drained the device between writes.

## Fix

- Row term: compute `dO . O` first; `_row_delta` reconstructs exactly only for
  rows with `1 / sum_k exp(score_k - max) > 0.5` (mass spread over fewer than
  two keys), and keeps the cheap value otherwise. The threshold is the module
  constant `_RECOMPUTE_INV_ABOVE`. Saturated and ordinary rows in one launch are
  decided row by row.
- `zero_grad`: collect every `.grad`, resolve readers with a single `sync_all`,
  then write the zero tensors in place. The single-tensor path still handles a
  deferred capture, an unbuilt tensor and a differentiable one.

## Results

`compat/tests/torch/test_ecosystem_speed.py -k EcosystemSpeedCUDA` (one card,
`JITTOR_ECOSYSTEM_LARGE=1`, ratio = Jittor / PyTorch):

| Case | PyTorch | Jittor | Ratio |
| --- | ---: | ---: | ---: |
| large_convnet | 17.3 ms | 17.7 ms | 1.02x |
| large_transformers_bert | 27.5 ms | 34.6 ms | **1.26x** (6.55x before) |
| large_transformers_gpt2 | 43.3 ms | 37.7 ms | 0.87x |
| large_transformers_llama | 39.3 ms | 38.8 ms | 0.99x |
| large_transformers_qwen3 | 48.1 ms | 38.5 ms | 0.80x |
| large_transformers_vit | 23.8 ms | 30.4 ms | **1.28x** (1.67x before) |
| large_diffusers_unet2d | 26.0 ms | 44.3 ms | **1.71x** |

Step by step (BERT-large, 4090): 186 ms -> 50 ms after the attention fix ->
41 ms after batching `zero_grad`; backward execution 148 ms -> 12 ms;
`zero_grad` 11.4 ms -> 3.4 ms. ViT's attention also takes the float32 fused
kernel, so both changes moved it from 1.67x to 1.28x. The diffusers case ran
against the `diffusers-installed-source/20261015` package site; its gap is not
one kernel -- the device side is 20.9 ms and close to PyTorch, while 63% of the
wall clock of its 1787 kernels goes to host graph construction and launches
(`cudnn_conv*` at 51/50/51 calls and GroupNorm's 90 `code` calls are ordinary).

## Alongside: a bf16 masked prefill takes the causal path

`qwen3_prefill` hands SDPA a `[1,1,2048,2048]` boolean causal mask built by
Transformers; with a mask the fused kernel can only treat it as a score-sized
bias, and the step measured 88.4 ms against PyTorch's 36.4 (2.43x).
`jittor.nn.functional.scaled_dot_product_attention` now compares the mask
element by element with `tril(ones)` and, when they are equal, calls with
`is_causal=True` -- the same statement -- which brings the case to **30.2 and
24.7 ms in two runs, 0.83x and 0.68x**. Masks that are not exactly causal
(padding, say) keep the old path, and a direct comparison of the two paths on
the same 2048-token inputs gives bit-identical outputs (max abs 0.0).

The same investigation found that a venv whose cuDNN and
`nvidia-cudnn-frontend` are from different majors (cuDNN 8.9 with frontend
1.30, for instance) answers "unsupported" for every shape, so attention
silently falls back to matmuls. `_library()` now self-checks one shape it must
support when it first loads and warns once when that fails.

## Alongside: the optimizer resolves readers once per parameter group

Every in-place optimizer store first runs a `sync_all` to resolve the
destination storage's readers; AdamW stores three times per parameter, so one
Qwen3 step ran several hundred full syncs. A parameter's readers in an update
are upstream of its new value, so the group now resolves them once and each
store only materializes its own source. `qwen3_train` on a 4090 went
212.2 -> **177.7 ms** (0.66x of PyTorch's 267.8); a collaborator's paired run
on a 910B3 measured 412 -> **268 ms** with steady memory 26.9 -> 28.9 GB. A
flat-buffer variant of the same path measured 264-270 ms for 6.1 GB more
memory, so it is not being merged.

## Reproduction

```bash
# Jittor side: an interpreter with the jittor[cuda12] component wheels;
# reference side: a real PyTorch interpreter
JITTOR_HOME=/path/to/jittor-home \
JITTOR_TORCH_SHIM=1 JITTOR_ECOSYSTEM_LARGE=1 \
REAL_TORCH_PYTHON=/path/to/torch/bin/python \
JITTOR_ECOSYSTEM_PACKAGE_SITE=/path/to/transformers-site \
python -m pytest -q compat/tests/torch/test_ecosystem_speed.py -s \
  -k 'EcosystemSpeedCUDA and large_transformers_bert'
```

The device profile needs CUPTI:
`JITTOR_CUPTI_LIB=<...>/nvidia/cuda_cupti/lib/libcupti.so.12`, then
`with jt.profile(device=True): ...`.

## Regression coverage

- `tests/backends/cuda/test_fused_attention_f32.py`: 25 cases, including a new
  one mixing saturated and ordinary rows in a single launch; the existing
  saturated-row and constant-value cases still pass.
- `compat/tests/torch/test_torch_compat_attention.py`: 48 passed / 38 skipped
  (the skips are flash-attn source not configured).
- `test_torch_module_method_owner_fidelity.py`, `test_optimizer_storage_updates.py`,
  `test_tensor_inplace_storage.py`, `test_torch_compat_optim.py`: 177 passed.
- `tests/structure`: 1571 passed / 4 skipped; the only failure is
  `test_asv_entrypoints` because that interpreter has no `asv`, unrelated here.
- Ecosystem parity (`test_ecosystem_parity.py -k EcosystemParityCUDA`):
  10 passed / 6 skipped / 1 failed. The failure is a T5 decoder MLP gradient at
  3.337e-02 against a 2e-2 band, identical at `008cae211`, so it is
  pre-existing and recorded as KI-COMPAT-021.

## Open

- The diffusers UNet's 1.71x is host graph construction, not a kernel; closing
  it means fewer operators per step or a cheaper host cost per operator.
- BERT/ViT's remaining 1.26-1.28x is the same host cost (about 5-7 us per
  operator, against PyTorch's 1-2 us).
