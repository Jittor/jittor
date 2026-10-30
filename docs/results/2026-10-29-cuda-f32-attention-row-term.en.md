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
kernel, so both changes moved it from 1.67x to 1.28x. Those two rows predate
the optimizer work; the optimizer and `order_after_readers` fixes at the end
of this page take `vit_b16_train` to 0.73x and `qwen3_train` to 0.66x. The diffusers case ran
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
on a 910B3 measured 412 -> **265 ms** (0.74 of torch's 357) with steady memory
26.9 -> 28.92 GB. A flat-buffer variant of the same path measured 264-270 ms
for 6.1 GB more memory, so it is not being merged.

The same round replaced the linear membership scans in `order_after_readers`
(`src/core/op.cc`) with hash sets; the vectors keep their insertion order and
the control edges are unchanged. That one helps the per-parameter path too:
`vit_b16_train` on a 4090 went 266.8 -> **141.5 ms** with the two fixes above
(0.73x of PyTorch's 194.4, against 1.67x before); on a 910B3 this tree's step
is 174.2-174.7 ms.

## Real-model matrix (`benchmarks/torch_compat`, 14 workloads)

One RTX 4090, paired runs, warm caches and no other tenant, at `86fa5dc73`:

| Workload | PyTorch | Jittor | Ratio |
| --- | ---: | ---: | ---: |
| qwen3_decode_static | 2714.9 ms | 542.5 ms | **0.20x** |
| resnet50_infer_b1 | 3.2 ms | 0.6 ms | **0.18x** |
| bert_base_infer | 3.1 ms | 1.2 ms | 0.38x |
| sd15_unet_train | 314.6 ms | 197.5 ms | 0.63x |
| sd15_vae_decode | 33.2 ms | 21.1 ms | 0.64x |
| qwen3_decode | 2122.4 ms | 1389.6 ms | 0.65x |
| qwen3_prefill | 36.8 ms | 24.4 ms | 0.66x |
| qwen3_train | 267.8 ms | 177.7 ms | 0.66x |
| resnet50_infer | 14.1 ms | 9.4 ms | 0.67x |
| bert_base_train | 66.1 ms | 57.3 ms | 0.87x |
| vit_b16_train | 193.2 ms | 172.0 ms | 0.89x |
| sd15_sample | 441.4 ms | 400.5 ms | 0.91x |
| resnet50_train | 91.7 ms | 85.5 ms | 0.93x |
| ddpm_unet_train | 75.4 ms | 74.4 ms | 0.99x |

Geometric mean **0.59x** (a second full run of the same commit on the same card
read 0.62x; the Jittor side is reproducible across the two, while PyTorch
drifted between 159 and 193 ms on the closest cases, so a single row is judged
by a paired run). Only `ddpm_unet_train` read above parity in the second run,
at 1.03x; `vit_b16_train` / `bert_base_train` / `ddpm_unet_train` re-measured
paired on an idle card at 0.88x / 0.86x / 1.03x. The two decode rows carry
agree flags (9.5e-02 and 1.2e-01, the harness marks anything above 5e-2):
those rows compare generated token ids, where greedy decoding of random weights
flips occasional near-ties; both runtimes produce the full 136 tokens.

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
