# Using Jittor as PyTorch (`import jittor as torch`)

Jittor ships a torch-compatibility layer so PyTorch code — including
**transformers**, **PEFT**, **accelerate**, and **LlamaFactory** — runs
**unmodified** on Jittor, on both **NVIDIA GPUs (CUDA/NCCL)** and **Huawei Ascend
NPUs (ACL/HCCL)**. You only change the import:

```python
import jittor as torch          # or deploy the shim so `import torch` resolves to jittor
import torch.nn as nn
```

The torch-style API lives in `jittor/torch_compat.py` (installed onto the jittor
module at import); `jittor/torch_shim/` re-exports it as the `torch` package so
third-party libraries' internal `import torch` works too. Deploy that shim into the
active environment (so `import torch` / `torchvision` resolve to jittor) with:

```bash
python -m jittor.torch_shim.deploy           # install into the active env's site-packages
python -m jittor.torch_shim.deploy --check   # verify
```

## Quickstart

```python
import jittor as torch
import jittor as jt

lin = torch.nn.Linear(8, 8)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
x = torch.randn(16, 8)
for _ in range(10):
    opt.zero_grad()
    loss = (lin(x) ** 2).mean()
    loss.backward()        # routed through the active optimizer
    opt.step()
```

The device is auto-enabled: on import, jittor sets `flags.use_cuda=1` whenever an
accelerator is present (Ascend `has_acl` or NVIDIA `has_cuda`). **Sanity-check it's
on the device:** `npu-smi`/`nvidia-smi` should show GB-scale memory for the proc;
~100 MB means ops are running on CPU (~1000× slower).

## Numerical correctness

Validated against **real PyTorch** with identical weights/inputs
(`test/ascend_tools/xcheck/`):
- GPT-2 forward+backward matches torch to ~1e-7 (CUDA) / ~1e-5 (Ascend).
- Real Qwen3-0.6B (via transformers) produces **identical top-5 next-token
  predictions**; logits within ~1e-4.
- jittor-ACL and jittor-CUDA are bit-identical.

## Mixed precision (bf16 / fp16)

```python
scaler = torch.cuda.amp.GradScaler()                 # functional dynamic loss scaler
with torch.autocast("cuda", dtype=torch.bfloat16):   # context-manager AND decorator
    loss = model(x, labels=y).loss
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
```

- **bf16 is the preferred precision on Ascend** (cube-unit native: a 2048³ matmul
  is ~0.9 ms bf16 vs ~11 ms fp32). It matches the fp32 loss curve.
- fp16 works with the GradScaler (loss scaling + inf/nan skip).

## Multi-card DDP — no mpirun

A single torchrun-style launcher works on both backends (NCCL on NVIDIA, HCCL on
Ascend) via an env/file rendezvous — no MPI required:

```bash
# 2 GPUs / NPUs, data-parallel:
python -m jittor.distributed.launch -n 2 -- python train.py
```

In the training script use the normal data-parallel API (`jt.rank`,
`jt.world_size`, `var.mpi_all_reduce("mean")`, `module.mpi_param_broadcast`).
Verified: 2-card == 1-card gradient on both backends.

## Migrating torch checkpoints & safetensors

```python
sd = torch.load("model.pt")                   # reads real torch .pt (zip + storages)
from safetensors.torch import load_file       # pure-python, yields jittor Vars
weights = load_file("model.safetensors")
```
Both load real torch-saved files directly (all dtypes incl. bf16), no real torch
needed.

## Running transformers / LlamaFactory

Required environment:
- Python 3.10+ (transformers needs `types.UnionType`).
- `HF_DEACTIVATE_ASYNC_LOAD=1` — single-thread weight materialization (worker
  threads have no device context).
- `HF_HUB_OFFLINE=1`, `DISABLE_VERSION_CHECK=1` for offline runs.
- Install `trl` with `--no-deps` (else it pulls real torch and clobbers the shim).

Do **not** let pip install the real `torch` into the shim env — it overwrites
`torch/__init__.py`. Use a separate env for real torch (e.g. as a reference).

## Validated model coverage

Run through the `import torch` → jittor stack and checked **layer-by-layer against
real PyTorch 2.12** (same weights/inputs; forward `last_hidden_state` and per-
parameter `jt.grad` vs torch `.grad`, network-scaled). Forward **and backward**
match to ~1e-6 across ~30 `transformers` architectures and the CNN/diffusers stacks:

- **Decoder LLMs:** gpt2, llama, qwen2/qwen3, mistral, gemma/gemma2, phi/phi3, opt,
  bloom, gpt_neox, gptj, gpt_neo, stablelm, starcoder2, mpt, **falcon** (multi-query),
  **mixtral** (MoE).
  Plus (forward-clean, many also backward-checked): cohere, gemma3, granite, olmo/olmo2,
  persimmon, qwen2_moe, qwen3_moe, glm/glm4, gpt_bigcode, biogpt, ctrl, mpt, xglm,
  codegen, **longformer** (sliding-window), **roformer** (rotary), **phimoe**, **dbrx**,
  **nemotron** (MoE / modern-LLM variants).
- **Encoders:** bert, roberta, electra, distilbert, albert, **deberta/deberta-v2**,
  mpnet, xlm, flaubert, camembert, ernie, fnet, layoutlm, mobilebert, nystromformer,
  mra, yoso, **convbert** (span-conv), megatron-bert, rembert, luke, markuplm, **canine**.
- **Encoder-decoder:** t5, bart, mbart, pegasus, **pegasus_x** (block-local), m2m_100,
  marian, blenderbot/-small, mvp, plbart, umt5, nllb-moe, fsmt, led, big_bird.
- **Audio:** wav2vec2, **hubert**, **wavlm** (via `F.multi_head_attention_forward`),
  sew, unispeech/-sat, data2vec-audio.
- **Vision:** vit, deit, swin, convnext, **beit**, data2vec-vision, dpt, segformer,
  **levit** (hardswish) (plus `jittor.models` resnet/vgg/... and the native ViT).
- **diffusers generation:** `UNet2DModel` forward 1.1e-6 / backward 1.5e-6, a DDIM
  denoising loop 3e-5, `AutoencoderKL` encode+decode 1.4e-6 — i.e. jittor *generates*
  and the numbers match torch. Build via constructors / `from_config`; see limitations
  for loading *pretrained* checkpoints.

Real end-to-end training works: `transformers.Trainer` fine-tunes (loss decreases,
`grad_norm`/`clip_grad_norm_` apply), and CNNs train (every conv weight updates).
Real end-to-end **inference** works too: `model.generate()` runs **greedy** (KV-cached
decode is bit-identical to a from-scratch recompute — the cache is correct), **beam
search**, **sampling** (temperature/top-k/top-p) and **batched** generation.
A regression suite covers ~30 architectures (`jittor.test.test_torch_hf_models`,
incl. a `generate()` greedy/beam/sampling test) and the diffusers generation path
(`jittor.test.test_diffusers`).

`jittor.models` provides the CNN classics plus a modern **Vision Transformer**
(`vit_b_16`/`vit_b_32`/`vit_l_16`). LLMs/diffusion models come from
`transformers`/`diffusers` directly.

## Complex numbers & FFT

```python
c = torch.complex(re, im)                 # -> a ComplexNumber (real/imag pair)
torch.view_as_complex(x); torch.view_as_real(c)
torch.polar(abs, angle); torch.real(c); torch.imag(c); torch.conj(c)
torch.fft.fft(x); torch.fft.ifft(x); torch.fft.rfft(x); torch.fft.irfft(r, n=N)
torch.fft.fft2(x2); torch.fft.fftn(x, dim=(-2,-1))   # norm='backward'|'forward'|'ortho'
```
Complex is represented by `jittor.nn.ComplexNumber` (real/imag pair) with full
arithmetic; the `torch.fft.*` transforms are DFT-matrix based (so they run and
backprop on **both** Ascend and CUDA) and match `numpy.fft` to ~1e-4. A true native
complex `Var` dtype is still emulated, not a first-class dtype.

## Lightning-style training

```python
import jittor.lightning as pl          # or: import pytorch_lightning as pl (aliased)

class Lit(pl.LightningModule):
    def training_step(self, batch, idx): ...; return loss
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

pl.Trainer(max_epochs=5, gradient_clip_val=1.0,
           callbacks=[pl.ModelCheckpoint(monitor="val_loss"),
                      pl.EarlyStopping(monitor="val_loss", patience=3)]).fit(model, train_loader)
```
The core training/validation loop is implemented (epochs, gradient accumulation,
clipping, lr schedulers, `self.log`, `ModelCheckpoint`/`EarlyStopping` callbacks).
DDP strategies / precision plugins / the full logger ecosystem are not yet covered.

## Error reporting

Op failures now surface the real cause (op type, input shapes/dtypes, `[Reason]`)
instead of the old "Wrong inputs arguments / help(jt.sync)" noise. Unsupported
dtypes (e.g. float64 on Ascend) raise a clean Python error instead of aborting;
set `JT_SYNC=1` to pinpoint an async op failure.

## torch API coverage (verified vs real PyTorch 2.12)

Every entry below is checked **bit-for-bit (or to ~1e-6) against real PyTorch with
identical inputs/weights, on both CPU/CUDA**, and locked in regression suites
(`test/test_torch_compat.py` ~140 checks, `test/test_torch_linalg.py`,
`test/test_distributions.py`, `test/test_torch_hf_models.py`, `test/test_peft.py`,
`test/test_diffusers.py`).

- **Attention / transformers:** `F.scaled_dot_product_attention` (plain/causal/bool-mask/
  scale/GQA, fwd+bwd), `F.multi_head_attention_forward`, `nn.MultiheadAttention`,
  `nn.TransformerEncoderLayer`/`TransformerEncoder`/`TransformerDecoderLayer`/
  `TransformerDecoder`/`Transformer` (pre/post-norm, `generate_square_subsequent_mask`).
- **Recurrent:** `nn.LSTM`/`GRU`/`RNN` (+ Cells) — forward bit-matches torch (gate order
  i/f/g/o), `batch_first` output correctly `(batch,seq,hidden)`, bidirectional, `h_n`/`c_n`.
- **Norms / activations:** `F.rms_norm` (Llama/Qwen), `group_norm`/`batch_norm`/
  `instance_norm`/`layer_norm`, `silu`/`mish`/`hardswish`/`hardsigmoid`/`glu`/`elu`/
  `selu`/`celu`/`softplus`/`tanhshrink`/`softmin`/`threshold`.
- **Losses:** `cross_entropy` (+ `label_smoothing`/`weight`/`ignore_index`), `kl_div`
  (distillation, `batchmean`), `ctc_loss` (speech ASR, the CTC forward DP),
  `F.logsigmoid` (DPO/RLHF), `binary_cross_entropy`(_with_logits), `huber_loss`,
  `cosine_embedding_loss`, `margin_ranking_loss`, `triplet_margin_loss`,
  `gaussian_nll_loss`, `poisson_nll_loss`, mse/l1/smooth_l1, + `nn.*Loss` class forms.
- **`torch.distributions`:** `Categorical` (logits=softmax, differentiable — fixes a
  silent sigmoid bug that broke PPO), `Normal`, `Bernoulli`, `Exponential`, `Uniform`,
  `Geometric`, `Independent`, `OneHotCategorical`, `kl_divergence`, `Distribution` base.
- **`torch.linalg`:** `svd` (`full_matrices`, named `(U,S,Vh)`), `svdvals`, `inv`,
  `solve`, `cholesky`, `det`/`slogdet`, `eigh`/`eigvalsh`/`eigvals`, `qr`, `pinv`,
  `matrix_rank`, `multi_dot`, `lstsq`, `norm`/`matrix_norm` (CUDA svd/eigh need `cupy`).
- **`torch.func`:** `functional_call`, `grad`/`grad_and_value`, `vmap`, `jacrev`,
  `stack_module_state` (LoRA / meta-learning / ensembling).
- **Ops / methods:** `einsum`, `take_along_dim` (broadcasting), `roll` (incl. negative
  dims / flatten), `cumprod` (sign-aware), `index_fill_`, `index_put_` (dup-accumulate),
  `movedim`/`tensor_split`/`take`, `cdist`, `bucketize`, `searchsorted`, `pixel_shuffle`/
  `pixel_unshuffle`, `gumbel_softmax`, `interpolate`, `grid_sample`, `all`/`any`(`axis=`).
- **`nn.utils`:** `weight_norm`/`spectral_norm` (real reparametrizations), `clip_grad_*`,
  `rnn.pad_sequence`; `torch.optim.lr_scheduler` (LambdaLR/Linear/Cosine/Step/MultiStep/
  Exponential/Polynomial/Constant/Sequential/ReduceLROnPlateau).

## Status / limitations

Done + verified on both backends: forward/backward/training accuracy parity across
~75 transformers (decoder/encoder/enc-dec/audio/vision/MoE) + CNN + diffusers-generation
models, device dispatch, bf16/mixed precision, DDP-without-mpirun, gradient checkpointing,
checkpoint/safetensors migration, `model.save()`/`load()`, real `torch.cuda` memory
reporting, complex numbers + `torch.fft.*`, `F.multi_head_attention_forward`,
`torch.func` (functorch: `functional_call`/`grad`/`grad_and_value`/`vmap`/`jacrev`/
`stack_module_state` — used by LoRA / meta-learning / model ensembling, verified
bit-identical to real torch), `nn.utils.weight_norm`/`spectral_norm` (real
reparametrizations — `weight`→`weight_g`/`weight_v` recomputed before forward, and
`weight`/σ via power iteration; verified vs real torch + `np.linalg.svd`) and
`nn.utils.rnn.pad_sequence`, a torch-compatible `torch.optim.lr_scheduler`
(LambdaLR/LinearLR/CosineAnnealingLR/StepLR/MultiStepLR/ExponentialLR/PolynomialLR/
ConstantLR/SequentialLR/ReduceLROnPlateau — single source on BOTH the `import jittor as
torch` and deployed-shim paths; the HF `get_*_schedule_with_warmup` helpers wrap
LambdaLR and produce the exact torch curve), a Lightning-style training core, and clear
errors. The torch op surface is broad — verified
by an op-level differential battery (`op_parity.py`: ~84 ops vs real torch, plus a
backward battery) on **both** Ascend and CUDA.

**Loading *pretrained* checkpoints via `from_pretrained`** — including the accelerate
fast path (diffusers, and transformers with `low_cpu_mem_usage=True`, which is the
default) — now works and reloads **exact** weights. accelerate constructs the model
under `init_empty_weights()` + `no_init_weights()` then fills each parameter via
`set_module_tensor_to_device`; jittor doesn't use a `meta` device but two fixes make
this path correct: (1) `torch.nn.init` is guarded so `no_init_weights()` can't null
jittor's own construction init, and (2) `Module._parameters`/`_buffers` are
write-through so accelerate's `module._parameters[name] = value` assignment actually
persists (and keeps the param/buffer classification). Verified: a diffusers
`UNet2DModel` save→`from_pretrained`→forward roundtrip matches to 0.0 on both the
meta and plain paths (`test_diffusers.test_unet_from_pretrained_roundtrip`), and a
transformers `BertModel` roundtrip matches to 0.0.

A known limitation — **numpy 2.x**: the data-marshalling ABI fix landed (op
*values* are correct under numpy 2.x), but a separate flaky heap-corruption crash under
load remains (needs memory-debugging tooling to pin); **use numpy < 2** for now. Python
3.13 ships numpy 2.x, so the same guidance applies there.

In progress (deeper/core): a native complex `Var` dtype, PP/TP,
memory-manager tuning, cuDNN9, full pypi-based CUDA/cuDNN dependency resolution, the
remaining Lightning surface (DDP/precision/loggers), the numpy-2.x stability fix, and
triton/tilelang custom-op support.
