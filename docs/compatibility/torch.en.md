# Writing Jittor with the PyTorch API

Jittor provides a Torch-compatible frontend so that code written for PyTorch—including **transformers**,
**PEFT**, **accelerate**, and **LlamaFactory**—runs on Jittor **without changing a single line**, while supporting
both **NVIDIA GPUs (CUDA/NCCL)** and **Huawei Ascend NPUs (ACL/HCCL)**.

A compatibility process must **activate explicitly**:

```python
from jittor.compat.shim import activate
activate()
import torch
import torch.nn as nn
```

Torch-style APIs belong to the `jittor.compat.torch` package. It is installed only when a Torch entry is selected:
an explicit `jittor.compat.shim.activate()` call, `JITTOR_TORCH_SHIM=1`, a deployed `import torch` package, or the
historical path `import jittor.torch_compat`.

- A plain `import jittor as jt` keeps native Jittor behavior and **does not occupy the `torch` namespace**.
- `import jittor as torch` is only a local Python alias and **does not activate the compatibility layer**.

`activate()` and the deployed package use **independent** Tensor, Parameter, Module, and NN types. The native
frontend and the Torch frontend share the Var/Op runtime, but **`torch is not jittor`**.

**The alias mode that installed the Torch API onto the native `jittor` module has been removed**, and it is no
longer an available form: `activate(independent_namespace=False)`, `JITTOR_TORCH_INDEPENDENT=0`, and direct calls
to `compat.torch.install(jittor)` all raise explicitly instead of falling back to the old behavior
(`require_independent_frontend()` in `compat/shim/preflight.py`, and the check at the top of the deployed
`torch/__init__.py`). The retained keyword exists only to produce a migration error. `JITTOR_TORCH_INDEPENDENT`,
once used to select the mode, is now **not needed**; setting it to a falsy value makes `import torch` fail
directly.

This separation is observable on APIs where the native and Torch contracts differ. A typical example: native
`Var.data` is still a shared NumPy view, while Torch mode returns a detached Tensor alias whose in-place writes
update the holder and **do not move CUDA/NPU data through the host**.

To deploy the shim into the current environment:

```bash
jittor-torch-shim            # 安装到当前环境
jittor-torch-shim --check    # 校验已部署的文件
```

## Quick start

```python
from jittor.compat.shim import activate
activate()
import torch

lin = torch.nn.Linear(8, 8)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
x = torch.randn(16, 8)
for _ in range(10):
    opt.zero_grad()
    loss = (lin(x) ** 2).mean()
    loss.backward()        # 由当前激活的优化器接管
    opt.step()
```

Devices are enabled automatically: at import time, if an accelerator exists (Ascend `has_acl` or NVIDIA
`has_cuda`), Jittor sets `flags.use_cuda` to 1.

> **Make sure it is really running on the device**: `npu-smi` / `nvidia-smi` should show this process occupying
> GB-scale device memory. Only about 100 MB means the operators are running on CPU (about 1000x slower).

## How devices are named

**Accelerators are always exposed under the device name `cuda`, including Ascend.** This is not a typo: on Ascend
builds `torch.cuda.is_available()` returns `True` (`is_available()` in `compat/torch/installers/cuda/api.py`
checks both `has_cuda` and `has_acl`), `torch.cuda.get_device_name()` returns `Ascend910B/NPU`, and `.cuda()` /
`.to("cuda")` place tensors on Ascend cards. This way upstream code that hard-codes `cuda` runs without changing
a line.

The `npu` spelling works too: `.npu()`, `.to("npu")`, and device strings such as `"npu:1"` all resolve to the ACL
backend (`compat/torch/types.py`, `compat/torch/frontend.py`,
`compat/torch/installers/nn/module_methods.py`).

**But `torch.npu` is a namespace deliberately kept inert: `torch.npu.is_available()` always returns `False`.**
`torch.npu`, `torch.xpu`, `torch.mps`, and `torch.mtia` are the same set of placeholder modules
(`compat/torch/installers/cuda/bindings.py`). This is intentional: many libraries use `torch.npu.is_available()`
as a switch to take `torch_npu`-specific paths—and `torch_npu` is a compiled PyTorch extension that is not
present here. Answering `False` keeps those branches closed and lets computation take the generic `cuda` path.
Likewise, the optional Transformers adapter makes `is_torch_npu_available()` return `False` too
(`adapters/jittor_adapters/transformers.py`). So: **to ask whether an accelerator exists, ask
`torch.cuda.is_available()`, not `torch.npu.is_available()`.**

`torch.__file__` points to the `torch/__init__.py` that was actually imported—the deployed entry file, or (when
not activated through a deployed entry) the shim source file it was copied from. The independent namespace has
no file of its own; `_adopt_entry_file()` in `compat/shim/runtime.py` attaches the entry's path to it, so common
code like `os.path.dirname(torch.__file__)` gets a real directory.

## Numerical correctness

Cross-checked against **real PyTorch** with identical weights and inputs (`tests/backends/acl/manual/xcheck/`,
run manually, requires a real Ascend device):

- GPT-2 forward + backward differs from torch by about `1e-7` (CUDA) / `1e-5` (Ascend);
- real Qwen3-0.6B (through transformers) produces **exactly the same top-5 next-token predictions**, with logits
  differing by about `1e-4`;
- jittor-ACL and jittor-CUDA are **bitwise identical**.

## Mixed precision (bf16 / fp16)

```python
scaler = torch.cuda.amp.GradScaler()                 # 函数式动态 loss scaler
with torch.autocast("cuda", dtype=torch.bfloat16):   # 既是上下文管理器也是装饰器
    loss = model(x, labels=y).loss
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
```

- **Prefer bf16 on Ascend** (natively supported by the cube unit: a 2048³ matmul takes about 0.9 ms, fp32 about
  11 ms), and the loss curve matches fp32;
- fp16 with GradScaler works (loss scaling + inf/nan step skipping).

The entire `torch.amp` surface is present: `autocast`, `GradScaler`, `custom_fwd`/`custom_bwd`,
`is_autocast_available`, and the two submodules `autocast_mode`/`grad_scaler`; `torch.cuda.amp` and
`torch.cpu.amp` are device-pinned subclasses (`issubclass` holds). The per-device state functions
(`is_autocast_enabled`/`set_autocast_enabled`/`get_autocast_dtype`/`set_autocast_dtype`/`clear_autocast_cache`/
`autocast_increment_nesting` and their older spellings) really read and write state; they are not empty
implementations.

Known differences (listed item by item in `fidelity_of("torch.autocast").detail`):

- autocast lands in jittor's global amp register rather than torch's per-operator allowlist. matmul / linear /
  convolution reduce precision and reductions stay float32 (consistent with torch), but fall-through classes
  that torch leaves alone (float32 `add`) also reduce precision here;
- one register serves all device types; CPU and CUDA regions are not independent of each other like torch's
  dispatch keys;
- `is_autocast_available` answers true only for backends this build can run (cpu/cuda, plus npu when ACL is
  installed), whereas torch also answers true for xpu/mps/xla/ipu/mtia—here that would amount to letting
  `torch.autocast("xla")` silently change the current backend's precision, so it is rejected outright.

## Multi-GPU DDP — no mpirun needed

A torchrun-style launcher works on both backends (NCCL on NVIDIA, HCCL on Ascend), using environment-variable or
file rendezvous and **without relying on MPI**:

```bash
# 2 张 GPU / NPU，数据并行：
python -m jittor.distributed.launch -n 2 -- python train.py
```

The training script uses the usual data-parallel APIs (`jt.rank`, `jt.world_size`, `var.mpi_all_reduce("mean")`,
`module.mpi_param_broadcast`). Verified: on both backends, **two-card gradients == single-card gradients**.

## Migrating torch checkpoints and safetensors

```python
sd = torch.load("model.pt")                   # 直接读真实 torch 的 .pt（zip + storages）
from safetensors.torch import load_file       # 纯 Python，产出 jittor Var
weights = load_file("model.safetensors")
```

Both can load files saved by real torch directly (all dtypes including bf16), **without installing real torch**.

The safety rules of `torch.load`:

- **`weights_only` defaults to `True`**, consistent with torch ≥ 2.6; format probing itself also goes through a
  restricted Unpickler, so a malicious ordinary `.pt` pickle is not executed during the probing stage.
- A standard torch zip checkpoint is read in restricted mode: reconstructed by storage offset, shape, and
  stride; negative strides and descriptors that run past the storage raise before construction, and unknown
  storage dtypes are rejected rather than treated as float32. Reading materializes values: shared-storage
  round-trip identity and non-contiguous physical layouts are not preserved.
- Legacy (non-zip) torch formats, native URL loading, and the native `.pkl` fallback have **no** interface where
  a restricted Unpickler can be injected, so `weights_only=False` must be passed explicitly, and only for
  trusted inputs.
- safetensors: NumPy requests still return NumPy arrays; torch requests preserve wide integers, BF16, and the
  requested device; float8 is explicitly unsupported; unsupported dtypes raise before the output file is
  opened.

## Running transformers / LlamaFactory

Environment requirements:

- Python 3.10+ (transformers needs `types.UnionType`);
- `HF_DEACTIVATE_ASYNC_LOAD=1`—single-threaded weight materialization (worker threads have no device context);
- for offline runs set `HF_HUB_OFFLINE=1` and `DISABLE_VERSION_CHECK=1`;
- installing `trl` requires `--no-deps`, otherwise it pulls in real torch and overwrites the shim.

> **Do not let pip install real `torch` into the shim environment**—it will overwrite `torch/__init__.py`. Keep
> real torch in a separate environment (for example, as a cross-check reference).

## Validated model coverage

The following models all ran through the `import torch` → jittor chain and were **layer-by-layer cross-checked
against an independently installed real PyTorch** (identical weights and inputs; forward compares
`last_hidden_state`, backward compares each parameter's `jt.grad` with torch's `.grad`, normalized by network
size). About 30 `transformers` architectures and CNN/diffusers stacks agree, for **both forward and backward**,
to about `1e-6`.

> **Which torch the cross-check uses.** The Torch API level declared by the compatibility layer is `2.11.0`
> (`torch.__torch_version__` / `torch.version.__version__`; `torch.__version__` reports Jittor's own version).
> The version of real PyTorch used as the reference **varies by report**: the nightly ecosystem gate uses the
> CPU build `2.7.1` (`.github/workflows/ecosystem.yml`), the
> [real-model gap table](../results/2026-09-24-torch-compat-real-models.en.md) uses `2.11.0+cu128`, and the more
> recent measurements in the [downstream library ledger](../performance/library-standing.en.md) use
> `2.13.0+cu129`. When citing a number, cite its reference version along with it.

- **Decoder LLMs**: gpt2, llama, qwen2/qwen3, mistral, gemma/gemma2, phi/phi3, opt, bloom, gpt_neox, gptj,
  gpt_neo, stablelm, starcoder2, mpt, **falcon** (multi-query), **mixtral** (MoE).
  Also (clean forward, most also backward-verified): cohere, gemma3, granite, olmo/olmo2, persimmon, qwen2_moe,
  qwen3_moe, glm/glm4, gpt_bigcode, biogpt, ctrl, xglm, codegen, **longformer** (sliding window), **roformer**
  (rotary position), **phimoe**, **dbrx**, **nemotron**.
- **Encoders**: bert, roberta, electra, distilbert, albert, **deberta/deberta-v2**, mpnet, xlm, flaubert,
  camembert, ernie, fnet, layoutlm, mobilebert, nystromformer, mra, yoso, **convbert** (span-conv), megatron-bert,
  rembert, luke, markuplm, **canine**.
- **Encoder-decoder**: t5, bart, mbart, pegasus, **pegasus_x** (block-local), m2m_100, marian, blenderbot/-small,
  mvp, plbart, umt5, nllb-moe, fsmt, led, big_bird.
- **Audio**: wav2vec2, **hubert**, **wavlm** (through `F.multi_head_attention_forward`), sew, unispeech/-sat,
  data2vec-audio.
- **Vision**: vit, deit, swin, convnext, **beit**, data2vec-vision, dpt, segformer, **levit** (hardswish), plus
  `jittor.models`' resnet/vgg/… and native ViT.
- **diffusers generation**: `UNet2DModel` forward `1.1e-6` / backward `1.5e-6`, DDIM denoising loop `3e-5`,
  `AutoencoderKL` encode+decode `1.4e-6`—that is, jittor is **really generating**, and the numbers agree with
  torch. Build models with the constructor / `from_config`; for loading **pretrained** checkpoints see below.

End-to-end training is available: `transformers.Trainer` can fine-tune (loss decreases, `grad_norm` /
`clip_grad_norm_` take effect), and CNNs can train (every convolution weight updates). End-to-end **inference** is
likewise available: `model.generate()` supports **greedy** (decoding with KV cache is bitwise identical to
recomputing from scratch, showing the cache is correct), **beam search**, **sampling** (temperature/top-k/top-p),
and **batched** generation.

The regression suite covers about 30 architectures (`compat/tests/torch/test_torch_hf_models.py`, including
greedy/beam/sampling tests for `generate()`) and the diffusers generation path
(`compat/tests/torch/test_diffusers.py`).

`jittor.models` provides classic CNNs plus modern **Vision Transformers** (`vit_b_16`/`vit_b_32`/`vit_l_16`).
LLMs and diffusion models come directly from `transformers` / `diffusers`.

## Downstream library status

`compat/tests/torch/test_ecosystem_parity.py` **runs each case twice**—once in an interpreter where `torch` is
real PyTorch, once in this interpreter—starting from identical weights and comparing forward outputs as well as
**every parameter gradient and input gradient**, on both CPU and CUDA. There is also a real-NPU scope class
covering cross-checks of MMCV ConvModule and MMEngine BaseModule against `torch_npu`, including fail-closed
CPU-fallback detection. Point `REAL_TORCH_PYTHON` at the real PyTorch interpreter; optionally use
`JITTOR_ECOSYSTEM_SPEED_RATIO` to constrain the wall-clock ratio.

| Library | Parity gate coverage | Notes |
| --- | --- | --- |
| `transformers` | gpt2, llama, bert, vit, t5, whisper | encoder, decoder, encoder-decoder, and audio paths |
| `diffusers` | `UNet2DModel`, `DiTTransformer2DModel` | convolutional and latent transformer backbones |
| `peft` | LoRA on llama | |
| `ms-swift` | its own LoRA tuner (llama) | requires `peft < 0.20`; ms-swift 4.5.2 with peft 0.19 raises the same `TypeError` under real PyTorch |
| `mmcv` / `mmengine` | `mmcv.cnn.ConvModule`, `mmengine.model.BaseModule` | CPU, CUDA, NPU; pure-Python layers only, see below |
| `torchmetrics` | classification and regression metrics (`compat/tests/torch/test_torchmetrics_compat.py`) | 1.7.4, through the `jittor_adapters.torchmetrics` adapter |

Two boundaries are **architectural**, not unfinished work:

- **Compiled PyTorch extensions.** `mmcv.ops` and any package shipping a `.so` linked against `torch` are
  compiled against PyTorch's C++ ABI. **A Python-level compatibility layer cannot load them**; those operators
  need Jittor's own implementations—`jittor.models` and the native operator surface exist for the cases that
  matter.
- **Runtimes that manage devices themselves.** vLLM embeds custom CUDA kernels and its own memory and scheduling
  layers rather than calling `torch.*`, so the shim alone cannot carry it. It runs through the optional adapter
  `jittor_adapters.vllm` (shipped with the `jittor-torch-adapters` distribution, entry point name
  `jittor_vllm`), which provides those kernels for Jittor: vLLM V1 can load Qwen3-0.6B, build a KV cache, and
  complete greedy decoding on real CUDA with tokens **exactly matching** real PyTorch/Transformers, and
  four-token warm-start generation takes `0.109s` versus `0.137s`. verl takes the same path—its import,
  protocol, FSDP2, and PPO gates all pass, including four-card FSDP2.

TRELLIS.2 4B once completed an aligned end-to-end pipeline with four real CUDA extensions on an external adapter
**not in this repository**, with a warm-state pipeline median of `7.515s` versus `6.878s` for real PyTorch—about
`1.09x`, **performance not yet accepted**; that measurement is a one-off conclusion, see the
[downstream library ledger](../performance/library-standing.en.md).

Project-specific glue code lives in optional adapters registered through entry points, **not in mainline
Jittor**; the rationale is in [repository layout](../development/repository-layout.en.md). The one maintained in
this repository is [`adapters/`](https://github.com/Jittor/jittor/blob/master/adapters/README.md) (distribution
`jittor-torch-adapters`, Python package `jittor_adapters`), which has three entry points: `jittor_transformers`
(Transformers 4.56.2 / 5.5.3: make its own `torch_npu` probe return false), `jittor_torchmetrics`
(TorchMetrics 1.7.4), and `jittor_vllm`. Versions are explicit: an unrecognized version raises
`UnsupportedAdapterVersion` rather than trying its luck. Glue code for TRELLIS and Gaussian Splatting is not in
this repository and is not shipped with this release line.

## Complex numbers and FFT

```python
c = torch.complex(re, im)                 # -> 原生 complex64 Var
torch.view_as_complex(x); torch.view_as_real(c)
torch.polar(abs, angle); torch.real(c); torch.imag(c); torch.conj(c)
torch.fft.fft(x); torch.fft.ifft(x); torch.fft.rfft(x); torch.fft.irfft(r, n=N)
torch.fft.fft2(x2); torch.fft.fftn(x, dim=(-2,-1))   # norm='backward'|'forward'|'ortho'
```

`complex64` is a first-class dtype of `jittor_core`, and the public APIs above accept and return native complex
`Var`s. Some linear-algebra and FFT algorithms still internally use the old `jittor.nn.ComplexNumber`
real/imaginary representation and then convert results back to native tensors. That bridge is an implementation
detail; the limitation is registered, see [complex dtype](../notes/complex-dtype.en.md).

## Error reporting

Operator failures give the real cause (operator type, input shapes/dtypes, `[Reason]`) instead of the old
"Wrong inputs arguments / help(jt.sync)" noise. Unsupported dtypes (for example float64 on Ascend) raise a clean
Python exception instead of aborting. When a lazily executed operator fails, the report gives the line that
created it by default; for the full creation stack set `JT_TRACE_PY_VAR=3`, and add `JT_SYNC_RUN=1` when the
error comes from an accelerator kernel. (`JT_SYNC` is a compile-time macro; changing it recompiles the core.)

## torch API coverage (validated against real PyTorch)

Each item below was cross-checked **on CPU and CUDA** against real PyTorch with identical inputs/weights
**bitwise (or to about 1e-6)** and locked into the regression suite (`test_torch_compat.py`,
`test_torch_compat_linalg.py`, `test_torch_compat_distributions.py`, `test_torch_hf_models.py`, `test_peft.py`,
`test_diffusers.py` under `compat/tests/torch/`). For reference versions see the note under "Validated model
coverage" above.

- **Attention / transformer**: `F.scaled_dot_product_attention` (normal/causal/bool mask/scale/GQA,
  forward+backward), `F.multi_head_attention_forward`, `nn.MultiheadAttention`,
  `nn.TransformerEncoderLayer`/`TransformerEncoder`/`TransformerDecoderLayer`/`TransformerDecoder`/`Transformer`
  (pre/post-norm, `generate_square_subsequent_mask`).
- **Recurrent networks**: `nn.LSTM`/`GRU`/`RNN` (including Cells)—forward is bitwise identical to torch (gate
  order i/f/g/o), `batch_first` output is correctly `(batch, seq, hidden)`, and bidirectionality and
  `h_n`/`c_n` are supported.
- **Normalization / activation**: `F.rms_norm` (Llama/Qwen), `group_norm`/`batch_norm`/`instance_norm`/
  `layer_norm`, `silu`/`mish`/`hardswish`/`hardsigmoid`/`glu`/`elu`/`selu`/`celu`/`softplus`/`tanhshrink`/
  `softmin`/`threshold`.
- **Losses**: `cross_entropy` (including `label_smoothing`/`weight`/`ignore_index`), `kl_div` (distillation,
  `batchmean`), `ctc_loss` (CTC forward DP for speech recognition), `F.logsigmoid` (DPO/RLHF),
  `binary_cross_entropy`(`_with_logits`), `huber_loss`, `cosine_embedding_loss`, `margin_ranking_loss`,
  `triplet_margin_loss`, `gaussian_nll_loss`, `poisson_nll_loss`, mse/l1/smooth_l1, and the corresponding
  `nn.*Loss` classes.
- **`torch.distributions`**: `Categorical` (logits=softmax, differentiable—fixing a silent sigmoid defect that
  would break PPO), `Normal`, `Bernoulli`, `Exponential`, `Uniform`, `Geometric`, `Independent`,
  `OneHotCategorical`, `kl_divergence`, and the `Distribution` base class.
- **`torch.linalg`**: `svd` (`full_matrices`, named `(U,S,Vh)`), `svdvals`, `inv`, `solve`, `cholesky`,
  `det`/`slogdet`, `eigh`/`eigvalsh`/`eigvals`, `qr`, `pinv`, `matrix_rank`, `multi_dot`, `lstsq`,
  `norm`/`matrix_norm` (CUDA svd/eigh require `cupy`).
- **`torch.func`**: `functional_call`, `grad`/`grad_and_value`, `vmap`, `jacrev`, `stack_module_state` (LoRA /
  meta-learning / ensembles).
- **Operators and methods**: `einsum`, `take_along_dim` (including broadcasting), `roll` (including negative
  dimensions and flattening), `cumprod` (sign-aware), `index_fill_`, `index_put_` (accumulating repeated
  indices), `movedim`/`tensor_split`/`take`, `cdist`, `bucketize`, `searchsorted`, `pixel_shuffle`/
  `pixel_unshuffle`, `gumbel_softmax`, `interpolate`, `grid_sample`, `all`/`any` (`axis=`).
- **`nn.utils`**: `weight_norm`/`spectral_norm` (real reparameterization), `clip_grad_*`, `rnn.pad_sequence`;
  `torch.optim.lr_scheduler` (LambdaLR/Linear/Cosine/Step/MultiStep/Exponential/Polynomial/Constant/Sequential/
  ReduceLROnPlateau).

## Status and limitations

**Done and validated on both backends**: precision consistency for forward/backward/training of the batch of
transformers listed above (decoder/encoder/encoder-decoder/audio/vision/MoE, of which 33 architectures are locked
into the regression suite in `compat/tests/torch/test_torch_hf_models.py`, with the rest from manual
cross-checks) plus CNNs plus diffusers generation models, device dispatch, bf16 and mixed precision, DDP without
mpirun, gradient checkpointing, checkpoint/safetensors migration, `model.save()`/`load()`, real `torch.cuda`
memory reporting, complex numbers and `torch.fft.*`, `F.multi_head_attention_forward`, `torch.func` (the
functorch family, bitwise identical to real torch), `nn.utils.weight_norm`/`spectral_norm` (real
reparameterization: `weight` → `weight_g`/`weight_v` recomputed before forward, σ by power iteration;
cross-checked against real torch and `np.linalg.svd`), `nn.utils.rnn.pad_sequence`, the torch-compatible
`torch.optim.lr_scheduler` (**a single implementation on both the `activate()` and deployed-shim paths**; HF's
`get_*_schedule_with_warmup` wraps LambdaLR and produces exactly the same curves as torch), and clear error
reporting. The operator surface is validated **on both Ascend and CUDA** through operator-level differential
cross-checks (`op_parity.py`: about 84 operators against real torch, plus backward cross-checks).

**Loading pretrained checkpoints with `from_pretrained`**—including the accelerate fast path (diffusers, and the
`low_cpu_mem_usage=True` that transformers enables by default)—is now available and restores weights **exactly**.
accelerate builds the model under `init_empty_weights()` + `no_init_weights()` and then fills parameters one by
one with `set_module_tensor_to_device`; jittor has no `meta` device, and two fixes make this path correct:
(1) `torch.nn.init` is protected so that `no_init_weights()` does not clear jittor's own construction
initialization; (2) `Module._parameters` / `_buffers` are made write-through so that accelerate's
`module._parameters[name] = value` really takes effect while preserving the parameter/buffer classification.
Verified: a diffusers `UNet2DModel` save → `from_pretrained` → forward round trip agrees to `0.0` on both the
meta and ordinary paths, and the transformers `BertModel` round trip also agrees to `0.0`.

**Known limitations of optimizers and schedulers.** Adam/AdamW call the shared native `adam_update`, and SGD,
RMSprop, and Adan delegate to their respective native steps; there is no second set of update equations.
`torch.optim.LBFGS` is not implemented (raises `NotImplementedError` at construction; fidelity is registered as
`unimplemented`); `SGD` does not yet accept `foreach=`/`fused=` (see known issue `KI-COMPAT-008`). Scheduler
formulas are kept as-is, but full resume semantics are **not promised**, and ignored optional arguments are not
filled in; `AveragedModel` still bases on the native Module and accepts `use_buffers` without averaging buffers.
These APIs are registered as `approximate` in the fidelity report.

**NumPy 2.x and Python 3.13** are compatibility paths under maintenance. Jittor selects versioned NumPy C-API
entry points for array copies, avoiding construction of legacy dtype descriptors and deriving transfer sizes
from Jittor dtypes. The Python 3.13 wheel gate runs under NumPy 2.x, covering the CPU self-check, non-C-contiguous
array transfer, and Python-variable tracing. The package accepts all NumPy versions below 3.

**Not provided**: `pytorch_lightning` / `lightning`. There used to be a homegrown Lightning-style training loop
(`jittor.lightning`), which was deleted in 2026-07 (`5d49afc3`); there is now no Lightning compatibility layer
and no `pytorch_lightning` alias. Use `transformers.Trainer` directly, or write your own training loop.

**In progress (lower level)**: `complex128` and native kernels that eliminate the remaining internal complex
bridge, PP/TP, memory-manager tuning, the CUDA 11/13 wheel families (an aligned CUDA 12.2 stack—cuDNN 8.9.7 or
newer—is already provided through `jittor[cuda12]`), and triton/tilelang custom-operator support.

User-visible limitations and their numbers in the known-issues ledger are collected in
[known limitations](../guides/known-limitations.en.md); where validation for each backend and Torch version
lives is in [platform support](../guides/platform-support.en.md).
