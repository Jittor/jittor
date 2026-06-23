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
- **Encoders:** bert, roberta, electra, distilbert, albert.
- **Encoder-decoder:** t5, bart, mbart, pegasus.
- **Vision:** vit, deit, swin, convnext (plus `jittor.models` resnet/vgg/... and the
  native ViT).
- **diffusers generation:** `UNet2DModel` forward 1.1e-6 / backward 1.5e-6, a DDIM
  denoising loop 3e-5, `AutoencoderKL` encode+decode 1.4e-6 — i.e. jittor *generates*
  and the numbers match torch. Build via constructors / `from_config`; see limitations
  for loading *pretrained* checkpoints.

Real end-to-end training works: `transformers.Trainer` fine-tunes (loss decreases,
`grad_norm`/`clip_grad_norm_` apply), and CNNs train (every conv weight updates).
A regression suite covers ~30 architectures (`jittor.test.test_torch_hf_models`) and
the diffusers generation path (`jittor.test.test_diffusers`).

`jittor.models` provides the CNN classics plus a modern **Vision Transformer**
(`vit_b_16`/`vit_b_32`/`vit_l_16`). LLMs/diffusion models come from
`transformers`/`diffusers` directly.

## Error reporting

Op failures now surface the real cause (op type, input shapes/dtypes, `[Reason]`)
instead of the old "Wrong inputs arguments / help(jt.sync)" noise. Unsupported
dtypes (e.g. float64 on Ascend) raise a clean Python error instead of aborting;
set `JT_SYNC=1` to pinpoint an async op failure.

## Status / limitations

Done + verified on both backends: forward/backward/training accuracy parity across
~30 transformers + CNN + diffusers-generation models, device dispatch, bf16/mixed
precision, DDP-without-mpirun, checkpoint/safetensors migration, clear errors, ViT.

Known limitation — **loading *pretrained* checkpoints via `from_pretrained` for
models that use accelerate's fast path** (diffusers, and transformers with
`low_cpu_mem_usage=True`): accelerate constructs parameters on a `meta` device then
fills each via `set_module_tensor_to_device`, which jittor doesn't yet emulate
(jittor params are real, not meta, so some get skipped → wrong weights). Workaround:
`from_config`/explicit construction works, and a name-keyed `safetensors`/`torch.load`
state-dict loads correctly; full `from_pretrained` for these models needs jittor
meta-device parameter emulation (tracked).

In progress: meta-device emulation, PP/TP, memory-manager tuning, cuDNN9, complex
dtype, full pypi-based CUDA/cuDNN dependency resolution, a jittor-lightning, and
triton/tilelang custom-op support.
