# flash-attention on Jittor

Jittor's compat layer ships a bridge that builds the **official** flash-attention
CUDA kernels and calls them on Jittor Vars, so code written against
`flash_attn` runs unchanged:

```python
import torch                                    # Jittor's compat shim
from flash_attn.flash_attn_interface import flash_attn_func

out = flash_attn_func(q, k, v, causal=True)     # q/k/v are Jittor-backed
```

The bridge lives in `compat/shim/backends/flash_attention/`. It compiles from a
flash-attention source checkout on first use and caches the result.

## Install

```bash
pip install pybind11                    # headers only; see below
git clone https://github.com/Dao-AILab/flash-attention
export JITTOR_FLASH_ATTN_JITTOR_SRC=/path/to/flash-attention
```

`JITTOR_FLASH_ATTN_JITTOR_SRC` is the only *variable* you need, but **pybind11
is a hard prerequisite**: the shim's `torch/extension.h` includes
`<pybind11/pybind11.h>` unconditionally, so without those headers no extension
compiles and the bridge reports `No module named 'flash_attn_jittor_cuda'` --
a symptom that names a missing Python module rather than a missing header. If
you cannot install into the environment, any directory holding
`pybind11/include` on `PYTHONPATH` is enough.

The checkout is **source**, not a built wheel -- the bridge generates and
compiles the kernels itself, so no `pip install flash-attn` build step and no
matching torch ABI.

`JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1` turns a bridge that fails to build into
an error instead of a silent fall back to another attention path. Worth setting
in a deployment, where a silent fallback is a performance mystery later.

## Check that it is actually being used

The bridge reports its own state; a fallback is otherwise invisible.

```python
from flash_attn.flash_attn_interface import (
    is_flashattn_jittor_available, flashattn_jittor_backend,
    flashattn_jittor_last_error)

print(is_flashattn_jittor_available())   # True
print(flashattn_jittor_backend())        # flashattn_jittor_official:/path/to/flash-attention
print(flashattn_jittor_last_error())     # None
```

`flashattn_jittor_last_error()` is the one to read when `available` is False --
it carries the build or import failure that the caller never saw.

## A run that proves it end to end

Measured output is in the comments, from a CUDA sm_90 build on 2026-09-22 with the official flash-attention checkout.

```python
import torch
from flash_attn.flash_attn_interface import flash_attn_func

B, S, H, D = 2, 256, 8, 64
q = torch.randn(B, S, H, D, device="cuda:0", dtype=torch.float16)
k = torch.randn(B, S, H, D, device="cuda:0", dtype=torch.float16)
v = torch.randn(B, S, H, D, device="cuda:0", dtype=torch.float16)

out = flash_attn_func(q, k, v, causal=True)      # (2, 256, 8, 64) float16

ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
).transpose(1, 2)
print((out.float() - ref.float()).abs().max().item())   # 0.001953125
```

**Not zero, and it should not be.** Flash-attention accumulates the softmax in
a different order than the math path, so in float16 the two differ by about
`2**-9` on this shape. Compare against a tolerance; an equality assertion here
fails for a reason that has nothing wrong with it.

## The variables that only make startup faster

These are **not** requirements. The bridge already learns the head dimension
and dtype it needs from the call
(`_ensure_capability_compile_env` merges them before the backend loads), so
leaving them unset costs a build on first use of each new combination, not a
failure.

```bash
export JITTOR_FLASH_ATTN_HEAD_DIMS=64,128      # or `all`
export JITTOR_FLASH_ATTN_DTYPES=bf16,fp16      # or `all`
```

Set them when you already know the shapes -- a server that would otherwise
compile a kernel in the middle of its first request -- and set them to `all` if
you would rather pay once and never think about it.

`JITTOR_FLASH_ATTN_CAST_FLOAT32=bf16` decides what a float32 q/k/v is cast to,
since the kernels are half-precision only. Unset, a float32 input does not
reach the bridge.

## Gotchas

* **`from flash_attn_interface import ...` (the top-level spelling) is a
  different module.** The bridged implementation is
  `flash_attn.flash_attn_interface`; the bare top-level name is what the
  upstream PyTorch package installs. With a flash-attention checkout on
  `PYTHONPATH` the bare import can resolve to the unbridged extension, and then
  every Jittor Var handed to it fails `TORCH_CHECK(x.is_cuda())`. If a
  dependency insists on the top-level name, put a module of that name on
  `PYTHONPATH` that re-exports from `flash_attn.flash_attn_interface`.
* **`available: False` with no error at the call site.** Read
  `flashattn_jittor_last_error()`; the caller usually fell back to SDPA and
  only got slower.
* **The first call compiles.** A first-use build of a new head-dim/dtype pair
  is not a latency number. Warm it up, or pre-seed the two variables above.

## Where this is used

`examples/minimax-h3/README.md` serves MiniMax-H3 with
`--diffusion-attention-backend FLASH_ATTN` through this bridge.
