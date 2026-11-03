# flash-attention on Jittor

The integration now lives in its own repository:
**[JittorEcosystem/jittor-flash-attn](https://github.com/JittorEcosystem/jittor-flash-attn)**,
pinned to FlashAttention 2.7.4.post1 (commit `5231d95fe1`) with jittor-compat
2.0.2 (commit `0a23c2410`).

Jittor's compat layer still ships the bridge this example described
(`compat/shim/backends/flash_attention/`): it builds the official CUDA kernels
from a flash-attention checkout and calls them on Jittor Vars, so code written
against `flash_attn` runs unchanged.

```python
import torch                                    # Jittor's compat shim
from flash_attn.flash_attn_interface import flash_attn_func

out = flash_attn_func(q, k, v, causal=True)     # q/k/v are Jittor-backed
```

What moved to that repository, and is maintained there rather than here:

- the install steps and the `JITTOR_FLASH_ATTN_*` environment;
- the first-use compile cost, where its cache lives, and what invalidates it
  (a deployment prewarms with its `tools/prewarm.py` instead);
- the verified numbers: `flash_attn_func` forward+backward on a 4090 (sm89,
  bf16, head_dim 64) agreeing with a float32 SDPA reference to 3.4e-3 forward
  and 2.3e-3 backward, at 0.075 ms against 0.173 ms for Jittor's fused SDPA on
  the same shape.

Start with that repository's README; this page is a pointer.
