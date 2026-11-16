# CUTLASS memory-efficient attention

[Core concepts](../index.en.md) · [中文](efficient-attention.md)

The CUDA backend can directly compile PyTorch's CUTLASS memory-efficient
attention headers without importing or linking native Torch. Configure two
independent source directories before use:

```bash
export JITTOR_EFFICIENT_ATTN_SRC=/path/to/pytorch
export JITTOR_CUTLASS_PATH=/path/to/cutlass
```

The first directory may be a PyTorch source root, `aten/src`, or an installed
package's `include` directory. It must contain
`ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`. The second
may be the CUTLASS root or its `include` directory. Building also requires CUDA,
a C++17 compiler and pybind11 headers. Retain source versions and licenses;
external sources are not copied into Jittor. Serialize the first build.
Artifacts live in the active JIT cache, identified by sources, headers,
toolchain and target architectures.

The source kernels require CUDA architecture SM80 or newer. The entry accepts
four-dimensional Q/K/V in FP32, FP16 or BF16, with head and
value dimensions divisible by 8 and at most 128. Boolean or additive masks
broadcast to `[batch, heads, queries, keys]`. Rectangular sequences, top-left
causal masks, dropout and first-order Q/K/V gradients are supported, without
storing the full score matrix. Additive masks must match the query dtype;
Torch CUDA autocast converts floating inputs to its selected precision. Mask
gradients, uncovered head layouts and dtypes use other enabled paths;
requesting only this backend raises an error when unavailable.

Native BF16 fused probability products require `allow_reduced_precision=True`;
the default retains stricter FP32 intermediate precision. The Torch frontend
follows its fused-attention convention. FP32 uses upstream
`OpMultiplyAddFastF32`, matching this native Torch CUDA kernel. The matmul TF32
switch does not rewrite that decomposition. Numerical validation still requires
an independent high-precision reference.

Within the enabled Torch frontend, select a backend explicitly:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

Both `sdpa_kernel` and legacy `torch.backends.cuda.sdp_kernel` save, apply and
restore switches. Explicit math bypasses fused attention; a request without an
available selected kernel raises an error. Automatic selection retains its
default policy. Backend sets are accepted; custom priority ordering is not
implemented.

Graph nodes own forward, backward, workspace and random state. Dropout backward
reads the forward's saved Philox state, and CUDA graph replay refreshes random
values through device state. Test outputs, all input gradients, fully masked
rows, replay with changing inputs/cotangents, and fresh dropout masks.
