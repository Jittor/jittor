"""Private CUDA inference kernels for gated activations."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel

from ._inference import cached_source


def _silu_and_mul_supported(x):
    if not isinstance(x, jt.Var):
        return False
    if _output_requires_grad(x):
        return False
    if str(x.dtype) not in ("float16", "bfloat16", "float32"):
        return False
    try:
        shape = tuple(int(size) for size in x.shape)
    except Exception:
        return False
    return bool(shape) and all(size > 0 for size in shape) and shape[-1] % 2 == 0


@optional_kernel("nn.silu_and_mul", ("cuda", "rocm_legacy", "corex_legacy"),
                 supports=_silu_and_mul_supported)
def _silu_and_mul_cuda(x):
    """Return fused ``silu(x[..., :d]) * x[..., d:]`` for CUDA inference."""
    shape = tuple(int(size) for size in x.shape)
    gated_size = shape[-1] // 2
    output_shape = shape[:-1] + (gated_size,)
    cuda_src = cached_source(r"""
    __global__ static void silu_and_mul(
            const in0_type* x, out0_type* y, int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int dim = (int)(index %% %(gated_size)d);
            int64_t row = index / %(gated_size)d;
            int64_t input_base = row * %(input_size)d;
            float gate = static_cast<float>(x[input_base + dim]);
            float value = static_cast<float>(
                x[input_base + %(gated_size)d + dim]);
            y[index] = out0_type((gate / (1.0f + expf(-gate))) * value);
        }
    }
    int64_t total = out0->num;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    if (total) silu_and_mul<<<blocks, threads>>>(in0_p, out0_p, total);
    CHECK(0 == cudaGetLastError());
    """, {
        "gated_size": gated_size,
        "input_size": shape[-1],
    })
    return _stop_grad_outputs(
        jt.code(output_shape, x.dtype, [x], cuda_src=cuda_src))


__all__ = []
