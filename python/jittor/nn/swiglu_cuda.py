"""Private CUDA inference kernels for gated activations."""

import jittor as jt

from ._cuda_inference import device_index


def _silu_and_mul_cuda(x):
    """Return fused ``silu(x[..., :d]) * x[..., d:]`` for CUDA inference."""
    if not isinstance(x, jt.Var):
        return None
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0) or device_index(x) < 0:
        return None
    if str(x.dtype) not in ("float16", "bfloat16", "float32"):
        return None
    try:
        shape = tuple(int(size) for size in x.shape)
    except Exception:
        return None
    if not shape or any(size <= 0 for size in shape) or shape[-1] % 2:
        return None
    gated_size = shape[-1] // 2
    output_shape = shape[:-1] + (gated_size,)
    cuda_src = r"""
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
    """ % {
        "gated_size": gated_size,
        "input_size": shape[-1],
    }
    return jt.code(output_shape, x.dtype, [x], cuda_src=cuda_src)


__all__ = []
