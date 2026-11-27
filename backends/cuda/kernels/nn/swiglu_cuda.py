"""Private CUDA inference kernels for gated activations."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel

from ._inference import cached_source


def _silu_and_mul_supported(x):
    if not isinstance(x, jt.Var):
        return False
    if _output_requires_grad(x):
        return False
    if _jittor_dtype_name(x.dtype) not in ("float16", "bfloat16", "float32"):
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
    __global__ static void silu_and_mul_vec16(
            const in0_type* __restrict__ x, out0_type* __restrict__ y,
            int64_t total_vecs) {
        constexpr int VEC = 16 / sizeof(in0_type);
        constexpr int GATED_VECS = %(gated_size)d / VEC;
        constexpr int INPUT_VECS = %(input_size)d / VEC;
        const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
        uint4* __restrict__ yv = reinterpret_cast<uint4*>(y);
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total_vecs; index += stride) {
            int64_t row = index / GATED_VECS;
            int col = (int)(index - row * GATED_VECS);
            int64_t base = row * INPUT_VECS + col;
            uint4 g_raw = xv[base];
            uint4 v_raw = xv[base + GATED_VECS];
            uint4 out_raw;
            const in0_type* ge = reinterpret_cast<const in0_type*>(&g_raw);
            const in0_type* ve = reinterpret_cast<const in0_type*>(&v_raw);
            out0_type* oe = reinterpret_cast<out0_type*>(&out_raw);
            #pragma unroll
            for (int k = 0; k < VEC; ++k) {
                float gate = static_cast<float>(ge[k]);
                float value = static_cast<float>(ve[k]);
                float activated = static_cast<float>(
                    in0_type(gate / (1.0f + expf(-gate))));
                oe[k] = out0_type(activated * value);
            }
            yv[index] = out_raw;
        }
    }
    __global__ static void silu_and_mul(
            const in0_type* __restrict__ x, out0_type* __restrict__ y, int64_t total) {
        int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        int64_t stride = (int64_t)blockDim.x * gridDim.x;
        for (; index < total; index += stride) {
            int dim = (int)(index %% %(gated_size)d);
            int64_t row = index / %(gated_size)d;
            int64_t input_base = row * %(input_size)d;
            float gate = static_cast<float>(x[input_base + dim]);
            float value = static_cast<float>(
                x[input_base + %(gated_size)d + dim]);
            // Preserve the public SiLU dtype boundary before gating, as
            // silu(x[..., :d]) * x[..., d:] does without this fusion.
            float activated = static_cast<float>(
                in0_type(gate / (1.0f + expf(-gate))));
            y[index] = out0_type(activated * value);
        }
    }
    int64_t total = out0->num;
    int threads = 256;
    constexpr int VEC = 16 / sizeof(in0_type);
    if (total && VEC > 1 && (%(gated_size)d %% VEC == 0)
            && ((((size_t)in0_p | (size_t)out0_p) & 15) == 0)) {
        int64_t total_vecs = total / VEC;
        int blocks = (int)((total_vecs + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        silu_and_mul_vec16<<<blocks, threads>>>(in0_p, out0_p, total_vecs);
    } else if (total) {
        int blocks = (int)((total + threads - 1) / threads);
        if (blocks > 4096) blocks = 4096;
        silu_and_mul<<<blocks, threads>>>(in0_p, out0_p, total);
    }
    CHECK(0 == cudaGetLastError());
    """, {
        "gated_size": gated_size,
        "input_size": shape[-1],
    })
    return _stop_grad_outputs(
        jt.code(output_shape, x.dtype, [x], cuda_src=cuda_src))


__all__ = []
