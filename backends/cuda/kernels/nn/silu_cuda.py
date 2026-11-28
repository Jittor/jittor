"""Single-rounding vectorized CUDA SiLU forward for gradient-free dense inputs."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


def _supports_silu_no_grad(x, inplace=False):
    if _output_requires_grad(x) or not x._storage_is_contiguous():
        return False
    size = int(x.numel())
    if dtype_name(x.dtype) == "float32":
        return size > 0
    # The elementwise implementation can vectorize low-precision I/O and
    # fuse with adjacent ops. Use a separate launch for a materialized input
    # with at least one full block of eight-element vectors.
    return x.is_finished and size >= 256 * 8


_SILU_CUDA_SRC = r'''
__device__ __forceinline__ float jt_silu_scalar(float v) {
    return v / (1.0f + expf(-v));
}
__global__ static void jt_silu_vec16(const in0_type* __restrict__ x,
                                     out0_type* __restrict__ y, long long n) {
    constexpr int VEC = 16 / sizeof(in0_type);
    long long nvec = n / VEC;
    long long stride = (long long)gridDim.x * blockDim.x;
    const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
    uint4* __restrict__ yv = reinterpret_cast<uint4*>(y);
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nvec; idx += stride) {
        uint4 in_raw = xv[idx];
        uint4 out_raw;
        const in0_type* in_e = reinterpret_cast<const in0_type*>(&in_raw);
        out0_type* out_e = reinterpret_cast<out0_type*>(&out_raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            out_e[k] = static_cast<out0_type>(jt_silu_scalar(static_cast<float>(in_e[k])));
        }
        yv[idx] = out_raw;
    }
    long long tail = nvec * VEC;
    for (long long i = tail + (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        y[i] = static_cast<out0_type>(jt_silu_scalar(static_cast<float>(x[i])));
    }
}
__global__ static void jt_silu_forward(const in0_type* __restrict__ x,
                                       out0_type* __restrict__ y, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        y[i] = static_cast<out0_type>(jt_silu_scalar(static_cast<float>(x[i])));
    }
}
long long n = in0->num;
constexpr int VEC = 16 / sizeof(in0_type);
if (VEC > 1 && n >= VEC && ((((size_t)in0_p | (size_t)out0_p) & 15) == 0)) {
    int blocks = (int)std::min(((n / VEC) + 255) / 256, 65535LL);
    jt_silu_vec16<<<blocks, 256>>>(in0_p, out0_p, n);
} else {
    int blocks = (int)std::min((n + 255) / 256, 65535LL);
    jt_silu_forward<<<blocks, 256>>>(in0_p, out0_p, n);
}
'''


@optional_kernel("nn.silu", "cuda", dtypes={"float32", "float16", "bfloat16"},
                 supports=_supports_silu_no_grad)
def _silu_no_grad_cuda(x, inplace=False):
    return jt.code(x.shape, x.dtype, [x], cuda_src=_SILU_CUDA_SRC).stop_grad()
