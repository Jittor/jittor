"""Vectorized CUDA training fast paths for exact and tanh-approximate GELU."""


_gelu_training_cuda = None

_GELU_EXACT_FWD = r"""
__device__ __forceinline__ float jt_gelu_exact_fwd(float v) {
    return (0.5f * v) * (1.0f + erff(v * 0.7071067811865476f));
}
__global__ static void gelu_forward_vec16(
        const in0_type* __restrict__ x, out0_type* __restrict__ y, long long n) {
    constexpr int VEC = 16 / sizeof(in0_type);
    long long nvec = n / VEC;
    long long stride = (long long)blockDim.x * gridDim.x;
    const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
    uint4* __restrict__ yv = reinterpret_cast<uint4*>(y);
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nvec; idx += stride) {
        uint4 in_raw = xv[idx];
        uint4 out_raw;
        const in0_type* ie = reinterpret_cast<const in0_type*>(&in_raw);
        out0_type* oe = reinterpret_cast<out0_type*>(&out_raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            oe[k] = out0_type(jt_gelu_exact_fwd(static_cast<float>(ie[k])));
        }
        yv[idx] = out_raw;
    }
    long long tail = nvec * VEC;
    for (long long i = tail + (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        y[i] = out0_type(jt_gelu_exact_fwd(static_cast<float>(x[i])));
    }
}
__global__ static void gelu_forward(
        const in0_type* __restrict__ x, out0_type* __restrict__ y, long long n) {
    long long start = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = start; i < n; i += stride) {
        y[i] = out0_type(jt_gelu_exact_fwd(static_cast<float>(x[i])));
    }
}
long long n = in0->num;
constexpr int VEC = 16 / sizeof(in0_type);
if (VEC > 1 && n >= VEC && ((((size_t)in0_p | (size_t)out0_p) & 15) == 0)) {
    long long blocks = ((n / VEC) + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_forward_vec16<<<(unsigned)blocks, 256>>>(in0_p, out0_p, n);
} else {
    long long blocks = (n + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_forward<<<(unsigned)blocks, 256>>>(in0_p, out0_p, n);
}
"""

_GELU_EXACT_BWD = r"""
__device__ __forceinline__ float jt_gelu_exact_bwd(float dy, float v) {
    float cdf = 0.5f * (1.0f + erff(v * 0.7071067811865476f));
    float pdf = 0.3989422804014327f * expf(-0.5f * v * v);
    return dy * (cdf + v * pdf);
}
__global__ static void gelu_backward_vec16(
        const in0_type* __restrict__ grad_y, const in1_type* __restrict__ x,
        out0_type* __restrict__ grad_x, long long n) {
    constexpr int VEC = 16 / sizeof(in0_type);
    long long nvec = n / VEC;
    long long stride = (long long)blockDim.x * gridDim.x;
    const uint4* __restrict__ gyv = reinterpret_cast<const uint4*>(grad_y);
    const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
    uint4* __restrict__ gxv = reinterpret_cast<uint4*>(grad_x);
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nvec; idx += stride) {
        uint4 gy_raw = gyv[idx];
        uint4 x_raw = xv[idx];
        uint4 gx_raw;
        const in0_type* gye = reinterpret_cast<const in0_type*>(&gy_raw);
        const in1_type* xe = reinterpret_cast<const in1_type*>(&x_raw);
        out0_type* gxe = reinterpret_cast<out0_type*>(&gx_raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            gxe[k] = out0_type(jt_gelu_exact_bwd(
                static_cast<float>(gye[k]), static_cast<float>(xe[k])));
        }
        gxv[idx] = gx_raw;
    }
    long long tail = nvec * VEC;
    for (long long i = tail + (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        grad_x[i] = out0_type(jt_gelu_exact_bwd(
            static_cast<float>(grad_y[i]), static_cast<float>(x[i])));
    }
}
__global__ static void gelu_backward(
        const in0_type* __restrict__ grad_y, const in1_type* __restrict__ x,
        out0_type* __restrict__ grad_x, long long n) {
    long long start = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = start; i < n; i += stride) {
        grad_x[i] = out0_type(jt_gelu_exact_bwd(
            static_cast<float>(grad_y[i]), static_cast<float>(x[i])));
    }
}
long long n = in0->num;
constexpr int VEC = 16 / sizeof(in0_type);
if (VEC > 1 && n >= VEC && ((((size_t)in0_p | (size_t)in1_p | (size_t)out0_p) & 15) == 0)) {
    long long blocks = ((n / VEC) + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_backward_vec16<<<(unsigned)blocks, 256>>>(in0_p, in1_p, out0_p, n);
} else {
    long long blocks = (n + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_backward<<<(unsigned)blocks, 256>>>(in0_p, in1_p, out0_p, n);
}
"""

_GELU_TANH_FWD = r"""
__device__ __forceinline__ float jt_gelu_tanh_fwd(float v) {
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(inner));
}
__global__ static void gelu_tanh_forward_vec16(
        const in0_type* __restrict__ x, out0_type* __restrict__ y, long long n) {
    constexpr int VEC = 16 / sizeof(in0_type);
    long long nvec = n / VEC;
    long long stride = (long long)blockDim.x * gridDim.x;
    const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
    uint4* __restrict__ yv = reinterpret_cast<uint4*>(y);
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nvec; idx += stride) {
        uint4 in_raw = xv[idx];
        uint4 out_raw;
        const in0_type* ie = reinterpret_cast<const in0_type*>(&in_raw);
        out0_type* oe = reinterpret_cast<out0_type*>(&out_raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            oe[k] = out0_type(jt_gelu_tanh_fwd(static_cast<float>(ie[k])));
        }
        yv[idx] = out_raw;
    }
    long long tail = nvec * VEC;
    for (long long i = tail + (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        y[i] = out0_type(jt_gelu_tanh_fwd(static_cast<float>(x[i])));
    }
}
__global__ static void gelu_tanh_forward(
        const in0_type* __restrict__ x, out0_type* __restrict__ y, long long n) {
    long long start = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = start; i < n; i += stride) {
        y[i] = out0_type(jt_gelu_tanh_fwd(static_cast<float>(x[i])));
    }
}
long long n = in0->num;
constexpr int VEC = 16 / sizeof(in0_type);
if (VEC > 1 && n >= VEC && ((((size_t)in0_p | (size_t)out0_p) & 15) == 0)) {
    long long blocks = ((n / VEC) + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_tanh_forward_vec16<<<(unsigned)blocks, 256>>>(in0_p, out0_p, n);
} else {
    long long blocks = (n + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_tanh_forward<<<(unsigned)blocks, 256>>>(in0_p, out0_p, n);
}
"""

_GELU_TANH_BWD = r"""
__device__ __forceinline__ float jt_gelu_tanh_bwd(float dy, float v) {
    float v2 = v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * v2 * v);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = 0.7978845608028654f * (1.0f + 0.134145f * v2);
    return dy * (0.5f * (1.0f + t) + 0.5f * v * sech2 * d_inner);
}
__global__ static void gelu_tanh_backward_vec16(
        const in0_type* __restrict__ grad_y, const in1_type* __restrict__ x,
        out0_type* __restrict__ grad_x, long long n) {
    constexpr int VEC = 16 / sizeof(in0_type);
    long long nvec = n / VEC;
    long long stride = (long long)blockDim.x * gridDim.x;
    const uint4* __restrict__ gyv = reinterpret_cast<const uint4*>(grad_y);
    const uint4* __restrict__ xv = reinterpret_cast<const uint4*>(x);
    uint4* __restrict__ gxv = reinterpret_cast<uint4*>(grad_x);
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nvec; idx += stride) {
        uint4 gy_raw = gyv[idx];
        uint4 x_raw = xv[idx];
        uint4 gx_raw;
        const in0_type* gye = reinterpret_cast<const in0_type*>(&gy_raw);
        const in1_type* xe = reinterpret_cast<const in1_type*>(&x_raw);
        out0_type* gxe = reinterpret_cast<out0_type*>(&gx_raw);
        #pragma unroll
        for (int k = 0; k < VEC; ++k) {
            gxe[k] = out0_type(jt_gelu_tanh_bwd(
                static_cast<float>(gye[k]), static_cast<float>(xe[k])));
        }
        gxv[idx] = gx_raw;
    }
    long long tail = nvec * VEC;
    for (long long i = tail + (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        grad_x[i] = out0_type(jt_gelu_tanh_bwd(
            static_cast<float>(grad_y[i]), static_cast<float>(x[i])));
    }
}
__global__ static void gelu_tanh_backward(
        const in0_type* __restrict__ grad_y, const in1_type* __restrict__ x,
        out0_type* __restrict__ grad_x, long long n) {
    long long start = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = start; i < n; i += stride) {
        grad_x[i] = out0_type(jt_gelu_tanh_bwd(
            static_cast<float>(grad_y[i]), static_cast<float>(x[i])));
    }
}
long long n = in0->num;
constexpr int VEC = 16 / sizeof(in0_type);
if (VEC > 1 && n >= VEC && ((((size_t)in0_p | (size_t)in1_p | (size_t)out0_p) & 15) == 0)) {
    long long blocks = ((n / VEC) + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_tanh_backward_vec16<<<(unsigned)blocks, 256>>>(in0_p, in1_p, out0_p, n);
} else {
    long long blocks = (n + 255) / 256;
    if (blocks > 65536) blocks = 65536;
    gelu_tanh_backward<<<(unsigned)blocks, 256>>>(in0_p, in1_p, out0_p, n);
}
"""


def _install_gelu_training_cuda(jt, output_requires_grad, optional_kernel):
    """Register the kernel without adding this module to Jittor's import cycle."""

    def supports(x, approximate="none"):
        return (
            approximate in ("none", "tanh")
            and isinstance(x, jt.Var)
            and output_requires_grad(x)
            and x.numel() > 0
            and x._storage_is_contiguous()
        )

    class GeluTrainingCUDA(jt.Function):
        def execute(self, x):
            self.x = x
            return jt.code(x.shape, x.dtype, [x], cuda_src=_GELU_EXACT_FWD)

        def grad(self, grad_y):
            x = self.x
            if not grad_y._storage_is_contiguous():
                grad_y = grad_y.contiguous()
            return jt.code(x.shape, x.dtype, [grad_y, x], cuda_src=_GELU_EXACT_BWD)

    class GeluTanhTrainingCUDA(jt.Function):
        def execute(self, x):
            self.x = x
            return jt.code(x.shape, x.dtype, [x], cuda_src=_GELU_TANH_FWD)

        def grad(self, grad_y):
            x = self.x
            if not grad_y._storage_is_contiguous():
                grad_y = grad_y.contiguous()
            return jt.code(x.shape, x.dtype, [grad_y, x], cuda_src=_GELU_TANH_BWD)

    @optional_kernel(
        "nn.gelu.training",
        "cuda",
        dtypes=("float32", "float16", "bfloat16"),
        supports=supports,
    )
    def gelu_training_cuda(x, approximate="none"):
        if approximate == "tanh":
            return GeluTanhTrainingCUDA.apply(x)
        return GeluTrainingCUDA.apply(x)

    global _gelu_training_cuda
    _gelu_training_cuda = gelu_training_cuda
    return gelu_training_cuda
