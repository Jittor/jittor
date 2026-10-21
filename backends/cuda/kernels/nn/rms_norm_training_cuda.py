"""CUDA training fast path for standard RMS normalization."""

from functools import lru_cache
import math

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import dispatch_context, native_rule, optional_kernel
from jittor._core.dtypes import dtype_name as _dtype_name

from .rms_norm_cuda import _autocast_enabled


def _supports_vector4_fp32(x, gamma, epsilon):
    """CUDA FP32 width1024 only; all other contracts retain their providers."""
    if not isinstance(x, jt.Var) or not isinstance(gamma, jt.Var):
        return False
    if _dtype_name(x.dtype) != "float32" or _dtype_name(gamma.dtype) != "float32":
        return False
    if dispatch_context(x, gamma).backend != "cuda":
        return False
    try:
        shape = tuple(int(size) for size in x.shape)
        weight_shape = tuple(int(size) for size in gamma.shape)
        epsilon = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        return False
    return (bool(shape) and all(size > 0 for size in shape)
            and shape[-1] == 1024 and weight_shape == (1024,)
            and x._storage_is_contiguous() and gamma._storage_is_contiguous()
            and math.isfinite(epsilon)
            and 1.1754943508222875e-38 <= epsilon <= 3.4028234663852886e38)


def _vector4_fp32_forward(x, gamma, epsilon):
    """One shared forward used by inference and the existing training Function.

    Four independent accumulators and ascending warp shuffle offsets match the
    public square/mean equation's tested CUDA reduction order. Explicit rounded
    arithmetic prevents the JIT's fast-math flags from contracting square+sum.
    """
    rows = int(x.numel()) // 1024
    return jt.code([x.shape, (rows,)], ["float32", "float32"], [x, gamma],
        cuda_src=f"""
        __global__ static void rms_norm_vector4_fp32(
                const in0_type* x, const in1_type* w,
                out0_type* y, out1_type* rstd, int rows) {{
            int row = blockIdx.x, lane = threadIdx.x;
            if (row >= rows) return;
            long long origin = (long long)row * 1024;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (int base = lane * 4; base < 1024; base += 128) {{
                float v0 = x[origin + base], v1 = x[origin + base + 1];
                float v2 = x[origin + base + 2], v3 = x[origin + base + 3];
                a0 = __fadd_rn(a0, __fmul_rn(v0, v0));
                a1 = __fadd_rn(a1, __fmul_rn(v1, v1));
                a2 = __fadd_rn(a2, __fmul_rn(v2, v2));
                a3 = __fadd_rn(a3, __fmul_rn(v3, v3));
            }}
            float total = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
            for (int offset = 1; offset < 32; offset *= 2)
                total = __fadd_rn(total, __shfl_down_sync(0xffffffff, total, offset));
            total = __shfl_sync(0xffffffff, total, 0);
            float mean = __fmul_rn(total, 1.f / 1024.f);
            float inv = rsqrtf(__fadd_rn(mean, {epsilon:.9e}f));
            if (lane == 0) rstd[row] = inv;
            for (int column = lane; column < 1024; column += 32) {{
                float normalized = __fmul_rn(x[origin + column], inv);
                y[origin + column] = __fmul_rn(w[column], normalized);
            }}
        }}
        int rows = in0->num / 1024;
        rms_norm_vector4_fp32<<<rows, 32>>>(in0_p, in1_p, out0_p, out1_p, rows);
        CHECK(0 == cudaGetLastError());
        """)


def _supports_vector4_fp32_inference(x, gamma, epsilon=1e-6):
    return (_supports_vector4_fp32(x, gamma, epsilon)
            and not _output_requires_grad(x, gamma))


@optional_kernel("nn.rms_norm.inference", ("cuda",), dtypes=("float32",),
                 priority=10, supports=_supports_vector4_fp32_inference)
def _rms_norm_vector4_fp32_cuda(x, gamma, epsilon=1e-6):
    output, _rstd = _vector4_fp32_forward(x, gamma, float(epsilon))
    return _stop_grad_outputs(output)


def _gamma_segments(hidden_size, rows):
    """Row segments of the gamma gradient: about a thousand blocks, none with
    fewer than eight rows a thread."""
    columns = -(-hidden_size // 32)
    return max(1, min(-(-1024 // columns), rows // 64))


#: As in layer_norm_training_cuda: rows at most this wide, when there are at
#: least this many, go a warp each, and the backward is one fused pass.
_WARP_ROW_LIMIT = 1024
_WARP_ROWS_MIN = 1024
_FUSED_BLOCKS = 256



def _warp_kernels(hidden, eps, backward):
    """The forward, or the fused backward, with a warp per row; see
    `layer_norm_training_cuda._warp_kernels`. (No comments inside the source:
    `jt.code` lifts the kernels out by their text.)"""
    per = -(-hidden // 32)
    allsum = """
    __device__ __forceinline__ float jt_rms_allsum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }
    """
    forward = f"""
    __global__ static void rms_norm_forward_warp(
            const in0_type* x, const in1_type* gamma, out0_type* y, out1_type* rstd,
            long long rows) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        const in0_type* xr = x + row * {hidden};
        float cache[{per}];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            cache[i] = j < {hidden} ? static_cast<float>(xr[j]) : 0.0f;
            sq += cache[i] * cache[i];
        }}
        float r = rsqrtf(jt_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
        if (lane == 0) rstd[row] = out1_type(r);
        out0_type* yr = y + row * {hidden};
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            // Rounded before the weight, as the inference kernel and ATen do.
            if (j < {hidden}) yr[j] = out0_type(static_cast<float>(out0_type(cache[i] * r))
                                                * static_cast<float>(gamma[j]));
        }}
    }}
    """
    backward_src = f"""
    __global__ static void rms_norm_backward_fused(
            const in0_type* grad_y, const in1_type* x, const in2_type* rstd,
            const in3_type* gamma, out0_type* grad_x, float* partial, long long rows) {{
        __shared__ float fold[8][{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        float acc[{per}], w[{per}];
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            acc[i] = 0.0f;
            w[i] = j < {hidden} ? static_cast<float>(gamma[j]) : 0.0f;
        }}
        for (long long row = (long long)blockIdx.x * 8 + warp; row < rows;
             row += (long long)gridDim.x * 8) {{
            const in0_type* gr = grad_y + row * {hidden};
            const in1_type* xr = x + row * {hidden};
            float r = static_cast<float>(rstd[row]);
            float dy[{per}], xhat[{per}];
            float sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                bool in = j < {hidden};
                dy[i] = in ? static_cast<float>(gr[j]) : 0.0f;
                xhat[i] = in ? static_cast<float>(xr[j]) * r : 0.0f;
                sum_gx += dy[i] * w[i] * xhat[i];
                acc[i] += dy[i] * xhat[i];
            }}
            float mean_gx = jt_rms_allsum(sum_gx) / {hidden}.0f;
            out0_type* dr = grad_x + row * {hidden};
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                if (j < {hidden}) dr[j] = out0_type(r * (dy[i] * w[i] - xhat[i] * mean_gx));
            }}
        }}
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden}) fold[warp][j] = acc[i];
        }}
        __syncthreads();
        for (int j = threadIdx.x; j < {hidden}; j += 256) {{
            float t = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) t += fold[k][j];
            partial[(long long)blockIdx.x * {hidden} + j] = t;
        }}
    }}
    __global__ static void rms_norm_backward_finish(
            const float* partial, out1_type* grad_gamma, int parts) {{
        __shared__ float fold[8][33];
        int channel = blockIdx.x * 32 + threadIdx.x;
        float t = 0.0f;
        if (channel < {hidden})
            for (int p = threadIdx.y; p < parts; p += 8)
                t += partial[(long long)p * {hidden} + channel];
        fold[threadIdx.y][threadIdx.x] = t;
        __syncthreads();
        if (threadIdx.y || channel >= {hidden}) return;
        t = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) t += fold[k][threadIdx.x];
        grad_gamma[channel] = out1_type(t);
    }}
    """
    return allsum + (backward_src if backward else forward)


@lru_cache(maxsize=128)
def _rms_norm_training_cuda_cls(hidden_size, epsilon):
    threads = 32
    while threads < min(hidden_size, 256):
        threads *= 2
    header = f"#include <{library_resource('cub', 'home')}cub/cub.cuh>"

    class RMSNormTrainingCUDA(jt.Function):
        def execute(self, x, gamma):
            if _supports_vector4_fp32(x, gamma, epsilon):
                y, rstd = _vector4_fp32_forward(x, gamma, epsilon)
                self.saved = x, rstd, gamma
                return y
            rows = int(x.numel()) // hidden_size
            self.warp = hidden_size <= _WARP_ROW_LIMIT and rows >= _WARP_ROWS_MIN
            if self.warp:
                y, rstd = jt.code(
                    [x.shape, (rows,)], [x.dtype, "float32"], [x, gamma],
                    cuda_src=_warp_kernels(hidden_size, epsilon, False) + f"""
                    long long rows = in0->num / {hidden_size};
                    rms_norm_forward_warp<<<(unsigned)((rows + 7) / 8), 256>>>(
                        in0_p, in1_p, out0_p, out1_p, rows);
                    CHECK(0 == cudaGetLastError());
                    """)
                self.saved = x, rstd, gamma
                return y
            y, rstd = jt.code(
                [x.shape, (rows,)],
                [x.dtype, "float32"],
                [x, gamma],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void rms_norm_forward(
                        const in0_type* x, const in1_type* gamma,
                        out0_type* y, out1_type* rstd, int rows) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float rstd_shared;
                    int row = blockIdx.x;
                    if (row >= rows) return;
                    int base = row * {hidden_size};
                    float local = 0.0f;
                    for (int j = threadIdx.x; j < {hidden_size}; j += blockDim.x) {{
                        float value = static_cast<float>(x[base + j]);
                        local += value * value;
                    }}
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0) {{
                        rstd_shared = rsqrtf(
                            reduced / {hidden_size}.0f + {epsilon:.9g}f);
                        rstd[row] = out1_type(rstd_shared);
                    }}
                    __syncthreads();
                    float row_rstd = rstd_shared;
                    for (int j = threadIdx.x; j < {hidden_size}; j += blockDim.x) {{
                        int index = base + j;
                        y[index] = out0_type(
                            static_cast<float>(out0_type(static_cast<float>(x[index]) * row_rstd))
                            * static_cast<float>(gamma[j]));
                    }}
                }}
                int rows = in0->num / {hidden_size};
                rms_norm_forward<<<rows, {threads}>>>(
                    in0_p, in1_p, out0_p, out1_p, rows);
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, rstd, gamma
            return y

        def grad(self, grad_y):
            x, rstd, gamma = self.saved
            rows = int(grad_y.numel()) // hidden_size
            if self.warp:
                blocks = min(_FUSED_BLOCKS, -(-rows // 8))
                grad_x, grad_gamma, _ = jt.code(
                    [grad_y.shape, gamma.shape, (blocks * hidden_size,)],
                    [grad_y.dtype, gamma.dtype, "float32"],
                    [grad_y, x, rstd, gamma],
                    cuda_src=_warp_kernels(hidden_size, epsilon, True) + f"""
                    long long rows = in0->num / {hidden_size};
                    rms_norm_backward_fused<<<{blocks}, 256>>>(
                        in0_p, in1_p, in2_p, in3_p, out0_p, out2_p, rows);
                    rms_norm_backward_finish<<<{-(-hidden_size // 32)}, dim3(32, 8)>>>(
                        out2_p, out1_p, {blocks});
                    CHECK(0 == cudaGetLastError());
                    """)
                return grad_x, grad_gamma
            # The gamma gradient is a column sum. A block is 32 channels by 8
            # rows, so a warp reads 32 consecutive channels of one row, and the
            # rows are cut into segments whose partial sums a second kernel
            # adds in a fixed order. It was one block per channel walking the
            # rows, a warp touching 32 cache lines for 32 floats: 62 us a
            # call on Qwen3, a quarter of the bandwidth.
            segments = _gamma_segments(hidden_size, rows)
            per_segment = -(-rows // segments)
            grad_x, grad_gamma, partial = jt.code(
                [grad_y.shape, gamma.shape, (segments * hidden_size,)],
                [grad_y.dtype, gamma.dtype, "float32"],
                [grad_y, x, rstd, gamma],
                cuda_header=header,
                cuda_src=f"""
                __global__ static void rms_norm_backward_x(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* rstd, const in3_type* gamma,
                        out0_type* grad_x, int rows) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    __shared__ float mean_gx_shared;
                    int row = blockIdx.x;
                    if (row >= rows) return;
                    int base = row * {hidden_size};
                    float row_rstd = static_cast<float>(rstd[row]);
                    float local = 0.0f;
                    for (int j = threadIdx.x; j < {hidden_size}; j += blockDim.x) {{
                        int index = base + j;
                        float normalized = static_cast<float>(x[index]) * row_rstd;
                        local += static_cast<float>(grad_y[index])
                            * static_cast<float>(gamma[j]) * normalized;
                    }}
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        mean_gx_shared = reduced / {hidden_size}.0f;
                    __syncthreads();
                    float mean_gx = mean_gx_shared;
                    for (int j = threadIdx.x; j < {hidden_size}; j += blockDim.x) {{
                        int index = base + j;
                        float normalized = static_cast<float>(x[index]) * row_rstd;
                        float g = static_cast<float>(grad_y[index])
                            * static_cast<float>(gamma[j]);
                        grad_x[index] = out0_type(
                            row_rstd * (g - normalized * mean_gx));
                    }}
                }}

                __global__ static void rms_norm_backward_gamma_partial(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* rstd, float* partial) {{
                    __shared__ float sums[8][33];
                    int channel = blockIdx.x * 32 + threadIdx.x;
                    int begin = blockIdx.y * {per_segment};
                    int end = min(begin + {per_segment}, {rows});
                    float local = 0.0f;
                    if (channel < {hidden_size})
                        for (int row = begin + threadIdx.y; row < end; row += 8) {{
                            long long index = (long long)row * {hidden_size} + channel;
                            local += static_cast<float>(grad_y[index])
                                * static_cast<float>(x[index])
                                * static_cast<float>(rstd[row]);
                        }}
                    sums[threadIdx.y][threadIdx.x] = local;
                    __syncthreads();
                    if (threadIdx.y == 0 && channel < {hidden_size}) {{
                        float total = 0.0f;
                        for (int k = 0; k < 8; k++) total += sums[k][threadIdx.x];
                        partial[blockIdx.y * {hidden_size} + channel] = total;
                    }}
                }}

                __global__ static void rms_norm_backward_gamma_finish(
                        const float* partial, out1_type* grad_gamma) {{
                    int channel = blockIdx.x * blockDim.x + threadIdx.x;
                    if (channel >= {hidden_size}) return;
                    float total = 0.0f;
                    for (int s = 0; s < {segments}; s++)
                        total += partial[s * {hidden_size} + channel];
                    grad_gamma[channel] = out1_type(total);
                }}

                int rows = in0->num / {hidden_size};
                rms_norm_backward_x<<<rows, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, out0_p, rows);
                rms_norm_backward_gamma_partial<<<dim3({-(-hidden_size // 32)}, {segments}),
                                                  dim3(32, 8)>>>(
                    in0_p, in1_p, in2_p, out2_p);
                rms_norm_backward_gamma_finish<<<{-(-hidden_size // 256)}, 256>>>(
                    out2_p, out1_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_gamma

    return RMSNormTrainingCUDA


@native_rule("rms_norm_training")
def _supports_rms_norm_training(x, gamma, epsilon=1e-6):
    if not (
        isinstance(x, jt.Var)
        and isinstance(gamma, jt.Var)
        and _output_requires_grad(x, gamma)
        and (not _autocast_enabled()
             or _supports_vector4_fp32(x, gamma, epsilon))
    ):
        return False
    try:
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        return False
    if not x_shape or any(size <= 0 for size in x_shape):
        return False
    hidden_size = x_shape[-1]
    if (
        hidden_size > 4096
        or gamma_shape != (hidden_size,)
        or not math.isfinite(epsilon_value)
        or epsilon_value <= 0.0
    ):
        return False
    return True


@optional_kernel("nn.rms_norm.training", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32",),
                 supports=_supports_rms_norm_training)
def _rms_norm_training_cuda(x, gamma, epsilon=1e-6):
    cls = _rms_norm_training_cuda_cls(int(x.shape[-1]), float(epsilon))
    return cls.apply(x, gamma)
