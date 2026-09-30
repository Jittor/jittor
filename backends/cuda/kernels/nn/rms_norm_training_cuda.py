"""CUDA training fast path for standard RMS normalization."""

from functools import lru_cache
import math

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import native_rule, optional_kernel

from .rms_norm_cuda import _autocast_enabled


def _gamma_segments(hidden_size, rows):
    """Row segments of the gamma gradient: about a thousand blocks, none with
    fewer than eight rows a thread."""
    columns = -(-hidden_size // 32)
    return max(1, min(-(-1024 // columns), rows // 64))


@lru_cache(maxsize=128)
def _rms_norm_training_cuda_cls(hidden_size, epsilon):
    threads = 32
    while threads < min(hidden_size, 256):
        threads *= 2
    header = f"#include <{library_resource('cub', 'home')}cub/cub.cuh>"

    class RMSNormTrainingCUDA(jt.Function):
        def execute(self, x, gamma):
            rows = int(x.numel()) // hidden_size
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
                            static_cast<float>(x[index]) * row_rstd
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
        and not _autocast_enabled()
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
