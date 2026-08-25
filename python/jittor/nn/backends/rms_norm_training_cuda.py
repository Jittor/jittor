"""CUDA training fast path for standard RMS normalization."""

from functools import lru_cache
import math

import jittor as jt

from ..rms_norm_cuda import _autocast_enabled


@lru_cache(maxsize=128)
def _rms_norm_training_cuda_cls(hidden_size, epsilon):
    threads = 32
    while threads < min(hidden_size, 256):
        threads *= 2
    header = f"#include <{jt.compile_extern.cub_home}cub/cub.cuh>"

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
            grad_x, grad_gamma = jt.code(
                [grad_y.shape, gamma.shape],
                [grad_y.dtype, gamma.dtype],
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

                __global__ static void rms_norm_backward_gamma(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* rstd, out1_type* grad_gamma,
                        int rows) {{
                    typedef cub::BlockReduce<float, {threads}> BlockReduce;
                    __shared__ typename BlockReduce::TempStorage storage;
                    int channel = blockIdx.x;
                    float local = 0.0f;
                    for (int row = threadIdx.x; row < rows; row += blockDim.x) {{
                        int index = row * {hidden_size} + channel;
                        local += static_cast<float>(grad_y[index])
                            * static_cast<float>(x[index])
                            * static_cast<float>(rstd[row]);
                    }}
                    float reduced = BlockReduce(storage).Sum(local);
                    if (threadIdx.x == 0)
                        grad_gamma[channel] = out1_type(reduced);
                }}

                int rows = in0->num / {hidden_size};
                rms_norm_backward_x<<<rows, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, out0_p, rows);
                rms_norm_backward_gamma<<<{hidden_size}, {threads}>>>(
                    in0_p, in1_p, in2_p, out1_p, rows);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_gamma

    return RMSNormTrainingCUDA


def _rms_norm_training_cuda(x, gamma, epsilon=1e-6):
    if not (
        isinstance(x, jt.Var)
        and isinstance(gamma, jt.Var)
        and jt.flags.use_cuda
        and not getattr(jt.flags, "no_grad", 0)
        and not getattr(jt.compiler, "has_acl", 0)
        and not _autocast_enabled()
        and str(x.dtype) == "float32"
        and str(gamma.dtype) == "float32"
    ):
        return None
    try:
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        epsilon_value = float(epsilon)
    except Exception:
        return None
    if not x_shape or any(size <= 0 for size in x_shape):
        return None
    hidden_size = x_shape[-1]
    if (
        hidden_size > 4096
        or gamma_shape != (hidden_size,)
        or not math.isfinite(epsilon_value)
        or epsilon_value <= 0.0
    ):
        return None
    cls = _rms_norm_training_cuda_cls(hidden_size, epsilon_value)
    return cls.apply(x, gamma)
