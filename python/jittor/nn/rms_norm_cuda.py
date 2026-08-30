"""CUDA inference kernels for multi-head RMS normalization."""

import math

import jittor as jt

from ._cuda_inference import cached_source, device_index


def _autocast_enabled():
    probe = getattr(jt, "is_autocast_enabled", None)
    if not callable(probe):
        return False
    try:
        return bool(probe())
    except Exception:
        return True


def _rms_norm_contract(x, gamma, epsilon, residual=None):
    if not isinstance(x, jt.Var) or not isinstance(gamma, jt.Var):
        return None
    if residual is not None and not isinstance(residual, jt.Var):
        return None
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0):
        return None
    try:
        if _autocast_enabled():
            return None
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        x_device = device_index(x)
        gamma_device = device_index(gamma)
        epsilon_value = float(epsilon)
        residual_device = (
            x_device if residual is None else device_index(residual))
    except Exception:
        return None
    if not x_shape or any(size <= 0 for size in x_shape):
        return None
    hidden_size = x_shape[-1]
    if hidden_size > 4096 or gamma_shape != (hidden_size,):
        return None
    if x_device < 0 or gamma_device != x_device or residual_device != x_device:
        return None
    if residual is not None and tuple(int(size) for size in residual.shape) != x_shape:
        return None
    if str(x.dtype) not in ("float16", "bfloat16", "float32"):
        return None
    if residual is not None and str(residual.dtype) != str(x.dtype):
        return None
    if str(gamma.dtype) not in ("float16", "bfloat16", "float32"):
        return None
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        return None

    threads = 32
    while threads < min(hidden_size, 1024):
        threads *= 2
    return hidden_size, threads, threads // 32, epsilon_value


def _rms_norm_cuda(x, gamma, epsilon=1e-6):
    """Inference-only fused CUDA RMSNorm, or ``None`` when unsupported."""
    contract = _rms_norm_contract(x, gamma, epsilon)
    if contract is None:
        return None
    hidden_size, threads, warps, epsilon_value = contract
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void rms_norm(
            const in0_type* x, const in1_type* gamma, out0_type* y) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(warps)d];
        __shared__ float inverse_rms;
        float sum = 0.0f;
        for (int dim = tid; dim < %(hidden_size)d; dim += blockDim.x) {
            float value = static_cast<float>(x[row * %(hidden_size)d + dim]);
            sum += value * value;
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(warps)d ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0)
                inverse_rms = rsqrtf(total / %(hidden_size)d.0f + %(epsilon).9gf);
        }
        __syncthreads();
        for (int dim = tid; dim < %(hidden_size)d; dim += blockDim.x) {
            int index = row * %(hidden_size)d + dim;
            y[index] = out0_type(static_cast<float>(x[index]) * inverse_rms
                                 * static_cast<float>(gamma[dim]));
        }
    }
    int rows = in0->num / %(hidden_size)d;
    rms_norm<<<rows, %(threads)d>>>(in0_p, in1_p, out0_p);
    CHECK(0 == cudaGetLastError());
    """, {
        "epsilon": epsilon_value,
        "hidden_size": hidden_size,
        "threads": threads,
        "warps": warps,
    })
    return jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src)


def _fused_add_rms_norm_cuda(x, residual, gamma, epsilon=1e-6):
    """Inference-only fused residual add and CUDA RMSNorm."""
    contract = _rms_norm_contract(x, gamma, epsilon, residual=residual)
    if contract is None:
        return None
    hidden_size, threads, warps, epsilon_value = contract
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void fused_add_rms_norm(
            const in0_type* x, const in1_type* residual,
            const in2_type* gamma, out0_type* y, out1_type* residual_out) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(warps)d];
        __shared__ float inverse_rms;
        float sum = 0.0f;
        for (int dim = tid; dim < %(hidden_size)d; dim += blockDim.x) {
            int index = row * %(hidden_size)d + dim;
            float value = static_cast<float>(x[index])
                        + static_cast<float>(residual[index]);
            sum += value * value;
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(warps)d ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0)
                inverse_rms = rsqrtf(total / %(hidden_size)d.0f + %(epsilon).9gf);
        }
        __syncthreads();
        for (int dim = tid; dim < %(hidden_size)d; dim += blockDim.x) {
            int index = row * %(hidden_size)d + dim;
            float value = static_cast<float>(x[index])
                        + static_cast<float>(residual[index]);
            residual_out[index] = out1_type(value);
            y[index] = out0_type(value * inverse_rms
                                 * static_cast<float>(gamma[dim]));
        }
    }
    int rows = in0->num / %(hidden_size)d;
    fused_add_rms_norm<<<rows, %(threads)d>>>(
        in0_p, in1_p, in2_p, out0_p, out1_p);
    CHECK(0 == cudaGetLastError());
    """, {
        "epsilon": epsilon_value,
        "hidden_size": hidden_size,
        "threads": threads,
        "warps": warps,
    })
    return jt.code(
        [x.shape, x.shape], [x.dtype, x.dtype], [x, residual, gamma],
        cuda_src=cuda_src,
    )


def multihead_rms_norm_cuda(x, gamma, scale=None, min_norm=1e-12):
    """Return a fused multi-head RMS normalization result when supported.

    ``x`` must end in ``(num_heads, head_dim)`` and ``gamma`` must have that
    exact shape.  The default scale is ``sqrt(head_dim)``, which turns the L2
    denominator into the usual root-mean-square denominator.  The kernel is an
    inference fast path and returns ``None`` when its CUDA/no-grad contract is
    not met, allowing callers to retain a differentiable fallback.
    """
    if not isinstance(x, jt.Var) or not isinstance(gamma, jt.Var):
        return None
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if getattr(jt.compiler, "has_acl", 0):
        return None
    if _autocast_enabled():
        return None
    if str(x.dtype) != "bfloat16" or str(gamma.dtype) != "float32":
        return None

    try:
        x_device = device_index(x)
        gamma_device = device_index(gamma)
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        scale_value = math.sqrt(float(x_shape[-1])) if scale is None else float(scale)
        min_norm_value = float(min_norm)
    except Exception:
        return None
    if x_device < 0 or gamma_device != x_device or len(x_shape) < 2:
        return None
    num_heads, head_dim = x_shape[-2:]
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim > 4096
        or gamma_shape != (num_heads, head_dim)
        or any(size <= 0 for size in x_shape[:-2])
    ):
        return None
    if not math.isfinite(scale_value) or not math.isfinite(min_norm_value) or min_norm_value <= 0:
        return None

    if head_dim <= 256:
        rows_per_block = 8
        cuda_src = cached_source(r"""
        __device__ __forceinline__ float warp_sum(float value) {
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }
        __global__ static void multihead_rms_norm(
                const in0_type* x, const in1_type* gamma, out0_type* y,
                int rows) {
            int warp = threadIdx.x >> 5;
            int lane = threadIdx.x & 31;
            int row = blockIdx.x * %(rows_per_block)d + warp;
            if (row >= rows) return;

            float sum = 0.0f;
            for (int dim = lane; dim < %(head_dim)d; dim += 32) {
                float value = static_cast<float>(
                    x[row * %(head_dim)d + dim]);
                sum += value * value;
            }
            sum = warp_sum(sum);
            float denominator = sqrtf(
                __shfl_sync(0xffffffff, sum, 0));
            if (denominator < %(min_norm).9g)
                denominator = %(min_norm).9g;
            float factor = %(scale).9g / denominator;

            int head = row %% %(num_heads)d;
            for (int dim = lane; dim < %(head_dim)d; dim += 32) {
                int index = row * %(head_dim)d + dim;
                float value = static_cast<float>(x[index]);
                float weight = static_cast<float>(
                    gamma[head * %(head_dim)d + dim]);
                y[index] = out0_type(value * weight * factor);
            }
        }
        int rows = in0->num / %(head_dim)d;
        multihead_rms_norm<<<
            (rows + %(rows_per_block)d - 1) / %(rows_per_block)d, 256>>>(
                in0_p, in1_p, out0_p, rows);
        CHECK(0 == cudaGetLastError());
        """, {
            "head_dim": head_dim,
            "min_norm": min_norm_value,
            "num_heads": num_heads,
            "rows_per_block": rows_per_block,
            "scale": scale_value,
        })
        return jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src)

    threads = 32
    while threads < min(head_dim, 1024):
        threads *= 2
    warps = threads // 32
    cuda_src = r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    })
    __global__ static void multihead_rms_norm(
            const in0_type* x, const in1_type* gamma, out0_type* y) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        __shared__ float warp_buf[%(warps)d];
        __shared__ float denominator;
        float sum = 0.0f;
        for (int dim = tid; dim < %(head_dim)d; dim += blockDim.x) {
            float value = static_cast<float>(x[row * %(head_dim)d + dim]);
            sum += value * value;
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_buf[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            float total = lane < %(warps)d ? warp_buf[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0) {
                denominator = sqrtf(total);
                if (denominator < %(min_norm).9g) denominator = %(min_norm).9g;
            }
        }
        __syncthreads();
        int head = row %% %(num_heads)d;
        for (int dim = tid; dim < %(head_dim)d; dim += blockDim.x) {
            float value = static_cast<float>(x[row * %(head_dim)d + dim]);
            float weight = static_cast<float>(
                gamma[head * %(head_dim)d + dim]);
            y[row * %(head_dim)d + dim] = out0_type(
                value / denominator * weight * %(scale).9g);
        }
    }
    int rows = in0->num / %(head_dim)d;
    multihead_rms_norm<<<rows, %(threads)d>>>(in0_p, in1_p, out0_p);
    CHECK(0 == cudaGetLastError());
    """ % {
        "warps": warps,
        "head_dim": head_dim,
        "min_norm": min_norm_value,
        "num_heads": num_heads,
        "scale": scale_value,
        "threads": threads,
    }
    return jt.code(x.shape, x.dtype, [x, gamma], cuda_src=cuda_src)


__all__ = ["multihead_rms_norm_cuda"]
