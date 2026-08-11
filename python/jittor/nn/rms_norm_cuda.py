"""CUDA inference kernels for multi-head RMS normalization."""

import math

import jittor as jt


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
    try:
        if bool(jt.is_autocast_enabled()):
            return None
    except Exception:
        return None
    if str(x.dtype) != "bfloat16" or str(gamma.dtype) != "float32":
        return None

    try:
        x_device = int(x.get_device())
        gamma_device = int(gamma.get_device())
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

    threads = 32
    while threads < min(head_dim, 1024):
        threads *= 2
    warps = threads // 32
    cuda_src = r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
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
