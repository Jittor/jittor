"""CUDA inference preprocessing for packed QKV tensors."""

import math

import jittor as jt

from ._cuda_inference import cached_source, device_index, on_acl


def packed_qkv_rms_rope_cuda(
    qkv,
    q_gamma,
    k_gamma,
    phases,
    *,
    scale=None,
    min_norm=1e-12,
):
    """Fuse packed-QKV RMS normalization and pairwise complex RoPE.

    ``qkv`` ends in ``(3, num_heads, head_dim)``. ``phases`` stores real and
    imaginary components in its final axis and covers every token preceding
    the packed axes. The inference-only kernel preserves the BF16 rounding
    point between RMS normalization and RoPE.
    """
    tensors = (qkv, q_gamma, k_gamma, phases)
    if not all(isinstance(value, jt.Var) for value in tensors):
        return None
    if not (jt.flags.use_cuda and getattr(jt.flags, "no_grad", 0)):
        return None
    if on_acl():
        return None
    autocast_probe = getattr(jt, "is_autocast_enabled", None)
    if callable(autocast_probe):
        try:
            if bool(autocast_probe()):
                return None
        except Exception:
            return None

    try:
        devices = tuple(device_index(value) for value in tensors)
        qkv_shape = tuple(int(size) for size in qkv.shape)
        q_gamma_shape = tuple(int(size) for size in q_gamma.shape)
        k_gamma_shape = tuple(int(size) for size in k_gamma.shape)
        phases_shape = tuple(int(size) for size in phases.shape)
        scale_value = (
            math.sqrt(float(qkv_shape[-1])) if scale is None else float(scale)
        )
        min_norm_value = float(min_norm)
    except Exception:
        return None
    if any(device < 0 for device in devices) or len(set(devices)) != 1:
        return None
    if len(qkv_shape) < 4 or qkv_shape[-3] != 3:
        return None
    num_heads, head_dim = qkv_shape[-2:]
    token_count = math.prod(qkv_shape[:-3])
    if (
        num_heads <= 0
        or head_dim <= 0
        or head_dim > 256
        or head_dim % 2
        or token_count <= 0
        or q_gamma_shape != (num_heads, head_dim)
        or k_gamma_shape != q_gamma_shape
        or len(phases_shape) < 2
        or phases_shape[-2:] != (head_dim // 2, 2)
        or math.prod(phases_shape[:-2]) != token_count
    ):
        return None
    if (
        str(qkv.dtype) != "bfloat16"
        or str(q_gamma.dtype) != "float32"
        or str(k_gamma.dtype) != "float32"
        or str(phases.dtype) != "float32"
    ):
        return None
    if (
        not math.isfinite(scale_value)
        or not math.isfinite(min_norm_value)
        or min_norm_value <= 0
    ):
        return None

    rows_per_block = 8
    cuda_src = cached_source(r"""
    __device__ __forceinline__ float warp_sum(float value) {
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        return value;
    }
    __global__ static void packed_qkv_rms_rope(
            const in0_type* qkv, const in1_type* q_gamma,
            const in2_type* k_gamma, const in3_type* phases,
            out0_type* output, int rows) {
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int row = blockIdx.x * %(rows_per_block)d + warp;
        if (row >= rows) return;

        int token = row / %(num_heads)d;
        int head = row %% %(num_heads)d;
        int token_stride = 3 * %(num_heads)d * %(head_dim)d;
        int head_stride = %(num_heads)d * %(head_dim)d;
        int q_base = token * token_stride + head * %(head_dim)d;
        int k_base = q_base + head_stride;
        int v_base = k_base + head_stride;

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int pair = lane; pair < %(pairs)d; pair += 32) {
            int dim = pair * 2;
            float q0 = static_cast<float>(qkv[q_base + dim]);
            float q1 = static_cast<float>(qkv[q_base + dim + 1]);
            float k0 = static_cast<float>(qkv[k_base + dim]);
            float k1 = static_cast<float>(qkv[k_base + dim + 1]);
            q_sum += q0 * q0 + q1 * q1;
            k_sum += k0 * k0 + k1 * k1;
        }
        q_sum = warp_sum(q_sum);
        k_sum = warp_sum(k_sum);
        float q_norm = sqrtf(__shfl_sync(0xffffffff, q_sum, 0));
        float k_norm = sqrtf(__shfl_sync(0xffffffff, k_sum, 0));
        if (q_norm < %(min_norm).9g) q_norm = %(min_norm).9g;
        if (k_norm < %(min_norm).9g) k_norm = %(min_norm).9g;
        float q_factor = %(scale).9g / q_norm;
        float k_factor = %(scale).9g / k_norm;

        int gamma_base = head * %(head_dim)d;
        int phase_base = token * %(head_dim)d;
        for (int pair = lane; pair < %(pairs)d; pair += 32) {
            int dim = pair * 2;
            out0_type q0 = out0_type(
                static_cast<float>(qkv[q_base + dim])
                * static_cast<float>(q_gamma[gamma_base + dim]) * q_factor);
            out0_type q1 = out0_type(
                static_cast<float>(qkv[q_base + dim + 1])
                * static_cast<float>(q_gamma[gamma_base + dim + 1]) * q_factor);
            out0_type k0 = out0_type(
                static_cast<float>(qkv[k_base + dim])
                * static_cast<float>(k_gamma[gamma_base + dim]) * k_factor);
            out0_type k1 = out0_type(
                static_cast<float>(qkv[k_base + dim + 1])
                * static_cast<float>(k_gamma[gamma_base + dim + 1]) * k_factor);
            float phase_real = static_cast<float>(phases[phase_base + dim]);
            float phase_imag = static_cast<float>(phases[phase_base + dim + 1]);
            output[q_base + dim] = out0_type(
                static_cast<float>(q0) * phase_real
                - static_cast<float>(q1) * phase_imag);
            output[q_base + dim + 1] = out0_type(
                static_cast<float>(q0) * phase_imag
                + static_cast<float>(q1) * phase_real);
            output[k_base + dim] = out0_type(
                static_cast<float>(k0) * phase_real
                - static_cast<float>(k1) * phase_imag);
            output[k_base + dim + 1] = out0_type(
                static_cast<float>(k0) * phase_imag
                + static_cast<float>(k1) * phase_real);
            output[v_base + dim] = qkv[v_base + dim];
            output[v_base + dim + 1] = qkv[v_base + dim + 1];
        }
    }
    int rows = %(token_count)d * %(num_heads)d;
    packed_qkv_rms_rope<<<
        (rows + %(rows_per_block)d - 1) / %(rows_per_block)d, 256>>>(
            in0_p, in1_p, in2_p, in3_p, out0_p, rows);
    CHECK(0 == cudaGetLastError());
    """, {
        "head_dim": head_dim,
        "min_norm": min_norm_value,
        "num_heads": num_heads,
        "pairs": head_dim // 2,
        "rows_per_block": rows_per_block,
        "scale": scale_value,
        "token_count": token_count,
    })
    return jt.code(qkv.shape, qkv.dtype, tensors, cuda_src=cuda_src)


__all__ = ["packed_qkv_rms_rope_cuda"]
