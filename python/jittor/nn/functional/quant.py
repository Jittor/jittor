"""Native FP8 / MXFP8 quantization and quantized matrix multiplication ops."""

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import try_dispatch as _try_dispatch


FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0

GROUP_SIZE_PER_TENSOR = 0
GROUP_SIZE_BLOCK_128 = (1 << 32) | (128 << 16) | 128
GROUP_SIZE_MXFP8_32 = (1 << 32) | (1 << 16) | 32


def encode_group_size(m_block: int, n_block: int, k_block: int) -> int:
    """Encode (M, N, K) quantization group sizes into CANN QuantMatmulV5 format."""
    return (int(m_block) << 32) | (int(n_block) << 16) | int(k_block)


def _fp8_max(dtype_name: str) -> float:
    if dtype_name == "float8_e4m3fn":
        return FP8_E4M3_MAX
    if dtype_name == "float8_e5m2":
        return FP8_E5M2_MAX
    raise ValueError(f"unsupported FP8 compute dtype: {dtype_name}")


def dynamic_block_quant(
    x,
    row_block_size: int = 1,
    col_block_size: int = 128,
    dtype="float8_e4m3fn",
    min_scale: float = 0.0,
):
    """Dynamically quantize ``x`` per (row_block_size, col_block_size) block to FP8.

    Dispatches to ``aclnnDynamicBlockQuant`` on Ascend 950PR and uses a pure
    Jittor float32 reference implementation on CPU or other backends.
    """
    dst_name = _jittor_dtype_name(dtype)
    if dst_name not in ("float8_e4m3fn", "float8_e5m2"):
        raise ValueError(f"dynamic_block_quant expects float8_e4m3fn or float8_e5m2, got {dst_name}")
    dispatched = _try_dispatch(
        "nn.dynamic_block_quant",
        x,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
        dst_dtype=dst_name,
        min_scale=min_scale,
    )
    if dispatched is not None:
        return dispatched

    # Reference implementation (CPU / general backend).
    import jittor as jt

    orig_shape = tuple(int(s) for s in x.shape)
    row_blk = int(row_block_size)
    col_blk = int(col_block_size)
    x2d = x.reshape((-1, orig_shape[-1])).float32()
    m_dim, k_dim = int(x2d.shape[0]), int(x2d.shape[1])
    if m_dim % row_blk != 0 or k_dim % col_blk != 0:
        raise ValueError(
            f"dynamic_block_quant shape {orig_shape} not divisible by block ({row_blk}, {col_blk})"
        )
    m_blks = m_dim // row_blk
    k_blks = k_dim // col_blk
    blocks = x2d.reshape((m_blks, row_blk, k_blks, col_blk))
    amax = jt.abs(blocks).max(dim=3).max(dim=1)  # [m_blks, k_blks]
    fp8_max = _fp8_max(dst_name)
    scale = amax / fp8_max
    if min_scale > 0.0:
        scale = jt.maximum(scale, jt.array(float(min_scale), dtype="float32"))
    safe_scale = jt.where(scale == 0.0, jt.ones_like(scale), scale)
    scaled = blocks / safe_scale.reshape((m_blks, 1, k_blks, 1))
    q = scaled.reshape((m_dim, k_dim)).cast(dst_name)
    if len(orig_shape) != 2:
        q = q.reshape(orig_shape)
        if row_blk == 1:
            scale = scale.reshape(orig_shape[:-1] + (k_blks,))
    return q, scale


def dynamic_mx_quant(
    x,
    axis: int = -1,
    block_size: int = 32,
    dtype="float8_e4m3fn",
    scale_alg: int = 0,
):
    """Dynamically quantize ``x`` to OCP MXFP8 with paired UE8M0 scales.

    Returns ``(q, mx_scale)`` where ``mx_scale`` has dtype ``float8_e8m0fnu``
    and shape ``[..., K // (2 * block_size), 2]`` along ``axis``.
    Dispatches to ``aclnnDynamicMxQuant`` on Ascend 950PR.
    """
    dst_name = _jittor_dtype_name(dtype)
    if dst_name not in ("float8_e4m3fn", "float8_e5m2"):
        raise ValueError(f"dynamic_mx_quant expects float8_e4m3fn or float8_e5m2, got {dst_name}")
    dispatched = _try_dispatch(
        "nn.dynamic_mx_quant",
        x,
        axis=axis,
        block_size=block_size,
        dst_dtype=dst_name,
        scale_alg=scale_alg,
    )
    if dispatched is not None:
        return dispatched

    # Reference OCP MXFP8 implementation (CPU / general backend).
    import jittor as jt
    import numpy as np

    orig_shape = tuple(int(s) for s in x.shape)
    norm_axis = int(axis) if int(axis) >= 0 else int(axis) + len(orig_shape)
    blk = int(block_size)
    if norm_axis != len(orig_shape) - 1:
        perm = [i for i in range(len(orig_shape)) if i != norm_axis] + [norm_axis]
        x_perm = x.transpose(perm)
        q_perm, s_perm = dynamic_mx_quant(x_perm, axis=-1, block_size=blk, dtype=dst_name, scale_alg=scale_alg)
        inv_perm = [0] * len(orig_shape)
        for idx, p in enumerate(perm):
            inv_perm[p] = idx
        q_out = q_perm.transpose(inv_perm)
        s_inv = inv_perm + [len(orig_shape)]
        return q_out, s_perm.transpose(s_inv)

    k_dim = orig_shape[-1]
    if k_dim % (2 * blk) != 0:
        raise ValueError(f"dynamic_mx_quant last dim {k_dim} must be divisible by {2 * blk}")
    x_np = x.float32().numpy().reshape((-1, k_dim // blk, blk))
    amax = np.max(np.abs(x_np), axis=-1)  # [M, K // blk]
    # Shared exponent for E4M3FN (max=448 -> 2^8) or E5M2 (max=57344 -> 2^15).
    max_exp = 8 if dst_name == "float8_e4m3fn" else 15
    with np.errstate(divide="ignore"):
        raw_exp = np.floor(np.log2(np.maximum(amax, np.float32(2.0**-127)))).astype(np.int32) - max_exp
    biased = np.where(amax == 0.0, 0, np.clip(raw_exp + 127, 0, 254)).astype(np.uint8)
    scale_f32 = np.where(biased == 0, np.float32(2.0**-127), np.ldexp(np.float32(1.0), biased.astype(np.int32) - 127))
    safe_scale = np.where(amax == 0.0, np.float32(1.0), scale_f32)
    q_f32 = (x_np / safe_scale[..., None]).reshape(orig_shape)
    q = jt.array(q_f32, dtype="float32").cast(dst_name)
    scale_shape = orig_shape[:-1] + (k_dim // (2 * blk), 2)
    mx_scale = jt.array(biased.reshape(scale_shape), dtype="uint8").reinterpret_view(
        scale_shape, "float8_e8m0fnu"
    )
    return q, mx_scale


def quant_matmul(
    x1,
    x2,
    x1_scale,
    x2_scale,
    *,
    bias=None,
    out_dtype="bfloat16",
    trans_a: bool = False,
    trans_b: bool = False,
    trans_x2_scale: bool = False,
    group_size: int = 0,
):
    """Quantized FP8 / MXFP8 matrix multiplication ``(x1 * s1) @ (x2 * s2) + bias``.

    Supports:
    - Per-tensor / per-token / per-channel FP8 (``group_size=0``)
    - Block-wise FP8 (e.g. ``group_size=GROUP_SIZE_BLOCK_128``)
    - OCP MXFP8 1x32 with ``float8_e8m0fnu`` paired scales (``group_size=GROUP_SIZE_MXFP8_32``)

    Dispatches natively to ``aclnnQuantMatmulV5`` on Ascend 950PR with zero
    host fallback, and falls back cleanly to a float32 reference on CPU.
    """
    out_name = _jittor_dtype_name(out_dtype)
    dispatched = _try_dispatch(
        "nn.quant_matmul",
        x1,
        x2,
        x1_scale,
        x2_scale,
        bias=bias,
        out_dtype=out_name,
        trans_a=trans_a,
        trans_b=trans_b,
        trans_x2_scale=trans_x2_scale,
        group_size=group_size,
    )
    if dispatched is not None:
        return dispatched

    # Reference implementation (CPU / general backend).
    import jittor as jt

    orig_lead = tuple(int(s) for s in x1.shape[:-2])
    a = x1.reshape((-1, int(x1.shape[-1]))).float32()
    if trans_a:
        a = a.transpose((1, 0))
    b = x2.float32()
    if trans_b:
        b = b.transpose((1, 0))
    m_dim, k_dim = int(a.shape[0]), int(a.shape[1])
    n_dim = int(b.shape[1])

    s1_dt = _jittor_dtype_name(x1_scale.dtype)
    s2_dt = _jittor_dtype_name(x2_scale.dtype)
    gs = int(group_size)
    if gs == 0:
        s1 = x1_scale.float32()
        s2 = x2_scale.float32()
        if s1.numel() == 1:
            s1 = s1.reshape((1, 1))
        elif s1.ndim == 1:
            s1 = s1.reshape((m_dim, 1))
        if s2.numel() == 1:
            s2 = s2.reshape((1, 1))
        elif s2.ndim == 1:
            s2 = s2.reshape((1, n_dim))
        a = a * s1
        b = b * s2
    elif s1_dt == "float8_e8m0fnu" and s2_dt == "float8_e8m0fnu":
        k_blk = gs & 0xFFFF
        s1 = x1_scale.float32().reshape((m_dim, k_dim // k_blk, 1))
        a = (a.reshape((m_dim, k_dim // k_blk, k_blk)) * s1).reshape((m_dim, k_dim))
        s2_raw = x2_scale.float32()
        if trans_x2_scale:
            # Storage is [N, K/(2*k_blk), 2] -> [N, K/k_blk] -> transpose to [K/k_blk, N].
            s2 = s2_raw.reshape((n_dim, k_dim // k_blk)).transpose((1, 0))
        else:
            # Storage is [K/(2*k_blk), N, 2] -> permute [K/(2*k_blk), 2, N] -> [K/k_blk, N].
            s2 = s2_raw.transpose((0, 2, 1)).reshape((k_dim // k_blk, n_dim))
        b = (b.reshape((k_dim // k_blk, k_blk, n_dim)) * s2.reshape((k_dim // k_blk, 1, n_dim))).reshape((k_dim, n_dim))
    else:
        m_blk = (gs >> 32) & 0xFFFF
        n_blk = (gs >> 16) & 0xFFFF
        k_blk = gs & 0xFFFF
        s1 = x1_scale.float32().reshape((m_dim // m_blk, 1, k_dim // k_blk, 1))
        a = (a.reshape((m_dim // m_blk, m_blk, k_dim // k_blk, k_blk)) * s1).reshape((m_dim, k_dim))
        s2 = x2_scale.float32()
        if trans_x2_scale:
            s2 = s2.transpose((1, 0))
        s2 = s2.reshape((k_dim // k_blk, 1, n_dim // n_blk, 1))
        b = (b.reshape((k_dim // k_blk, k_blk, n_dim // n_blk, n_blk)) * s2).reshape((k_dim, n_dim))

    out = jt.matmul(a, b)
    if bias is not None:
        out = out + bias.float32()
    out = out.cast(out_name)
    if orig_lead:
        out = out.reshape(orig_lead + (int(x1.shape[-2]), n_dim))
    return out


_MAPPED_FP8_LINEAR = None
_FAST_FP8_LINEAR = None
_JT_FLAGS = None
_ACL_FP8_VERIFIED = False
_ACL_FP8_GEN = -1
_FP8_IN_DTYPES = ("bfloat16", "float16", "torch.bfloat16", "torch.float16")
_FP8_WT_DTYPES = ("float8_e4m3fn", "float8_e5m2", "torch.float8_e4m3fn", "torch.float8_e5m2")


def fp8_linear(
    x,
    weight_fp8,
    weight_scale,
    bias=None,
    *,
    mode: str = "block128",
    block_size: int = 128,
    out_dtype=None,
):
    """High-level FP8 linear projection ``x @ weight_fp8.T + bias``.

    Modes:
    - ``"block128"``: dynamic 1x128 block activation quantization + Block-128x128
      FP8 weights (matches DeepSeek-V3 / Qwen3.8-FP8 safetensors layout directly).
    - ``"mxfp8"``: dynamic 1x32 MXFP8 activation quantization + MXFP8 UE8M0 weights.
    """
    global _MAPPED_FP8_LINEAR, _FAST_FP8_LINEAR, _JT_FLAGS, _ACL_FP8_VERIFIED, _ACL_FP8_GEN
    if _ACL_FP8_VERIFIED and bias is None:
        from jittor._runtime import dispatch as _dispatch
        if (_dispatch.generation == _ACL_FP8_GEN
                and _dispatch.dispatch_context is _dispatch._NATIVE_DISPATCH_CONTEXT):
            mode_id = 1 if mode == "mxfp8" else (0 if mode == "block128" else -1)
            if mode_id >= 0 and _FAST_FP8_LINEAR is not None:
                fast = _FAST_FP8_LINEAR(
                    x, weight_fp8, weight_scale, mode_id,
                    32 if mode_id == 1 else block_size, out_dtype,
                )
                if fast is not None:
                    return fast
            if _JT_FLAGS.no_grad and (
                x.placement_backend == 2
                and weight_fp8.placement_backend == 2
                and weight_scale.placement_backend == 2
                and x._storage_is_contiguous()
                and weight_fp8._storage_is_contiguous()
                and weight_scale._storage_is_contiguous()
            ):
                if mode == "mxfp8":
                    wshape = weight_fp8.shape
                    sshape = weight_scale.shape
                    if (
                        len(wshape) == 2
                        and len(sshape) == 3
                        and x.ndim >= 2
                        and x.shape[-1] == wshape[1]
                        and sshape[0] == wshape[0]
                        and sshape[2] == 2
                        and sshape[1] * 64 == wshape[1]
                        and str(x.dtype) in _FP8_IN_DTYPES
                        and str(weight_fp8.dtype) in _FP8_WT_DTYPES
                        and str(weight_scale.dtype) in ("float8_e8m0fnu", "torch.float8_e8m0fnu")
                    ):
                        target = (
                            out_dtype
                            if out_dtype in ("bfloat16", "float16")
                            else (_jittor_dtype_name(x.dtype) if out_dtype is None else _jittor_dtype_name(out_dtype))
                        )
                        if target in ("bfloat16", "float16"):
                            return _MAPPED_FP8_LINEAR(x, weight_fp8, weight_scale, 1, 32, target)
                elif mode == "block128":
                    wshape = weight_fp8.shape
                    sshape = weight_scale.shape
                    if (
                        len(wshape) == 2
                        and len(sshape) == 2
                        and x.ndim >= 2
                        and x.shape[-1] == wshape[1]
                        and block_size > 0
                        and sshape[0] * block_size == wshape[0]
                        and sshape[1] * block_size == wshape[1]
                        and str(x.dtype) in _FP8_IN_DTYPES
                        and str(weight_fp8.dtype) in _FP8_WT_DTYPES
                        and str(weight_scale.dtype) in ("float32", "torch.float32")
                    ):
                        target = (
                            out_dtype
                            if out_dtype in ("bfloat16", "float16")
                            else (_jittor_dtype_name(x.dtype) if out_dtype is None else _jittor_dtype_name(out_dtype))
                        )
                        if target in ("bfloat16", "float16"):
                            return _MAPPED_FP8_LINEAR(x, weight_fp8, weight_scale, 0, block_size, target)

    target_dtype = _jittor_dtype_name(x.dtype) if out_dtype is None else _jittor_dtype_name(out_dtype)
    if mode not in ("block128", "mxfp8"):
        raise ValueError(f"unsupported fp8_linear mode: {mode!r}; expected 'block128' or 'mxfp8'")
    dispatched = _try_dispatch(
        "nn.fp8_linear",
        x,
        weight_fp8,
        weight_scale,
        bias,
        mode=mode,
        block_size=block_size,
        out_dtype=target_dtype,
    )
    if dispatched is not None:
        if (getattr(x, "placement_backend", -1) == 2
                and getattr(dispatched, "_producer_op", None) is not None):
            from jittor._runtime import dispatch as _dispatch
            if (not _ACL_FP8_VERIFIED or _ACL_FP8_GEN != _dispatch.generation):
                import jittor as jt
                entries = _dispatch._candidates("nn.fp8_linear", "acl")
                if (len(entries) == 1
                        and getattr(entries[0].implementation, "__name__", None) == "fp8_linear_acl"):
                    _MAPPED_FP8_LINEAR = jt.core.ops.mapped_fp8_linear
                    _FAST_FP8_LINEAR = getattr(jt.core, "_fast_fp8_linear", None)
                    _JT_FLAGS = jt.flags
                    _ACL_FP8_GEN = _dispatch.generation
                    _ACL_FP8_VERIFIED = True
        return dispatched
    if mode == "block128":
        qx, sx = dynamic_block_quant(
            x, row_block_size=1, col_block_size=block_size, dtype=weight_fp8.dtype
        )
        return quant_matmul(
            qx,
            weight_fp8,
            sx,
            weight_scale,
            bias=bias,
            out_dtype=target_dtype,
            trans_b=True,
            trans_x2_scale=True,
            group_size=encode_group_size(1, block_size, block_size),
        )
    qx, sx = dynamic_mx_quant(x, axis=-1, block_size=32, dtype=weight_fp8.dtype)
    return quant_matmul(
        qx,
        weight_fp8,
        sx,
        weight_scale,
        bias=bias,
        out_dtype=target_dtype,
        trans_b=True,
        trans_x2_scale=True,
        group_size=GROUP_SIZE_MXFP8_32,
    )


__all__ = [
    "FP8_E4M3_MAX",
    "FP8_E5M2_MAX",
    "GROUP_SIZE_BLOCK_128",
    "GROUP_SIZE_MXFP8_32",
    "GROUP_SIZE_PER_TENSOR",
    "dynamic_block_quant",
    "dynamic_mx_quant",
    "encode_group_size",
    "fp8_linear",
    "quant_matmul",
]
