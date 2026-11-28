"""Ascend 950PR native FP8 / MXFP8 ACL kernel wrappers."""

from jittor._core.dtypes import dtype_name as _jittor_dtype_name


_QUANT_PROGRAMS = {}


def _is_contiguous(v):
    checker = getattr(v, "_storage_is_contiguous", None)
    if checker is not None:
        return bool(checker())
    checker = getattr(v, "is_contiguous", None)
    return True if checker is None else bool(checker())


def _dynamic_block_quant_program(min_scale, row_block_size, col_block_size):
    from ._code import acl_program, scalar_key

    key = ("DynamicBlockQuant", scalar_key(float(min_scale)), int(row_block_size), int(col_block_size))
    program = _QUANT_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            "DynamicBlockQuant",
            1,
            2,
            attributes={
                "minScale": float(min_scale),
                "rowBlockSize": int(row_block_size),
                "colBlockSize": int(col_block_size),
            },
        )
        _QUANT_PROGRAMS[key] = program
    return program


def _dynamic_mx_quant_program(axis, block_size, scale_alg):
    from ._code import acl_program

    key = ("DynamicMxQuant", int(axis), int(block_size), int(scale_alg))
    program = _QUANT_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            "DynamicMxQuant",
            1,
            2,
            attributes={
                "axis": int(axis),
                "blockSize": int(block_size),
                "scaleAlg": int(scale_alg),
            },
        )
        _QUANT_PROGRAMS[key] = program
    return program


def _quant_matmul_program(num_inputs, trans_x1, trans_x2, trans_x2_scale, group_size, has_bias):
    from ._code import acl_program

    key = (
        "QuantMatmul",
        int(num_inputs),
        bool(trans_x1),
        bool(trans_x2),
        bool(trans_x2_scale),
        int(group_size),
        bool(has_bias),
    )
    program = _QUANT_PROGRAMS.get(key)
    if program is None:
        program = acl_program(
            "QuantMatmul",
            int(num_inputs),
            1,
            attributes={
                "transposeX1": bool(trans_x1),
                "transposeX2": bool(trans_x2),
                "transposeX2Scale": bool(trans_x2_scale),
                "groupSize": int(group_size),
                "hasBias": bool(has_bias),
            },
        )
        _QUANT_PROGRAMS[key] = program
    return program


class DynamicBlockQuantACL:
    def __call__(self, x, row_block_size=1, col_block_size=128, dst_dtype="float8_e4m3fn", min_scale=0.0):
        from ._code import acl_emit

        row_block_size = int(row_block_size)
        col_block_size = int(col_block_size)
        dst_name = _jittor_dtype_name(dst_dtype)
        orig_shape = tuple(int(s) for s in x.shape)
        if x.ndim != 2:
            x2d = x.reshape((-1, orig_shape[-1]))
        else:
            x2d = x
        if not _is_contiguous(x2d):
            x2d = x2d.contiguous()
        m_dim, k_dim = int(x2d.shape[0]), int(x2d.shape[1])
        scale_shape = (m_dim // row_block_size, k_dim // col_block_size)
        program = _dynamic_block_quant_program(min_scale, row_block_size, col_block_size)
        q, scale = acl_emit(
            program,
            [x2d],
            [dst_name, "float32"],
            [(m_dim, k_dim), scale_shape],
        )
        if len(orig_shape) != 2:
            q = q.reshape(orig_shape)
            if row_block_size == 1:
                scale = scale.reshape(orig_shape[:-1] + (k_dim // col_block_size,))
        return q, scale


class DynamicMxQuantACL:
    def __call__(self, x, axis=-1, block_size=32, dst_dtype="float8_e4m3fn", scale_alg=0):
        from ._code import acl_emit

        axis = int(axis)
        block_size = int(block_size)
        scale_alg = int(scale_alg)
        dst_name = _jittor_dtype_name(dst_dtype)
        orig_shape = tuple(int(s) for s in x.shape)
        norm_axis = axis if axis >= 0 else axis + len(orig_shape)
        if not _is_contiguous(x):
            x = x.contiguous()
        # CANN DynamicMxQuant packs two UE8M0 block scales per pair along axis:
        # [..., K // (2 * block_size), 2] for block_size=32 -> [..., K // 64, 2].
        scale_shape = list(orig_shape)
        scale_shape[norm_axis] = (scale_shape[norm_axis] + 2 * block_size - 1) // (2 * block_size)
        scale_shape.append(2)
        program = _dynamic_mx_quant_program(axis, block_size, scale_alg)
        q, mx_scale = acl_emit(
            program,
            [x],
            [dst_name, "float8_e8m0fnu"],
            [orig_shape, tuple(scale_shape)],
        )
        return q, mx_scale


class QuantMatmulACL:
    def __call__(
        self,
        x1,
        x2,
        x1_scale,
        x2_scale,
        *,
        bias=None,
        out_dtype="bfloat16",
        trans_a=False,
        trans_b=False,
        trans_x2_scale=False,
        group_size=0,
    ):
        from ._code import acl_emit

        out_name = _jittor_dtype_name(out_dtype)
        orig_lead = tuple(int(s) for s in x1.shape[:-2])
        if x1.ndim > 2:
            x1_2d = x1.reshape((-1, int(x1.shape[-1])))
        else:
            x1_2d = x1
        if not _is_contiguous(x1_2d):
            x1_2d = x1_2d.contiguous()

        m_dim = int(x1_2d.shape[1] if trans_a else x1_2d.shape[0])
        n_dim = int(x2.shape[0] if trans_b else x2.shape[1])

        s1 = x1_scale
        s2 = x2_scale
        if int(group_size) == 0:
            # CANN aclnnQuantMatmulV5 per-tensor/per-token/per-channel requires 1-D scales.
            if s1.ndim != 1:
                s1 = s1.reshape((-1,))
            if s2.ndim != 1:
                s2 = s2.reshape((-1,))
        elif x1.ndim > 2:
            if s1.ndim > 2 and _jittor_dtype_name(s1.dtype) != "float8_e8m0fnu":
                s1 = s1.reshape((-1, int(s1.shape[-1])))
            elif s1.ndim > 3 and _jittor_dtype_name(s1.dtype) == "float8_e8m0fnu":
                s1 = s1.reshape((-1, int(s1.shape[-2]), int(s1.shape[-1])))

        if not _is_contiguous(s1):
            s1 = s1.contiguous()
        if not _is_contiguous(s2):
            s2 = s2.contiguous()

        # Pass bias=None to aclnnQuantMatmulV5 and add bias in out_dtype below:
        # CANN QuantMatmulV5 rejects BF16/FP16 bias for FP8 inputs (status 161002),
        # while on-device aclnnAdd handles every output dtype with zero fallback.
        program = _quant_matmul_program(
            4, trans_a, trans_b, trans_x2_scale, int(group_size), False
        )
        out = acl_emit(
            program,
            [x1_2d, x2, s1, s2],
            [out_name],
            [(m_dim, n_dim)],
        )[0]
        if bias is not None:
            bias_cast = bias if _jittor_dtype_name(bias.dtype) == out_name else bias.cast(out_name)
            out = out + bias_cast
        if orig_lead:
            out = out.reshape(orig_lead + (int(x1.shape[-2]), n_dim))
        return out


__all__ = ["DynamicBlockQuantACL", "DynamicMxQuantACL", "QuantMatmulACL"]
