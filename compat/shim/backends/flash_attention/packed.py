"""FlashAttention packed implementation."""
from __future__ import annotations

def _packed_split_enabled() -> bool:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    return _facade._truthy(_facade.os.environ.get("JITTOR_FLASH_ATTN_FUSED_PACKED_SPLIT"))


def _is_cuda_jittor_var(x) -> bool:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if not _facade._packed_split_enabled():
        return False
    try:
        import jittor as jt
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _is_cuda_jittor_var: import jittor as jt", exc)
        return False
    try:
        return bool(jt.flags.use_cuda) and isinstance(x, jt.Var)
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _is_cuda_jittor_var: return bool(jt.flags.use_cuda) and isinstance(x, jt.Var)", exc)
        return False


def _split_qkvpacked_cuda(qkv):
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if not _facade._is_cuda_jittor_var(qkv):
        return None
    try:
        import jittor as jt

        shape = list(qkv.shape)
        dtype = qkv.dtype
        if len(shape) == 4 and int(shape[1]) == 3:
            total, _, heads, dim = shape
            n = int(total) * int(heads) * int(dim)
            if n == 0:
                return qkv[:, 0], qkv[:, 1], qkv[:, 2]
            out_shape = [total, heads, dim]
            q, k, v = jt.code(
                [out_shape, out_shape, out_shape],
                [dtype, dtype, dtype],
                [qkv],
                cuda_src="""
__global__ static void split_qkv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape2;
    int h = (i / out0_shape2) % out0_shape1;
    int t = i / ((int64_t)out0_shape2 * out0_shape1);
    @out0(t, h, d) = @in0(t, 0, h, d);
    @out1(t, h, d) = @in0(t, 1, h, d);
    @out2(t, h, d) = @in0(t, 2, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
split_qkv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _facade._PACKED_SPLIT_STATS["qkv_cuda"] += 1
            return q, k, v
        if len(shape) == 5 and int(shape[2]) == 3:
            batch, seqlen, _, heads, dim = shape
            n = int(batch) * int(seqlen) * int(heads) * int(dim)
            if n == 0:
                return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            out_shape = [batch, seqlen, heads, dim]
            q, k, v = jt.code(
                [out_shape, out_shape, out_shape],
                [dtype, dtype, dtype],
                [qkv],
                cuda_src="""
__global__ static void split_qkv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape3;
    int h = (i / out0_shape3) % out0_shape2;
    int s = (i / ((int64_t)out0_shape3 * out0_shape2)) % out0_shape1;
    int b = i / ((int64_t)out0_shape3 * out0_shape2 * out0_shape1);
    @out0(b, s, h, d) = @in0(b, s, 0, h, d);
    @out1(b, s, h, d) = @in0(b, s, 1, h, d);
    @out2(b, s, h, d) = @in0(b, s, 2, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
split_qkv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _facade._PACKED_SPLIT_STATS["qkv_cuda"] += 1
            return q, k, v
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _split_qkvpacked_cuda: import jittor as jt", exc)
        _facade._PACKED_SPLIT_STATS["error"] += 1
        _facade._remember_error("fused qkvpacked split failed: %s" % exc)
        return None
    _facade._PACKED_SPLIT_STATS["fallback"] += 1
    return None


def _split_kvpacked_cuda(kv):
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    if not _facade._is_cuda_jittor_var(kv):
        return None
    try:
        import jittor as jt

        shape = list(kv.shape)
        dtype = kv.dtype
        if len(shape) == 4 and int(shape[1]) == 2:
            total, _, heads, dim = shape
            n = int(total) * int(heads) * int(dim)
            if n == 0:
                return kv[:, 0], kv[:, 1]
            out_shape = [total, heads, dim]
            k, v = jt.code(
                [out_shape, out_shape],
                [dtype, dtype],
                [kv],
                cuda_src="""
__global__ static void split_kv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape2;
    int h = (i / out0_shape2) % out0_shape1;
    int t = i / ((int64_t)out0_shape2 * out0_shape1);
    @out0(t, h, d) = @in0(t, 0, h, d);
    @out1(t, h, d) = @in0(t, 1, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2;
split_kv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _facade._PACKED_SPLIT_STATS["kv_cuda"] += 1
            return k, v
        if len(shape) == 5 and int(shape[2]) == 2:
            batch, seqlen, _, heads, dim = shape
            n = int(batch) * int(seqlen) * int(heads) * int(dim)
            if n == 0:
                return kv[:, :, 0], kv[:, :, 1]
            out_shape = [batch, seqlen, heads, dim]
            k, v = jt.code(
                [out_shape, out_shape],
                [dtype, dtype],
                [kv],
                cuda_src="""
__global__ static void split_kv(@ARGS_DEF) {
    @PRECALC
    int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = i % out0_shape3;
    int h = (i / out0_shape3) % out0_shape2;
    int s = (i / ((int64_t)out0_shape3 * out0_shape2)) % out0_shape1;
    int b = i / ((int64_t)out0_shape3 * out0_shape2 * out0_shape1);
    @out0(b, s, h, d) = @in0(b, s, 0, h, d);
    @out1(b, s, h, d) = @in0(b, s, 1, h, d);
}
int64_t n = (int64_t)out0_shape0 * out0_shape1 * out0_shape2 * out0_shape3;
split_kv<<<(n + 255) / 256, 256>>>(@ARGS);
""",
            )
            _facade._PACKED_SPLIT_STATS["kv_cuda"] += 1
            return k, v
    except _facade.EXPECTED as exc:
        _facade.swallowed("shim/backends/flash_attention.py _split_kvpacked_cuda: import jittor as jt", exc)
        _facade._PACKED_SPLIT_STATS["error"] += 1
        _facade._remember_error("fused kvpacked split failed: %s" % exc)
        return None
    _facade._PACKED_SPLIT_STATS["fallback"] += 1
    return None


def _direct_packed_enabled() -> bool:
    import importlib as _importlib
    _facade = _importlib.import_module(__package__)
    value = _facade.os.environ.get("JITTOR_FLASH_ATTN_DIRECT_ADAPTER")
    if value is None:
        value = _facade.os.environ.get("JITTOR_FLASH_ATTN_DIRECT_PACKED")
    return not _facade._falsey(value)
