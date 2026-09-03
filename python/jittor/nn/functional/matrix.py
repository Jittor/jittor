"""Matrix multiplication and bilinear neural-network operations."""

import jittor as jt


def _broadcast_batch_dims(a, b):
    """Broadcast the leading batch dims of two tensors with equal ndim>=3 to a
    common shape (torch matmul/bmm semantics), leaving the trailing two (matrix)
    dims untouched. cublasGemmStridedBatchedEx only supports a single batch
    stride per operand, so a batch dim of size 1 broadcast against >1 (e.g.
    Falcon multi-query attention: [b,nh,q,d] @ [b,1,d,k]) must be materialized
    here before dispatch."""
    if a.ndim != b.ndim or a.ndim < 3:
        return a, b
    bshape = []
    need = False
    for i in range(a.ndim - 2):
        an, bn = a.shape[i], b.shape[i]
        if an != bn:
            assert an == 1 or bn == 1, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
            need = True
        bshape.append(max(an, bn))
    if not need:
        return a, b
    if list(a.shape[:-2]) != bshape:
        a = a.expand(bshape + list(a.shape[-2:]))
    if list(b.shape[:-2]) != bshape:
        b = b.expand(bshape + list(b.shape[-2:]))
    return a, b


#: Every accelerated relay a matrix product can be sent to, and what it takes.
#:
#: | relay                     | device | dtypes                             |
#: | ------------------------- | ------ | ---------------------------------- |
#: | cublas_matmul (2-D)       | CUDA   | both operands the same float dtype |
#: | cublas_batched_matmul     | CUDA   | both operands the same float dtype |
#: | mkl_batched_matmul        | CPU    | both operands float32              |
#: | broadcast * mul + reduce  | any    | everything else, complex included  |
#:
#: Four call sites used to spell the cuBLAS row four different ways -- see
#: ``_cublas_can_take``.


def _same_floating_dtype(a, b):
    """Both operands carry the *same* floating-point dtype.

    This replaces ``"float" in str(dtype)``. The substring is true of bfloat16
    and float64 as well as float32 -- which happens to be right, cuBLAS takes
    all three -- but it is a test on how a dtype is *spelled*, so it also made
    the ``"complex" not in str(dtype)`` tests beside it look load-bearing when
    they can never fire: ``is_float()`` is already false for complex64.
    ``is_float()`` is a flag the dtype carries.

    Same dtype, not same width, and that distinction is the reachable one: the
    C++ relays only assert that the two widths agree, float16 and bfloat16 both
    being two bytes, while the kernel is instantiated from ``a``'s dtype alone.
    """
    return a.dtype == b.dtype and a.dtype.is_float()


def _cublas_can_take(a, b):
    """Whether the cuBLAS relays can compute this product. One predicate.

    The 2-D path tested both operands for complex, the batched path tested only
    ``a`` (harmless: it had already required the two dtypes equal), and
    ``bmm_transpose`` tested nothing at all -- so it handed integer and complex
    operands straight to the relay, which asserts on them, while exactly the
    same product written as ``matmul(a, b.transpose(...))`` computed fine on the
    generic path. Same mathematics, two spellings, one of them a crash.
    """
    return bool(
        jt.flags.use_cuda
        and jt.compile_extern.cublas_ops
        and _same_floating_dtype(a, b)
    )


def matmul_transpose(a, b):
    """
    returns a * b^T
    """
    assert a.shape[-1] == b.shape[-1], (a.shape, b.shape)
    if len(a.shape) != 2:
        aa = a.reshape((-1, a.shape[-1]))
        cc = jt.nn.matmul_transpose(aa, b)
        return cc.reshape(a.shape[:-1] + (-1,))
    assert len(a.shape) == 2 and len(b.shape) == 2
    if _cublas_can_take(a, b):
        return jt.compile_extern.cublas_ops.cublas_matmul(a, b, 0, 1)

    shape = list(a.shape)[:-1] + list(b.shape)
    with jt.flag_scope(amp_reg=jt.flags.amp_reg | jt.amp_flags.keep_reduce
                      | jt.amp_flags.reduce16_no_fp32_acc):
        a = a.broadcast(shape, [len(shape) - 2])
        b = b.broadcast(shape)
        return (a * b).sum(len(shape) - 1)


def bmm_transpose(a, b):
    """
    returns a * b^T
    """
    # The amp_reg scope is matmul's and matmul_transpose's too. It is what tells
    # the reduce in the generic path below to keep its input dtype rather than
    # accumulate in float32, so leaving it off here made the same product depend
    # on which of the two names the caller reached for.
    with jt.flag_scope(amp_reg=jt.flags.amp_reg | 36):
        if _cublas_can_take(a, b):
            a, b = jt.nn._broadcast_batch_dims(a, b)
            return jt.compile_extern.cublas_ops.cublas_batched_matmul(a, b, 0, 1)
        t = list(range(b.ndim))
        t[-1], t[-2] = t[-2], t[-1]
        return jt.nn.bmm(a, b.transpose(t))


def bmm(a, b):
    """batch matrix multiply,
    shape of input a is [batch, n, m],
    shape of input b is [batch, m, k],
    return shape is [batch, n, k]

    Example::

        import jittor as jt
        from jittor import nn

        batch, n, m, k = 100, 5, 6, 7

        a = jt.random((batch, n, m))
        b = jt.random((batch, m, k))
        c = nn.bmm(a, b)
    """
    assert len(a.shape) > 2 and len(b.shape) > 2
    return jt.nn.matmul(a, b)


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    res = jt.nn.bmm(batch1, batch2)
    if alpha != 1:
        res = res * alpha
    if beta == 0:
        return res
    return beta * input + res


def _matmul_2d_cublas(a, b, trans_a=0, trans_b=0):
    if _cublas_can_take(a, b):
        return jt.compile_extern.cublas_ops.cublas_matmul(a, b, trans_a, trans_b)
    return None


def _transpose_base_last2(x):
    try:
        base = getattr(x, "_jittor_transpose_base", None)
        if base is not None and getattr(x, "_jittor_transpose_last2", False):
            return base
    except Exception:
        pass
    return None


def _mkl_batched_matmul_is_available(a, b):
    """The CPU row of the table above: oneDNN's batched relay is float32-only.

    Anything else -- float64, float16, and the complex dtypes the native
    reindex kernels do support -- keeps the generic path.
    """
    if jt.flags.use_cuda:
        return False
    ops = getattr(jt.compile_extern, "mkl_ops", None)
    if ops is None or not hasattr(ops, "mkl_batched_matmul"):
        return False
    return a.dtype == b.dtype == jt.float32


def matmul(a, b):
    """matrix multiply,

    Example::

        a = jt.random([3])
        b = jt.random([3])
        c = jt.matmul(a, b)
        assert c.shape == [1]

        a = jt.random([3, 4])
        b = jt.random([4])
        c = jt.matmul(a, b)
        assert c.shape == [3]

        a = jt.random([10, 3, 4])
        b = jt.random([4])
        c = jt.matmul(a, b)
        assert c.shape == [10, 3]

        a = jt.random([10, 3, 4])
        b = jt.random([4, 5])
        c = jt.matmul(a, b)
        assert c.shape == [10, 3, 5]

        a = jt.random([10, 3, 4])
        b = jt.random([10, 4, 5])
        c = jt.matmul(a, b)
        assert c.shape == [10, 3, 5]

        a = jt.random([8, 1, 3, 4])
        b = jt.random([10, 4, 5])
        c = jt.matmul(a, b)
        assert c.shape == [8, 10, 3, 5]
    """
    with jt.flag_scope(amp_reg=jt.flags.amp_reg | jt.amp_flags.keep_reduce
                      | jt.amp_flags.reduce16_no_fp32_acc):
        len_a = len(a.shape)
        len_b = len(b.shape)
        if len_b == 1:
            # a: [n, m], b:[m], c:[n]
            return (a * b).sum(-1)
        if len_a == 1:
            # a: [n], b:[n,k], c:[k]
            return (a.broadcast(b, [-1]) * b).sum(0)
        if len_a == 2 and len_b == 2:
            # a: [n, m], b: [m, k], c: [n, k]
            a_base = jt.nn._transpose_base_last2(a)
            b_base = jt.nn._transpose_base_last2(b)
            aa = a_base if a_base is not None else a
            bb = b_base if b_base is not None else b
            fast = jt.nn._matmul_2d_cublas(
                aa,
                bb,
                1 if a_base is not None else 0,
                1 if b_base is not None else 0,
            )
            if fast is not None:
                return fast
        if len_a >= 3 and len_a == len_b:
            # bmm
            # a: [..., n, m], b: [..., m, k], c:[..., n, k]
            # cublas_batched_matmul only supports float dtypes; complex64 falls through to
            # the reindex path below (broadcast * multiply + sum-reduce), which the native
            # complex kernels support on both CPU and CUDA.
            if _cublas_can_take(a, b):
                a_base = jt.nn._transpose_base_last2(a)
                b_base = jt.nn._transpose_base_last2(b)
                if a_base is not None:
                    a = a_base
                if b_base is not None:
                    b = b_base
                a, b = jt.nn._broadcast_batch_dims(a, b)
                return jt.compile_extern.cublas_ops.cublas_batched_matmul(
                    a, b, 1 if a_base is not None else 0, 1 if b_base is not None else 0
                )
            # The reindex path below expresses a batched matmul as
            # broadcast * multiply + sum-reduce over one index space with an
            # extra dimension. The matmul tuner only relays the two-dimensional
            # form, so on CPU every batched matmul -- both attention products in
            # every transformer -- ran as that generic kernel at roughly a
            # fortieth of oneDNN's throughput.
            if _mkl_batched_matmul_is_available(a, b):
                a_base = jt.nn._transpose_base_last2(a)
                b_base = jt.nn._transpose_base_last2(b)
                if a_base is not None:
                    a = a_base
                if b_base is not None:
                    b = b_base
                a, b = jt.nn._broadcast_batch_dims(a, b)
                return jt.compile_extern.mkl_ops.mkl_batched_matmul(
                    a, b, 1 if a_base is not None else 0, 1 if b_base is not None else 0
                )
        shape = []
        len_c = max(len_a, len_b)
        (n, m), (m_, k) = a.shape[-2:], b.shape[-2:]
        assert m == m_, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
        # a: [..., n, m]
        # b: [..., m, k]
        # cc:[..., n, m, k]
        #     -->
        #     012
        if len_b == 2 and len_a > 2:
            # TODO:ugly implementation for tuner
            aa = a.reshape((-1, m))
            cc = jt.nn.matmul(aa, b)
            # print(a.shape, b.shape, cc.shape)
            return cc.reshape(a.shape[:-1] + [k])
        for i in range(len_c - 2):
            ai = len_a - (len_c - i)
            bi = len_b - (len_c - i)
            an = a.shape[ai] if ai >= 0 else 1
            bn = b.shape[bi] if bi >= 0 else 1
            if an != 1 and bn != 1:
                assert an == bn, f"dimension not match, a.shape:{a.shape}, b.shape:{b.shape}"
            cn = max(an, bn)
            shape.append(cn)
        shape.extend([n, m, k])
        a = a.broadcast(shape, [-1])
        b = b.broadcast(shape, [-3])
        return (a * b).sum(-2)


def bilinear(in1, in2, weight, bias):
    if weight.shape[1] != in1.shape[1]:
        raise RuntimeError(
            f"bilinear(): input1 size deos not match weight size: got {in1.shape[1]} but expected {weight.shape[1]}"
        )
    if weight.shape[2] != in2.shape[1]:
        raise RuntimeError(
            f"bilinear(): input2 size deos not match weight size: got {in2.shape[1]} but expected {weight.shape[2]}"
        )
    w = weight.transpose((1, 0, 2))
    w = w.reshape((w.shape[0], -1))
    x = jt.nn.matmul(in1, w)
    x = x.reshape(x.shape[:-1] + [weight.shape[0], weight.shape[2]])
    y = in2.broadcast(x, (-2,))
    z = (x * y).sum(-1)
    if bias is not None:
        z += bias
    return z


__all__ = [
    "baddbmm",
    "bilinear",
    "bmm",
    "bmm_transpose",
    "matmul",
    "matmul_transpose",
]
