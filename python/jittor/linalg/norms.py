# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Matrix and vector norms, ranks and condition numbers."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name


def matrix_rank(x, tol=None, hermitian=False):
    r"""
    Return the numerical rank of a (batch of) matrix.

    Equivalent to ``torch.linalg.matrix_rank`` / ``numpy.linalg.matrix_rank``:
    the rank is the number of singular values greater than ``tol``. When ``tol``
    is ``None`` the default threshold ``S.max(-1) * max(M, N) * eps`` is used
    (``eps`` is the machine epsilon of the input dtype), matching numpy/torch.

    The singular values are obtained from the existing differentiable
    :func:`svd`; the rank itself is an integer count and therefore has no
    gradient (matching numpy/torch).

    :param x (...,M,N): batch of matrices.
    :param tol (float, optional): explicit singular-value threshold.
    :param hermitian (bool): if ``True``, use the symmetric eigensolver
        (eigenvalue magnitudes) instead of the SVD; ``x`` must be square.
    :return: integer rank tensor with shape ``x.shape[:-2]``.
    """
    import jittor as jt
    from .decompositions import eigh
    from .decompositions import svd
    if hermitian:
        if x.shape[-2] != x.shape[-1]:
            raise ValueError("matrix_rank(hermitian=True) expects square matrices")
        w, _ = eigh(x)
        s = jt.abs(w)
    else:
        # reduced svd keeps S differentiable and is shape-agnostic; the rank
        # itself is non-differentiable (an integer count), but other callers of
        # the returned S (matrix_norm/cond) rely on the gradient.
        _, s, _ = svd(x, full_matrices=False)

    smax = s.max(dim=-1)
    # torch_compat gives max(dim=...) the torch return type (values, indices),
    # while this native linalg helper needs only the values tensor.
    if hasattr(smax, "values"):
        smax = smax.values
    if tol is None:
        import numpy as _np
        try:
            eps = float(_np.finfo(_np.dtype(_jittor_dtype_name(x.dtype))).eps)
        except Exception:
            eps = float(_np.finfo(_np.float32).eps)
        m, n = x.shape[-2], x.shape[-1]
        tol_v = smax * (max(m, n) * eps)
    else:
        tol_v = jt.array(tol).broadcast(smax.shape) if not isinstance(tol, (int, float)) \
            else smax * 0 + float(tol)
    # count singular values strictly greater than the threshold
    keep = (s > tol_v.unsqueeze(-1)).int32()
    return keep.sum(dim=-1)


def matrix_norm(x, ord='fro', dim=(-2, -1), keepdim=False):
    r"""
    Compute a matrix norm of a (batch of) matrix.

    Matches ``torch.linalg.matrix_norm`` / ``numpy.linalg.norm`` for the
    matrix-valued orders:

    - ``'fro'`` (default): Frobenius norm ``sqrt(sum(|x|**2))`` (native jittor).
    - ``'nuc'``: nuclear norm, the sum of singular values (via :func:`svd`).
    - ``2`` / ``-2``: largest / smallest singular value (via :func:`svd`).
    - ``1`` / ``-1``: max / min absolute column sum (native jittor).
    - ``inf`` / ``-inf``: max / min absolute row sum (native jittor).

    The Frobenius and the ``1/inf`` family are built from native jittor ops and
    stay differentiable on-device; the spectral (``2``/``-2``/``'nuc'``) orders
    use the existing differentiable :func:`svd`.

    :param x (...,M,N): batch of matrices.
    :param ord: matrix order, see above. Default ``'fro'``.
    :param dim (tuple[int, int]): the two dims that form each matrix.
    :param keepdim (bool): keep the reduced dims as size-1.
    :return: matrix norm with the two ``dim`` axes reduced.
    """
    import jittor as jt
    import math
    d0, d1 = dim
    nd = len(x.shape)
    d0 = d0 % nd
    d1 = d1 % nd
    if d0 == d1:
        raise ValueError("matrix_norm: dim must reference two distinct axes")

    def _restore(res, reduced_axes):
        # re-insert size-1 axes so keepdim=True lines up with the input rank
        if not keepdim:
            return res
        out = res
        for ax in sorted(reduced_axes):
            out = out.unsqueeze(ax)
        return out

    if ord == 'fro':
        res = jt.sqrt((x * x).sum(dims=[d0, d1]))
        return _restore(res, [d0, d1])
    if ord == 'nuc':
        # move matrix dims to the end, take sum of singular values
        s = _matrix_singular_values(x, d0, d1)
        res = s.sum(dim=-1)
        return _restore(res, [d0, d1])
    if ord in (2, -2):
        s = _matrix_singular_values(x, d0, d1)
        res = s.max(dim=-1) if ord == 2 else s.min(dim=-1)
        return _restore(res, [d0, d1])
    if ord in (1, -1):
        # absolute column sums (reduce the row dim d0), then max/min over cols
        col_sums = jt.abs(x).sum(dim=d0)
        # after reducing d0, the column axis d1 shifts left if d0 < d1
        col_axis = d1 - 1 if d0 < d1 else d1
        res = col_sums.max(dim=col_axis) if ord == 1 else col_sums.min(dim=col_axis)
        return _restore(res, [d0, d1])
    if ord in (math.inf, -math.inf, float('inf'), float('-inf')):
        # absolute row sums (reduce the column dim d1), then max/min over rows
        row_sums = jt.abs(x).sum(dim=d1)
        row_axis = d0 - 1 if d1 < d0 else d0
        if ord > 0:
            res = row_sums.max(dim=row_axis)
        else:
            res = row_sums.min(dim=row_axis)
        return _restore(res, [d0, d1])
    raise ValueError(f"matrix_norm: unsupported matrix order {ord!r}")


def _matrix_singular_values(x, d0, d1):
    # Bring the two matrix axes to the last two positions, run the existing
    # differentiable svd, and return the singular values (shape batch + [k]).
    from .decompositions import svd
    nd = len(x.shape)
    others = [a for a in range(nd) if a not in (d0, d1)]
    perm = others + [d0, d1]
    xt = x.permute(perm) if perm != list(range(nd)) else x
    # reduced form -> differentiable singular values (nuc/2/-2 matrix norms)
    _, s, _ = svd(xt, full_matrices=False)
    return s


def vector_norm(x, ord=2, dim=None, keepdim=False):
    r"""
    Compute a vector norm, matching ``torch.linalg.vector_norm``.

    The whole tensor (``dim=None``) or the given ``dim`` axes are treated as a
    flat vector. Supported orders:

    - ``2`` (default): Euclidean norm ``sqrt(sum(|x|**2))``.
    - ``1``: sum of absolute values.
    - ``0``: number of non-zero entries.
    - any finite ``p``: ``(sum(|x|**p))**(1/p)``.
    - ``inf`` / ``-inf``: max / min absolute value.

    Built entirely from native jittor ops, so it is differentiable on-device.

    :param x: input tensor.
    :param ord: vector order, see above. Default ``2``.
    :param dim (int | tuple[int] | None): axis/axes to reduce; ``None`` flattens.
    :param keepdim (bool): keep reduced dims as size-1 (ignored when ``dim`` is
        ``None`` and the whole tensor is flattened).
    :return: the requested vector norm.
    """
    import math
    if dim is None:
        flat = x.reshape([-1])
        ax = 0
        out = _vector_norm_reduce(flat, ord, ax)
        return out
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    return _vector_norm_reduce(x, ord, dims, keepdim=keepdim)


def _vector_norm_reduce(x, ord, dims, keepdim=False):
    import jittor as jt
    import math
    ax = dims
    absx = jt.abs(x)
    if ord == 2:
        res = jt.sqrt((absx * absx).sum(dims=ax) if isinstance(ax, list)
                      else (absx * absx).sum(dim=ax))
    elif ord == 1:
        res = absx.sum(dims=ax) if isinstance(ax, list) else absx.sum(dim=ax)
    elif ord == 0:
        nz = (x != 0).float32()
        res = nz.sum(dims=ax) if isinstance(ax, list) else nz.sum(dim=ax)
    elif ord in (math.inf, float('inf')):
        res = _reduce_axes(absx, ax, reduce='max')
    elif ord in (-math.inf, float('-inf')):
        res = _reduce_axes(absx, ax, reduce='min')
    else:
        p = float(ord)
        powed = absx ** p
        s = powed.sum(dims=ax) if isinstance(ax, list) else powed.sum(dim=ax)
        res = s ** (1.0 / p)
    if keepdim and isinstance(ax, list):
        for a in sorted(a % len(x.shape) for a in ax):
            res = res.unsqueeze(a)
    return res


def _reduce_axes(x, ax, reduce='max'):
    # max/min over possibly several axes (jittor reduces one axis at a time
    # for the index-returning variants, but the value reductions accept lists).
    if isinstance(ax, int):
        ax = [ax]
    # reduce from the highest axis down so earlier indices stay valid
    out = x
    for a in sorted((a % len(x.shape) for a in ax), reverse=True):
        out = out.max(dim=a) if reduce == 'max' else out.min(dim=a)
    return out


def norm(x, ord=None, dim=None, keepdim=False):
    r"""
    General ``torch.linalg.norm`` / ``numpy.linalg.norm`` dispatcher.

    Behaviour (matching torch/numpy):

    - ``dim`` is a 2-tuple, or ``dim is None`` and ``x`` is 2-D: matrix norm
      (default ``ord`` is the Frobenius norm), see :func:`matrix_norm`.
    - ``dim`` is an int, or ``dim is None`` and ``x`` is 1-D, or ``ord`` is set
      with ``dim is None``: vector norm (default ``ord`` is 2), see
      :func:`vector_norm`.
    - ``dim is None`` and ``ord is None``: the 2-norm of the flattened tensor
      (Frobenius), built from native jittor ops.

    :param x: input tensor.
    :param ord: order of the norm; meaning depends on whether a matrix or a
        vector norm is selected (see :func:`matrix_norm` / :func:`vector_norm`).
    :param dim (int | tuple | None): axis/axes defining the norm.
    :param keepdim (bool): keep reduced dims as size-1.
    :return: the requested norm.
    """
    if dim is None:
        if ord is None:
            # flatten and take the 2-norm (== Frobenius for matrices)
            return vector_norm(x, ord=2, dim=None)
        if len(x.shape) == 2:
            # ord given, 2-D input, dim=None -> matrix norm (torch/numpy rule)
            return matrix_norm(x, ord=ord, dim=(-2, -1), keepdim=keepdim)
        # 1-D, or higher-rank with ord set but no dim -> vector norm over the
        # flattened input (matches numpy.linalg.norm).
        return vector_norm(x, ord=ord, dim=None)
    if isinstance(dim, (tuple, list)) and len(dim) == 2:
        m_ord = 'fro' if ord is None else ord
        return matrix_norm(x, ord=m_ord, dim=tuple(dim), keepdim=keepdim)
    v_ord = 2 if ord is None else ord
    return vector_norm(x, ord=v_ord, dim=dim, keepdim=keepdim)


def cond(x, p=None):
    r"""
    Condition number of a (batch of) matrix, matching
    ``torch.linalg.cond`` / ``numpy.linalg.cond``.

    - ``p`` in ``{None, 2}``: ratio of largest to smallest singular value
      (via the existing differentiable :func:`svd`).
    - ``p == -2``: smallest over largest singular value.
    - ``p`` in ``{1, -1, inf, -inf}``: ``norm(x, p) * norm(inv(x), p)`` using the
      matrix-norm definitions above; ``x`` must be invertible.
    - ``p == 'fro'``: ``norm(x, 'fro') * norm(inv(x), 'fro')``.

    :param x (...,M,M): batch of square matrices (square required for all ``p``
        except the ``None/2/-2`` singular-value forms, which also accept the
        general case numpy/torch allow).
    :param p: order of the norm used for the condition number.
    :return: condition number with shape ``x.shape[:-2]``.
    """
    from .decompositions import svd
    from .solving import inv
    import math
    if p is None or p == 2 or p == -2:
        # reduced form -> differentiable singular values
        _, s, _ = svd(x, full_matrices=False)
        smax = s.max(dim=-1)
        smin = s.min(dim=-1)
        if p == -2:
            return smin / smax
        return smax / smin
    # p in {1, -1, inf, -inf, 'fro'} -> norm(x, p) * norm(inv(x), p)
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("cond(p={!r}) expects square matrices".format(p))
    xi = inv(x)
    return matrix_norm(x, ord=p, dim=(-2, -1)) * matrix_norm(xi, ord=p, dim=(-2, -1))
