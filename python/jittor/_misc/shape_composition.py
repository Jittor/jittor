"""Minimum-rank views and compound shape construction helpers."""

from .runtime import jt, preserve_facade_origins


def atleast_1d(*tensors):
    r'''
    Returns a 1-dimensional view (or the original) of each input that has zero
    dimensions. Inputs with one or more dimensions are returned unchanged.

    Mirrors ``torch.atleast_1d``: a single input returns a single Var, multiple
    inputs return a tuple of Vars.

    Args:

        tensors – one or more Vars (or values castable to Var).
    '''
    res = []
    for t in tensors:
        if not isinstance(t, jt.Var):
            t = jt.array(t)
        if t.ndim == 0:
            t = t.reshape(1)
        res.append(t)
    if len(res) == 1:
        return res[0]
    return tuple(res)


def atleast_2d(*tensors):
    r'''
    Returns a view (or the original) of each input with at least 2 dimensions.

    Scalars become shape ``(1, 1)`` and 1-D inputs of shape ``(N,)`` become
    ``(1, N)``. Mirrors ``torch.atleast_2d``: a single input returns a single
    Var, multiple inputs return a tuple of Vars.

    Args:

        tensors – one or more Vars (or values castable to Var).
    '''
    res = []
    for t in tensors:
        if not isinstance(t, jt.Var):
            t = jt.array(t)
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.reshape(1, t.shape[0])
        res.append(t)
    if len(res) == 1:
        return res[0]
    return tuple(res)


def atleast_3d(*tensors):
    r'''
    Returns a view (or the original) of each input with at least 3 dimensions.

    Following ``torch.atleast_3d``: scalars become shape ``(1, 1, 1)``, 1-D
    inputs of shape ``(N,)`` become ``(1, N, 1)``, and 2-D inputs of shape
    ``(M, N)`` become ``(M, N, 1)``. A single input returns a single Var,
    multiple inputs return a tuple of Vars.

    Args:

        tensors – one or more Vars (or values castable to Var).
    '''
    res = []
    for t in tensors:
        if not isinstance(t, jt.Var):
            t = jt.array(t)
        if t.ndim == 0:
            t = t.reshape(1, 1, 1)
        elif t.ndim == 1:
            t = t.reshape(1, t.shape[0], 1)
        elif t.ndim == 2:
            t = t.reshape(t.shape[0], t.shape[1], 1)
        res.append(t)
    if len(res) == 1:
        return res[0]
    return tuple(res)


def cartesian_prod(*tensors):
    r'''
    Do cartesian product of the given sequence of 1-D Vars. Equivalent to
    ``itertools.product`` on the inputs. Mirrors ``torch.cartesian_prod``.

    Args:

        tensors – one or more 1-D Vars.

    Returns:

        A Var of shape ``(prod(len_i), N)`` where ``N`` is the number of inputs.
        With a single 1-D input, a 1-D Var is returned (matching torch).
    '''
    norm = []
    for t in tensors:
        if not isinstance(t, jt.Var):
            t = jt.array(t)
        assert t.ndim == 1, "cartesian_prod only accepts 1-D Vars"
        norm.append(t)
    if len(norm) == 1:
        return norm[0]
    grids = jt.misc.meshgrid(norm)
    cols = [g.reshape(-1, 1) for g in grids]
    return jt.concat(cols, dim=1)


def block_diag(*tensors):
    r'''
    Create a block diagonal matrix from the provided Vars. Mirrors
    ``torch.block_diag``: each input may be 0-D, 1-D or 2-D; 0-D/1-D inputs are
    treated as a single row, and the result is a 2-D matrix whose diagonal
    blocks are the inputs (off-diagonal entries are zero).

    Args:

        tensors – one or more Vars with at most 2 dimensions.
    '''
    norm = []
    for t in tensors:
        if not isinstance(t, jt.Var):
            t = jt.array(t)
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.reshape(1, t.shape[0])
        elif t.ndim > 2:
            raise ValueError(
                "block_diag: input tensors must have at most 2 dimensions, "
                f"got {t.ndim}")
        norm.append(t)

    total_rows = sum(t.shape[0] for t in norm)
    total_cols = sum(t.shape[1] for t in norm)
    if total_rows == 0 or total_cols == 0:
        dtype = norm[0].dtype if norm else "float32"
        return jt.zeros((total_rows, total_cols), dtype)

    rows = []
    col_offset = 0
    for t in norm:
        r, c = t.shape
        left = col_offset
        right = total_cols - col_offset - c
        pieces = []
        if left > 0:
            pieces.append(jt.zeros((r, left), t.dtype))
        pieces.append(t)
        if right > 0:
            pieces.append(jt.zeros((r, right), t.dtype))
        row = pieces[0] if len(pieces) == 1 else jt.concat(pieces, dim=1)
        rows.append(row)
        col_offset += c

    return rows[0] if len(rows) == 1 else jt.concat(rows, dim=0)


_FACADE_SYMBOLS = (
    atleast_1d, atleast_2d, atleast_3d, cartesian_prod, block_diag,
)
preserve_facade_origins(_FACADE_SYMBOLS)
