"""Scan tensor operations."""

import numpy as np
import collections as _collections
from .._core.function import Function
from .._runtime.dispatch import select_kernel, try_dispatch

def _prod(x,dim=0):
    import jittor as jt
    x = jt.log(x)
    x = x.sum(dim=dim)
    return jt.exp(x)


def numpy_cumsum(x, dim=None):
    ''' cumsum implemented with numpy or cupy.

        This function should not be called directly. Instead, jittor.misc.cumsum is recommended.
    '''
    import jittor as jt
    def cumsum_forward(np, data):
        a = data['inputs'][0]
        b = data['outputs'][0]
        np.cumsum(a, axis=dim, out=b)

    def cumsum_backward(np, data):
        dout = data['dout']
        out = data['outputs'][0]
        np.cumsum(np.flip(dout, dim), axis=dim, out=out)
        np.copyto(out, np.flip(out, dim))
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    return jt.numpy_code(x.shape, x.dtype, [x], cumsum_forward, [cumsum_backward])


def cub_cumsum(x, dim=None):
    ''' cumsum implemented with CUB.

        This function should not be called directly. Instead, jittor.misc.cumsum is recommended.
    '''
    from jittor.backends.cuda.kernels.misc.tensor_ops import (
        _repeat_interleave_dim0_cuda, _stack_no_grad_cuda_fast,
        _unbind_no_grad_cuda_fast, _unique_code_cuda, _scan_2d_cuda,
    )
    result = try_dispatch("tensor.cumsum", x, -1 if dim is None else dim)
    if result is not None:
        return result
    if (dim == None):
        dim = -1
    assert(dim >= -1 and dim < len(x.shape))
    shape = list(x.shape)
    if (dim != -1 and dim != len(shape) - 1):
        order = list(range(len(shape)))
        order[dim], order[-1] = order[-1], order[dim]
        shape[dim], shape[-1] = shape[-1], shape[dim]
        x = x.permute(order)
    if (len(shape) > 2):
        x = x.reshape([-1, shape[-1]])
    x = _scan_2d_cuda(x, False)
    if (len(shape) > 2):
        x = x.reshape(shape)
    if (dim != -1 and dim != len(shape) - 1):
        x = x.permute(order)
    return x


def _cumsum_dim(dim, ndim):
    '''torch's dim contract: ``-ndim <= dim < ndim``, negatives from the end.

    The old guard was ``assert(dim >= -1 and dim < len(x.shape))``, which
    accepted exactly *one* negative value: ``cumsum(x, -2)`` on a 3-D tensor
    raised, while ``cumsum(x, 1)`` -- the same axis -- did not.
    '''
    if dim is None:
        dim = -1
    ndim = max(ndim, 1)
    if not -ndim <= dim < ndim:
        raise IndexError(
            "cumsum: dim %d is out of range for a %d-dimensional input"
            % (dim, ndim))
    return dim % ndim


def _scan_2d(x, reverse):
    '''Inclusive prefix sum along axis 1 of a 2-D var. The kernel, only.

    Which backend runs it is the one thing that varies here; the shape
    handling, the dim contract and the derivative all sit above it in
    :func:`cumsum` and :class:`_Cumsum`. The CPU side used to be
    ``jt.numpy_code``: a host callback that pulled the input out of the lazy
    graph, ran a Python function per execution, and carried its own separate
    backward.
    '''
    kernel = select_kernel("misc.scan_2d", x, reverse)
    if kernel is None:
        raise NotImplementedError("cumsum has no scan kernel for this device")
    return kernel(x, reverse)


def _scan_2d_cpu(x, reverse):
    import jittor as jt
    index = "n - 1 - k" if reverse else "k"
    return jt.code(x.shape, x.dtype, [x], cpu_src=f'''
        @alias(x, in0)
        @alias(y, out0)
        int64 rows = y_shape0, n = y_shape1;
        for (int64 r = 0; r < rows; ++r) {{
            y_type acc = 0;
            for (int64 k = 0; k < n; ++k) {{
                int64 i = {index};
                acc += @x(r, i);
                @y(r, i) = acc;
            }}
        }}
    ''')


class _Cumsum(Function):
    '''cumsum's derivative, written once.

    ``d/dx_j sum_{i<=k} x_i`` is 1 exactly when ``j <= k``, so the gradient of
    an inclusive forward scan is an inclusive *reverse* scan of the seed. Each
    backend used to carry its own copy of that rule -- ``CubCumsumOp::grad`` in
    C++, a numpy flip/cumsum/flip in Python -- and neither knew about the
    other.
    '''

    def execute(self, x):
        import jittor as jt
        return jt.misc._scan_2d(x, False)

    def grad(self, g):
        import jittor as jt
        return jt.misc._scan_2d(g, True)


def cumsum(x, dim=None):
    '''
    Parameters:
    -----------
    x: jt.var
    dim: int

    Returns:
    --------
    the cumulative sum in dim of x

    One implementation for every backend. It used to be two, chosen by
    ``jt.flags.use_cuda``: CUB on CUDA and a numpy host callback on CPU, with a
    gradient rule each and a ``dim`` guard each.
    '''
    import jittor as jt
    dim = jt.misc._cumsum_dim(dim, x.ndim)
    result = try_dispatch("tensor.cumsum", x, dim)
    if result is not None:
        return result
    shape = list(x.shape)
    last = max(len(shape) - 1, 0)
    order = None
    if dim != last:
        order = list(range(len(shape)))
        order[dim], order[last] = order[last], order[dim]
        x = x.permute(order)
    moved_shape = x.shape
    if x.numel() == 0:
        # Nothing to scan, and the empty case has to be caught here rather than
        # in the kernel: flattening to 2-D cannot infer -1 against a zero-length
        # axis, and a zero-row result asks the CUDA block scan for a grid of
        # zero blocks -- "invalid configuration argument", raised
        # asynchronously, so it surfaces in whatever runs next.
        y = x.clone()
    else:
        y = jt.misc._Cumsum.apply(
            x.reshape([-1, x.shape[-1]])).reshape(moved_shape)
    if order is not None:
        y = y.permute(order)
    return y


def cumprod(x,dim=None):
    # Sign-aware cumulative product. The old exp(cumsum(log(x))) returns NaN for any
    # NEGATIVE element (log of a negative) -- torch handles signs. Split into magnitude
    # and a running sign parity: cumprod = (-1)^(#negatives so far) * exp(cumsum(log|x|)).
    # Zeros are masked out (mag clamped to 1 for the log so we never hit log(0)=-inf,
    # which can trip jittor's inf/nan JIT codegen; positions at/after the first zero are
    # forced to 0). Reduces to the old behaviour for all-positive input.
    import jittor as jt
    mag = jt.abs(x)
    is_zero = (mag == 0)
    mag_safe = jt.ternary(is_zero, jt.ones_like(mag), mag)
    mag_cp = jt.exp(jt.misc.cumsum(jt.log(mag_safe), dim=dim))
    zero_seen = jt.misc.cumsum(is_zero.int32(), dim=dim) > 0
    mag_cp = jt.ternary(zero_seen, jt.zeros_like(mag_cp), mag_cp)
    sign = (
        1 - 2 * (jt.misc.cumsum((x < 0).int32(), dim=dim) % 2)
    ).float32()
    return sign * mag_cp


_CumMax = _collections.namedtuple("cummax", ["values", "indices"])

_CumMin = _collections.namedtuple("cummin", ["values", "indices"])

def _cummax_min(x, dim, is_max):
    ''' prefix max/min + argmax/argmin along dim. O(L^2) masked reduction (fine for
    typical L): M[...,i,j] = x[...,j] if j<=i else sentinel, reduce over j. Uses a
    FINITE sentinel (not +/-inf) to dodge jittor's inf/nan JIT codegen segfault, and
    jt.argmax (which picks the FIRST max -> matches torch's cummax tie behavior). '''
    import jittor as jt
    if dim is None:
        dim = x.ndim - 1
    d = dim if dim >= 0 else dim + x.ndim
    perm = [k for k in range(x.ndim) if k != d] + [d]      # move dim d to last
    xt = x.permute(perm)
    L = xt.shape[-1]
    tgt = list(xt.shape[:-1]) + [L, L]
    xe = xt.unsqueeze(-2).broadcast(tgt)                   # [...,i,j] = xt[...,j]
    ii = jt.index((L, L), dim=0)
    jj = jt.index((L, L), dim=1)
    mask = (jj <= ii).broadcast(tgt)                       # valid where j<=i
    sentinel = -3.4e38 if is_max else 3.4e38
    sval = jt.array(sentinel).cast(xt.dtype).broadcast(tgt)
    masked = jt.ternary(mask, xe, sval)
    # NB: native jt.argmax returns (indices, values); the torch_compat layer overrides
    # it to indices-only, and overrides Var.max/min to return a (values, indices)
    # namedtuple. Handle both so cummax works regardless of install state.
    def _argidx(am):
        return am[0] if isinstance(am, (tuple, list)) else am
    def _vals(mm):
        return mm.values if hasattr(mm, "values") else (
            mm[0] if isinstance(mm, (tuple, list)) else mm)
    if is_max:
        vals = _vals(masked.max(dim=-1)); idxs = _argidx(jt.argmax(masked, -1))
    else:
        vals = _vals(masked.min(dim=-1)); idxs = _argidx(jt.argmin(masked, -1))
    inv = [0] * x.ndim
    for newpos, oldpos in enumerate(perm):
        inv[oldpos] = newpos
    return vals.permute(inv), idxs.int64().permute(inv)


def cummax(x, dim=None):
    ''' torch's cummax(input, dim) -> namedtuple(values, indices). '''
    v, i = _cummax_min(x, dim, True)
    return _CumMax(v, i)


def cummin(x, dim=None):
    ''' torch's cummin(input, dim) -> namedtuple(values, indices). '''
    v, i = _cummax_min(x, dim, False)
    return _CumMin(v, i)


def numpy_cumprod(a, dim):
    import jittor as jt
    class CumprodFunc(Function):
        def forward_code(self, np, data):
            a = data["inputs"][0]
            b = data["outputs"][0]
            out = np.cumprod(a, self.dim)
            np.copyto(b, out)

        def backward_code(self, np, data):
            a, b, dout = data["inputs"]
            out = data["outputs"][0]

            sdim = a.shape[self.dim]
            dim = (len(a.shape)+1)*[1]
            dim[self.dim+1] = sdim
            res = np.tile(np.expand_dims(b, self.dim+1), dim)
            dout = np.tile(np.expand_dims(dout, self.dim+1), dim)

            dim[self.dim]=sdim
            dim[self.dim+1]=1
            a = np.tile(np.expand_dims(a, self.dim), dim)
            res = res/a

            mask = np.tril(np.ones((sdim, sdim)))
            for i in range(self.dim):
                mask = np.expand_dims(mask, 0)
            for i in range(len(a.shape)-self.dim-2):
                mask = np.expand_dims(mask, -1)
            res = np.sum(mask*res*dout, self.dim)

            np.copyto(out, res)

        def execute(self, a, dim):
            self.save_vars = a
            self.dim = dim
            self.res = jt.numpy_code(
                a.shape,
                a.dtype,
                [a],
                self.forward_code,
            )
            return self.res

        def grad(self, grad_a):
            a = self.save_vars
            b = self.res
            return jt.numpy_code(
                a.shape,
                a.dtype,
                [a, b, grad_a],
                self.backward_code,
            )

    func = CumprodFunc()
    if dim<0:
        dim+=len(a.shape)
    return func(a, dim)
