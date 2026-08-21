"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt
from jittor import nn
import numpy as np

from ..functional import (
    _diff, _isin, _repeat_interleave, _trapz,
)
from ..grad import (
    _AutocastContext,
)
from ..nested import (
    _NestedTensor,
)
from ..types import (
    _dtype_to_str,
)

def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    import collections as _collections
    def _alias(name, fn):
        if not hasattr(g, name):
            setattr(g, name, fn)
    # complex-dtype API (#3): jittor represents complex via nn.ComplexNumber (real/imag
    # pair); wire the torch entry points onto it. torch.complex(re,im), view_as_complex
    # (last dim of 2 -> complex), view_as_real (complex -> last dim of 2), polar, real/
    # imag/conj/is_complex. The arithmetic (* / + matmul exp conj) is on ComplexNumber.
    _CN = jt.nn.ComplexNumber
    # A complex value is either the legacy ComplexNumber (still produced by torch.complex and
    # consumed by torch.fft.* -- migrated in P3) OR the native complex64 dtype (Phase 6). The
    # accessors below handle both; Var.real/imag/angle are patched in jittor.nn. We force-set
    # (not _alias) the accessors because _alias skips names that already exist as native ops --
    # that is why torch.conj(ComplexNumber) used to fall through to the native conj op and crash.
    def _is_cplx(x):
        return isinstance(x, _CN) or (isinstance(x, Var) and "complex" in str(x.dtype))
    _alias("complex", lambda real, imag, **k: jt.nn.view_as_complex(jt.stack([real, imag], dim=-1)))  # native complex64
    _alias("view_as_complex", lambda x: jt.nn.view_as_complex(x))   # -> native complex64
    _alias("view_as_real", lambda x: jt.nn.view_as_real(x))         # polymorphic
    g.is_complex = lambda x: _is_cplx(x)
    g.real = lambda x: x.real if isinstance(x, (_CN, Var)) else x
    g.imag = lambda x: x.imag if isinstance(x, (_CN, Var)) else jt.zeros_like(x)
    g.polar = lambda abs, angle, **k: jt.nn.polar(abs, angle)       # -> native complex64
    g.conj = lambda x: x.conj() if isinstance(x, (_CN, Var)) else x
    g.angle = lambda x: x.angle() if isinstance(x, (_CN, Var)) else jt.zeros_like(x)
    # torch.abs of a complex tensor is its magnitude; jittor's abs only takes real Vars.
    _jt_abs = jt.abs
    def _abs(x):
        return x.abs() if isinstance(x, _CN) else _jt_abs(x)
    g.abs = _abs
    Var.abs = lambda self: _jt_abs(self)

    # ``jittor.fft`` is the native owner. Torch mode publishes that same module
    # object under its historical namespace instead of carrying a duplicate DFT.
    from jittor import fft as _fft_ns
    g.fft = _fft_ns
    _modules["torch.fft"] = _fft_ns
    # torch.softmax / log_softmax / relu top-level function forms (convbert calls
    # torch.softmax(x, dim=...)). jittor exposes these via nn, not the top level.
    _alias("softmax", lambda input, dim=None, **k: jt.nn.softmax(input, dim=dim))
    _alias("log_softmax", lambda input, dim=None, **k: jt.nn.log_softmax(input, dim=dim))
    _alias("relu", lambda input, **k: jt.nn.relu(input))
    # elementwise / functional top-level forms missing from jittor's top level
    _alias("log1p", lambda x: jt.log(1.0 + x))
    _alias("reciprocal", lambda x: 1.0 / x)
    _alias("lerp", lambda input, end, weight: input + weight * (end - input))
    def _isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **k):
        out = jt.abs(a - b) <= (atol + rtol * jt.abs(b))
        if equal_nan:
            out = out | (jt.isnan(a) & jt.isnan(b))
        return out
    _alias("isclose", _isclose)
    def _allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, **k):
        return bool(_isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan).all().item())
    _alias("allclose", _allclose)
    _alias("cosine_similarity", lambda x1, x2, dim=1, eps=1e-8: nn.cosine_similarity(x1, x2, dim=dim, eps=eps))
    _alias("pairwise_distance", lambda x1, x2, p=2.0, eps=1e-6, keepdim=False:
           nn.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim))
    # torch.take_along_dim(input, indices, dim): like gather, but torch BROADCASTS
    # indices against input on every dim except `dim` first. transformers' beam search
    # _gather_beams passes indices of shape (batch, k, 1) to gather full sequences of
    # shape (batch, beams, seq_len) along dim=1 -> expects (batch, k, seq_len). A plain
    # jt.gather returns the index's shape (batch, k, 1), collapsing seq_len -> beam
    # search crashed on the next `seq[:, :, cur_len] = ...` setitem. Broadcast first.
    def _take_along_dim(input, indices, dim=None):
        if dim is None:
            return jt.gather(input.reshape(-1), 0, indices.reshape(-1))
        nd = input.ndim
        d = dim % nd
        target = list(input.shape)
        target[d] = indices.shape[d]            # keep index extent along the gather dim
        if list(indices.shape) != target:
            indices = jt.broadcast(indices, target)   # broadcast size-1 dims to input
        return jt.gather(input, d, indices)
    _alias("take_along_dim", _take_along_dim)
    # torch.all/any accept numpy-style axis=/keepdims= aliases (transformers' beam
    # search _update_finished_beams: torch.all(x, axis=-1, keepdims=True)). jittor's
    # native all/any take only `dim` and have no keepdims. Wrap to accept both spellings
    # (dim/axis, keepdim/keepdims) while staying backward-compatible with all(x)/all(x,d).
    def _reduce_alias(orig):
        def f(input, dim=None, keepdim=False, *, axis=None, keepdims=None, out=None):
            d = axis if axis is not None else dim
            kd = keepdims if keepdims is not None else keepdim
            if d is None or d == ():
                return orig(input)
            r = orig(input, d)
            if kd:
                dims = (d,) if isinstance(d, int) else tuple(d)
                nd = input.ndim
                for dd in sorted(x % nd for x in dims):
                    r = r.unsqueeze(dd)
            return r
        return f
    _orig_all = getattr(g, "all", None)
    _orig_any = getattr(g, "any", None)
    if callable(_orig_all):
        g.all = _reduce_alias(_orig_all)
    if callable(_orig_any):
        g.any = _reduce_alias(_orig_any)
    def _movedim(x, source, destination):
        nd = x.ndim
        src = [s % nd for s in (source if isinstance(source, (list, tuple)) else [source])]
        dst = [d % nd for d in (destination if isinstance(destination, (list, tuple)) else [destination])]
        order = [d for d in range(nd) if d not in src]
        for d, s in sorted(zip(dst, src)):
            order.insert(d, s)
        return x.permute(order)
    _alias("movedim", _movedim)
    _alias("moveaxis", _movedim)
    # Var.movedim/moveaxis (the functions exist but weren't bound as methods), plus
    # index_put_/index_put (scatter-style assignment), tensor_split (uneven split), take.
    Var.movedim = lambda self, source, destination: _movedim(self, source, destination)
    Var.moveaxis = lambda self, source, destination: _movedim(self, source, destination)
    def _index_put_(self, indices, values, accumulate=False):
        idx = tuple(indices) if isinstance(indices, (tuple, list)) else (indices,)
        if not accumulate:
            self[idx if len(idx) > 1 else idx[0]] = values
            return self
        # accumulate=True must add ALL contributions at duplicate indices (a plain
        # read-add-write keeps only the last). Route through index_add (dup-correct).
        vals = values if isinstance(values, Var) else jt.array(values)
        if len(idx) == self.ndim:                          # full advanced index -> linearize
            shape = self.shape
            strides = [1] * self.ndim
            for k in range(self.ndim - 2, -1, -1):
                strides[k] = strides[k + 1] * int(shape[k + 1])
            lin = None
            for k, ind in enumerate(idx):
                term = (ind if isinstance(ind, Var) else jt.array(ind)).int64().reshape((-1,)) * strides[k]
                lin = term if lin is None else lin + term
            vflat = vals.reshape((-1,))
            if int(vflat.shape[0]) == 1 and int(lin.shape[0]) > 1:
                vflat = vflat.broadcast(lin.shape)
            self.assign(self.reshape((-1,)).index_add(0, lin, vflat).reshape(shape))
            return self
        if len(idx) == 1:                                  # index along dim 0
            i0 = (idx[0] if isinstance(idx[0], Var) else jt.array(idx[0])).int64().reshape((-1,))
            self.assign(self.index_add(0, i0, vals))
            return self
        raise NotImplementedError("index_put_(accumulate=True) with a partial multi-dim index")
    Var.index_put_ = _index_put_
    Var.index_put = lambda self, indices, values, accumulate=False: _index_put_(self.clone(), indices, values, accumulate)
    # index_copy_(dim, index, source): self[..,index[i],..] = source[i,..] along dim
    # (overwrite, NOT accumulate -- cf. index_add).
    def _index_copy_(self, dim, index, source):
        d = dim % self.ndim
        idx = index if isinstance(index, Var) else jt.array(index)
        if d == 0:
            self[idx] = source
        else:
            sl = [slice(None)] * self.ndim; sl[d] = idx
            self[tuple(sl)] = source
        return self
    Var.index_copy_ = _index_copy_
    Var.index_copy = lambda self, dim, index, source: _index_copy_(self.clone(), dim, index, source)
    g.index_copy = lambda input, dim, index, source: _index_copy_(input.clone(), dim, index, source)
    g.index_put = lambda input, indices, values, accumulate=False: _index_put_(input.clone(), indices, values, accumulate)
    def _tensor_split(self, indices_or_sections, dim=0):
        d = dim % self.ndim
        L = int(self.shape[d])
        def _slice(a, b):
            ix = [slice(None)] * self.ndim; ix[d] = slice(a, b)
            return self[tuple(ix)]
        if isinstance(indices_or_sections, int):
            n = indices_or_sections
            base, rem = L // n, L % n
            sizes = [base + 1] * rem + [base] * (n - rem)
            out, start = [], 0
            for s in sizes:
                out.append(_slice(start, start + s)); start += s
            return out
        pts, out, prev = list(indices_or_sections), [], 0
        for p in pts + [L]:
            out.append(_slice(prev, p)); prev = p
        return out
    Var.tensor_split = _tensor_split
    g.tensor_split = lambda input, indices_or_sections, dim=0: _tensor_split(input, indices_or_sections, dim)
    Var.take = lambda self, index: self.reshape((-1,))[index]
    g.take = lambda input, index: input.reshape((-1,))[index]
    # torch.eye(n, m=None, *, dtype=, ...): identity / rectangular-identity
    # matrix. jittor has no top-level eye (only jt.init.eye), so add one.
    def _eye(n, m=None, dtype=None, **k):
        # torch.eye(n) is the n x n identity; torch.eye(n, m) is n x m.
        # jittor's init.eye requires a 2-element shape (a bare (n,) asserts),
        # so always pass (n, n) / (n, m).
        shape = (int(n), int(n)) if m is None else (int(n), int(m))
        import jittor.init as _init
        return _init.eye(shape, _dtype_to_str(dtype) or "float32")
    _alias("eye", _eye)
    # torch.narrow(input, dim, start, length) / torch.tile(input, dims) --
    # function forms mirroring the Var methods (added in _install_tensor_methods).
    _alias("narrow", lambda input, dim, start, length: input.narrow(dim, start, length))
    _alias("tile", lambda input, *dims: input.tile(*dims))
    # torch.equal returns a Python bool (True iff same shape & all elements
    # equal). jittor's native `equal` is elementwise, so force-override.
    def _torch_equal(a, b):
        try:
            if isinstance(a, _NestedTensor) or isinstance(b, _NestedTensor):
                return bool(a.equal(b)) if isinstance(a, _NestedTensor) else False
            if not isinstance(a, jt.Var) or not isinstance(b, jt.Var):
                return bool(a == b)
            if tuple(a.shape) != tuple(b.shape):
                return False
            if a.numel() == 0:
                return True
            return bool((a == b).all().item())
        except Exception:
            return False
    g.equal = _torch_equal
    Var.equal = lambda self, other: _torch_equal(self, other)
    _alias("diff", lambda x, n=1, dim=-1, prepend=None, append=None:
           _diff(x, n=n, dim=dim, prepend=prepend, append=append))
    _alias("trapz", _trapz)
    _alias("trapezoid", _trapz)
    _alias("repeat_interleave", _repeat_interleave)
    _alias("autocast", lambda *a, **k: _AutocastContext())
    # Real loop-based torch.vmap. The old no-op stub (`lambda fn,*a,**k: fn`)
    # ignored in_dims/out_dims, so transformers' vmap-based causal-mask builder
    # (taken when a model passes and_mask/or_mask -- e.g. falcon) collapsed to a
    # single direct call and produced a wrong all-True (seq,) mask instead of the
    # (b,1,q,kv) causal triangle -> bidirectional attention -> ~79% forward error.
    # Map over in_dims and stack along out_dims. jittor has no 0-d tensors, so a
    # scalar leaf is (1,) where torch has (); collapse that spurious trailing
    # singleton so the stacked rank matches torch.vmap.
    def _vmap(func, in_dims=0, out_dims=0, *_a, **_k):
        def wrapped(*args):
            ids = (in_dims,) * len(args) if (isinstance(in_dims, int) or in_dims is None) else tuple(in_dims)
            size = None
            for a, d in zip(args, ids):
                if d is not None:
                    size = int(a.shape[d]); break
            if size is None:
                return func(*args)
            outs = []
            for i in range(size):
                sub = []
                for a, d in zip(args, ids):
                    if d is None:
                        sub.append(a)
                    else:
                        idx = [slice(None)] * a.ndim; idx[d] = i
                        sub.append(a[tuple(idx)])
                r = func(*sub)
                if not isinstance(r, jt.Var):
                    r = jt.array(r)
                outs.append(r)
            if all(o.ndim >= 1 and o.shape[-1] == 1 for o in outs) and all(o.ndim == outs[0].ndim for o in outs):
                outs = [o.reshape(o.shape[:-1]) if o.ndim > 1 else o for o in outs]
            od = out_dims if isinstance(out_dims, int) else (out_dims[0] if out_dims else 0)
            return jt.stack(outs, dim=od)
        return wrapped
    _alias("vmap", _vmap)
    _alias("outer", lambda a, b: jt.matmul(a.reshape(-1, 1), b.reshape(1, -1)))
    _alias("isin", _isin)
    # torch.cdist(x1,x2,p): pairwise p-distances (...,P,M),(...,R,M)->(...,P,R). Used by
    # contrastive/clustering/retrieval. torch.bucketize: indices to insert into sorted
    # boundaries (samplers / piecewise schedules).
    def _cdist(x1, x2, p=2.0, compute_mode=None, **k):
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)          # (...,P,R,M)
        if p == 2:
            return jt.sqrt((diff * diff).sum(-1))
        if p == 1:
            return jt.abs(diff).sum(-1)
        return (jt.abs(diff) ** p).sum(-1) ** (1.0 / p)
    _alias("cdist", _cdist)
    def _bucketize(input, boundaries, out_int32=False, right=False, **k):
        b = boundaries.reshape((-1,))
        cmp = (input.unsqueeze(-1) >= b) if right else (input.unsqueeze(-1) > b)
        r = cmp.int32().sum(-1)
        return r if out_int32 else r.int64()
    _alias("bucketize", _bucketize)
    # trace / diag_embed / diagflat / kron / logcumsumexp / tensordot / pdist.
    def _trace(input):
        k = min(int(input.shape[0]), int(input.shape[1]))
        ar = jt.arange(k)
        return input[ar, ar].sum()
    _alias("trace", _trace); Var.trace = _trace
    def _diag_embed(input, offset=0, dim1=-2, dim2=-1):
        N = int(input.shape[-1])
        return input.unsqueeze(-1) * jt.init.eye(N)
    _alias("diag_embed", _diag_embed); Var.diag_embed = lambda self, offset=0, dim1=-2, dim2=-1: _diag_embed(self)
    _alias("diagflat", lambda input, offset=0: _diag_embed(input.reshape((-1,))))
    def _kron(a, b):
        nd = max(a.ndim, b.ndim)
        a2 = a.reshape((1,) * (nd - a.ndim) + tuple(a.shape))
        b2 = b.reshape((1,) * (nd - b.ndim) + tuple(b.shape))
        aex, bex, fin = [], [], []
        for i in range(nd):
            aex += [int(a2.shape[i]), 1]; bex += [1, int(b2.shape[i])]
            fin.append(int(a2.shape[i]) * int(b2.shape[i]))
        return (a2.reshape(aex) * b2.reshape(bex)).reshape(fin)
    _alias("kron", _kron); Var.kron = _kron
    def _logcumsumexp(input, dim):
        m = input.max(dim, keepdims=True)
        return m + jt.log(jt.cumsum(jt.exp(input - m), dim))
    _alias("logcumsumexp", _logcumsumexp); Var.logcumsumexp = _logcumsumexp
    def _tensordot(a, b, dims=2):
        if isinstance(dims, int):
            adims, bdims = list(range(a.ndim - dims, a.ndim)), list(range(dims))
        else:
            adims, bdims = list(dims[0]), list(dims[1])
        a_free = [i for i in range(a.ndim) if i not in adims]
        b_free = [i for i in range(b.ndim) if i not in bdims]
        import numpy as _np_td
        af = int(_np_td.prod([int(a.shape[i]) for i in a_free])) if a_free else 1
        cs = int(_np_td.prod([int(a.shape[i]) for i in adims])) if adims else 1
        bf = int(_np_td.prod([int(b.shape[i]) for i in b_free])) if b_free else 1
        out = jt.matmul(a.permute(a_free + adims).reshape((af, cs)), b.permute(bdims + b_free).reshape((cs, bf)))
        fin = [int(a.shape[i]) for i in a_free] + [int(b.shape[i]) for i in b_free]
        return out.reshape(fin) if fin else out.reshape((1,))   # full contraction -> scalar (jittor (1,))
    _alias("tensordot", _tensordot)
    def _pdist(input, p=2.0):
        N = int(input.shape[0])
        diff = input.unsqueeze(1) - input.unsqueeze(0)
        d = ((jt.abs(diff) ** p).sum(-1)) ** (1.0 / p)
        ii = [i for i in range(N) for j in range(i + 1, N)]
        jj = [j for i in range(N) for j in range(i + 1, N)]
        return d[jt.array(ii), jt.array(jj)]
    _alias("pdist", _pdist)
    # shape ops: unflatten / swapaxes / swapdims / ravel + numpy-style stacking helpers.
    def _unflatten(input, dim, sizes):
        d = dim % input.ndim
        return input.reshape(list(input.shape[:d]) + list(sizes) + list(input.shape[d + 1:]))
    _alias("unflatten", _unflatten); Var.unflatten = _unflatten
    def _swapaxes(input, axis0, axis1):
        perm = list(range(input.ndim))
        a, b = axis0 % input.ndim, axis1 % input.ndim
        perm[a], perm[b] = perm[b], perm[a]
        return input.permute(perm)
    _alias("swapaxes", _swapaxes); _alias("swapdims", _swapaxes)
    Var.swapaxes = _swapaxes; Var.swapdims = _swapaxes
    _alias("ravel", lambda input: input.reshape((-1,))); Var.ravel = lambda self: self.reshape((-1,))
    def _vstack(tensors):
        return jt.concat([t if t.ndim >= 2 else t.reshape((1, -1)) for t in tensors], dim=0)
    _alias("vstack", _vstack); _alias("row_stack", _vstack)
    _alias("hstack", lambda tensors: jt.concat(list(tensors), dim=0) if all(t.ndim == 1 for t in tensors)
           else jt.concat(list(tensors), dim=1))
    def _dstack(tensors):
        out = []
        for t in tensors:
            out.append(t.reshape((1, -1, 1)) if t.ndim == 1 else (t.unsqueeze(-1) if t.ndim == 2 else t))
        return jt.concat(out, dim=2)
    _alias("dstack", _dstack)
    _alias("column_stack", lambda tensors: jt.concat([t.reshape((-1, 1)) if t.ndim == 1 else t for t in tensors], dim=1))
    # element-wise ops: copysign / xlogy / heaviside / float_power / signbit.
    def _copysign(input, other):
        s = (other >= 0).float32() * 2 - 1                 # +1 where other>=0 (incl +0), -1 else
        return jt.abs(input) * s
    _alias("copysign", _copysign); Var.copysign = _copysign
    def _xlogy(input, other):
        return jt.ternary(input == 0, jt.zeros_like(input), input * jt.log(other))  # xlogy(0,y)=0
    _alias("xlogy", _xlogy); Var.xlogy = _xlogy
    def _heaviside(input, values):
        return (input > 0).float32() + (input == 0).float32() * values
    _alias("heaviside", _heaviside); Var.heaviside = _heaviside
    def _float_power(input, exponent):
        b = exponent.float64() if isinstance(exponent, Var) else exponent
        return (input.float64() ** b)
    _alias("float_power", _float_power); Var.float_power = _float_power
    _alias("signbit", lambda input: input < 0); Var.signbit = lambda self: self < 0
    # reductions: logsumexp (attention/MoE/loss/beam), nansum/nanmean, std_mean/var_mean,
    # aminmax, quantile. NaN handling uses nan_to_num plus an explicit isnan mask.
    def _logsumexp(input, dim, keepdim=False):
        m = input.max(dim, keepdims=True)
        out = m + jt.log(jt.exp(input - m).sum(dim, keepdims=True))
        if keepdim:
            return out
        # torch removes the reduced dim(s) entirely (1D -> 0-dim scalar). jittor's
        # squeeze keeps a trailing (1,) for the last remaining dim, so reshape to
        # the explicit reduced shape instead.
        dims = [dim] if isinstance(dim, int) else list(dim)
        nd = input.ndim
        dims = [d % nd for d in dims]
        target = [s for i, s in enumerate(input.shape) if i not in dims]
        # jittor has no 0-dim tensors; a full reduction stays (1,).
        return out.reshape(target) if target else out.reshape(-1)
    _alias("logsumexp", _logsumexp); Var.logsumexp = _logsumexp
    def _nansum(input, dim=None, keepdim=False, **k):
        z = jt.nan_to_num(input, nan=0.0)
        return z.sum() if dim is None else z.sum(dim, keepdims=keepdim)
    _alias("nansum", _nansum); Var.nansum = _nansum
    def _nanmean(input, dim=None, keepdim=False, **k):
        # Keep the non-NaN count explicit rather than coupling it to comparison codegen.
        cnt = 1.0 - jt.isnan(input).float32()
        z = jt.nan_to_num(input, nan=0.0)
        if dim is None:
            return z.sum() / cnt.sum()
        return z.sum(dim, keepdims=keepdim) / cnt.sum(dim, keepdims=keepdim)
    _alias("nanmean", _nanmean); Var.nanmean = _nanmean
    def _std_mean(input, dim=None, unbiased=True, keepdim=False, correction=None, **k):
        mean = input.mean() if dim is None else input.mean(dim, keepdims=keepdim)
        std = input.std() if dim is None else input.std(dim)  # jittor std is unbiased
        return (std, mean)
    _alias("std_mean", _std_mean)
    def _var_mean(input, dim=None, unbiased=True, keepdim=False, correction=None, **k):
        s, m = _std_mean(input, dim, unbiased, keepdim)
        return (s * s, m)
    _alias("var_mean", _var_mean)
    _AMinMax = _collections.namedtuple("aminmax", ["min", "max"])
    def _aminmax(input, dim=None, keepdim=False):
        if dim is None:
            return _AMinMax(input.min(), input.max())
        return _AMinMax(input.min(dim, keepdims=keepdim), input.max(dim, keepdims=keepdim))
    _alias("aminmax", _aminmax); Var.aminmax = _aminmax
    def _quantile(input, q, dim=None, keepdim=False, interpolation="linear", **k):
        import numpy as _np_q
        arr = input.numpy()
        qn = q.numpy() if isinstance(q, Var) else q
        r = _np_q.quantile(arr, qn, axis=dim, keepdims=keepdim)
        return jt.array(r.astype("float32"))
    _alias("quantile", _quantile)
    def _nanquantile(input, q, dim=None, keepdim=False, interpolation="linear", **k):
        import numpy as _np_q
        arr = input.numpy()
        qn = q.numpy() if isinstance(q, Var) else q
        r = _np_q.nanquantile(arr, qn, axis=dim, keepdims=keepdim)
        return jt.array(r.astype("float32"))
    _alias("nanquantile", _nanquantile)
    _alias("square", lambda x: x * x)   # torch.square (jittor only had jt.sqr); persimmon
    # torch.addmm(input, mat1, mat2, *, beta=1, alpha=1):
    #   out = beta * input + alpha * (mat1 @ mat2)   (gpt2 uses this for its
    #   Conv1D linear). jittor has no top-level addmm, so add one.
    def _addmm(input, mat1, mat2, *, beta=1, alpha=1):
        res = jt.matmul(mat1, mat2)
        if alpha != 1:
            res = res * alpha
        if beta == 0:
            return res
        return beta * input + res
    _alias("addmm", _addmm)

    # ---- torch.* ops used by mmdetection (additive aliases) ----
    _alias("mm", lambda input, mat2, out=None: jt.matmul(input, mat2))   # 2-D matmul
    def _mv(input, vec, out=None):
        if input.ndim != 2 or vec.ndim != 1:
            raise RuntimeError(
                f"mv: expected a 2-D matrix and a 1-D vector, got "
                f"{input.ndim}-D and {vec.ndim}-D tensors")
        if input.shape[1] != vec.shape[0]:
            raise RuntimeError(
                f"mv: size mismatch, matrix has {input.shape[1]} columns but "
                f"vector has {vec.shape[0]} elements")
        result = jt.matmul(input, vec)
        if out is not None:
            out.assign(result)
            return out
        return result
    _alias("mv", _mv)
    _alias("masked_select", lambda input, mask, out=None: input[mask])   # -> 1-D selected
    _alias("split_with_sizes",
           lambda input, split_sizes, dim=0: input.split(split_sizes, dim))
    _alias("_shape_as_tensor",
           lambda input: jt.array(np.asarray(input.shape, dtype=np.int64)))
    def _nan_to_num_inplace(input, nan=0.0, posinf=None, neginf=None):
        r = g.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf)
        try:
            input.assign(r); return input          # honour in-place semantics
        except Exception:
            return r
    _alias("nan_to_num_", _nan_to_num_inplace)
    # torch.randint_like(input, low, high=None, *, dtype=...): jittor's native lacks
    # the dtype kwarg (DINO's denoising uses it). Force-override with torch semantics.
    def _randint_like(input, low, high=None, dtype=None, device=None,
                      requires_grad=False, **kw):
        if high is None:
            low, high = 0, low
        r = jt.randint(int(low), int(high), tuple(int(s) for s in input.shape))
        return r.cast(_dtype_to_str(dtype)) if dtype is not None else r
    g.randint_like = _randint_like

    # torch.sparse_coo_tensor + torch.sparse.sum: mmdet's free_anchor head builds a
    # (hybrid) COO tensor then immediately densifies it. Back it with a dense Var
    # materialised eagerly via index_add_ (COO accumulates duplicate coordinates).
    class _SparseCOO:
        def __init__(self, dense): self._dense = dense
        def to_dense(self): return self._dense
        @property
        def shape(self): return self._dense.shape
        @property
        def dtype(self): return self._dense.dtype
        def t(self): return _SparseCOO(self._dense.t())
        def sum(self, dim=None):
            return _SparseCOO(self._dense.sum(dim) if dim is not None else self._dense.sum())
    def _sparse_coo_tensor(indices, values, size=None, dtype=None, device=None,
                           requires_grad=False, **kw):
        if not isinstance(indices, jt.Var): indices = jt.array(indices)
        if not isinstance(values, jt.Var): values = jt.array(values)
        S = int(indices.shape[0])
        nnz = int(indices.shape[1]) if indices.ndim == 2 else int(indices.shape[0])
        tail = [int(d) for d in values.shape[1:]]
        idx_np = indices.numpy().astype("int64").reshape(S, -1)
        if size is not None:
            full = [int(s) for s in size]
        else:
            full = [int(idx_np[s].max()) + 1 if nnz > 0 else 0 for s in range(S)] + tail
        sparse_shape = full[:S]; tail2 = full[S:]
        prod = 1
        for d in sparse_shape: prod *= int(d)
        lin = np.zeros(nnz, dtype="int64"); stride = 1     # row-major linear index
        for s in range(S - 1, -1, -1):
            lin = lin + idx_np[s] * stride
            stride *= int(sparse_shape[s])
        flat = jt.zeros([prod] + tail2, dtype=str(values.dtype))
        if nnz > 0:
            flat.index_add_(0, jt.array(lin), values.reshape([nnz] + tail2))  # in-place
        return _SparseCOO(flat.reshape(sparse_shape + tail2))
    _alias("sparse_coo_tensor", _sparse_coo_tensor)
    import jittor.sparse as _jt_sparse
    if not hasattr(_jt_sparse, "sum"):
        def _sparse_sum(x, dim=None):
            d = x._dense if isinstance(x, _SparseCOO) else x
            return _SparseCOO(d.sum(dim) if dim is not None else d.sum())
        _jt_sparse.sum = _sparse_sum

    # det/inverse on (batched) square matrices (mmrotate GWD/KLD/KFIoU Gaussian losses)
    def _vdet(self):
        import jittor.linalg as _la; return _la.det(self)
    def _vinv(self):
        import jittor.linalg as _la; return _la.inv(self)
    if not hasattr(Var, "det"):       Var.det = _vdet
    if not hasattr(Var, "inverse"):   Var.inverse = _vinv
    g.det = lambda x: _vdet(x)
    g.inverse = lambda x: _vinv(x)

    # ---- linalg (peft / lora init need svd_lowrank, svd) ----
    def _svd(x, some=True, compute_uv=True, **kw):
        import jittor.linalg as _la
        u, s, v = _la.svd(x)
        return u, s, v
    def _svd_lowrank(A, q=6, niter=2, M=None):
        # torch.svd_lowrank returns (U, S, V) of a rank-q approximation.
        import jittor.linalg as _la
        if M is not None:
            A = A - M
        u, s, v = _la.svd(A)
        q = min(q, s.shape[0])
        return u[:, :q], s[:q], v[:, :q]
    _alias("svd", _svd)
    _alias("svd_lowrank", _svd_lowrank)
    _alias("pca_lowrank", lambda A, q=6, center=True, niter=2: _svd_lowrank(
        A - (A.mean(0, keepdims=True) if center else 0), q, niter))


def install_parity(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    def module(name):
        return registry.ensure(name)
    import jittor.linalg as linalg
    registry.publish("torch.linalg", linalg)
    g.linalg = linalg

    import jittor.sparse as sparse
    registry.publish("torch.sparse", sparse)
    g.sparse = sparse

    special = module("torch.special")
    for name in ("erf", "erfc", "exp", "expm1", "log1p", "sinc"):
        value = getattr(g, name, None)
        if value is not None:
            setattr(special, name, value)
    special.expit = getattr(g, "sigmoid")
    g.special = special
