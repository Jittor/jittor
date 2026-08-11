"""Torch-grade scatter/gather/index-op-semantics tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (sibling of ``test_torch_compat_indexing.py``).
Every check compares jittor-as-torch against an INDEPENDENT numpy reference computed from
scratch here, and runs on BOTH CPU and CUDA when the build has it, so it locks torch-API
*semantics* (transformers/diffusers lean on these heavily for masking/beam-search/MoE).

Covered: ``Tensor.scatter_``/``scatter`` (overwrite + reduce='add'/'multiply', SCALAR src),
``scatter_add``/``scatter_add_`` (out-of-place vs in-place), ``scatter_reduce``
(sum/prod/amax/amin/mean, include_self True/False), ``gather`` (dim0/dim1/3-D),
``index_select`` (every dim, method + functional), ``index_add``/``index_add_``
(DUP-ACCUMULATE -- the torch contract), ``index_copy_``, ``index_fill``/``index_fill_``,
``masked_fill``/``masked_fill_``, ``masked_select``, ``take``, ``take_along_dim``
(incl. torch's broadcast-then-gather), ``one_hot``.

The torch contract that bites people: ``scatter_`` (overwrite) leaves duplicate-write
order unspecified, but ``scatter_add_`` / ``index_add_`` ACCUMULATE all contributions at
duplicate indices -- so those are tested with deliberate collisions.

Run:  python -m pytest tests/compat/torch/test_torch_compat_scatter.py
      python -m pytest tests/compat/torch/test_torch_compat_scatter.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt

F = jt.nn.functional

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def t(a):
    """np array -> jittor Var (keep dtype, including int32/int64/bool index tensors)."""
    return torch.array(a)


# ---------------------------------------------------------------------------
# Independent numpy references (the "gold" side of every check).
# ---------------------------------------------------------------------------
def np_scatter(base, dim, idx, src, reduce=None):
    """torch scatter semantics: out[.. idx[m] @dim ..] (op)= src[m] for every m."""
    out = base.copy()
    if np.isscalar(src) or (isinstance(src, np.ndarray) and src.ndim == 0):
        scalar = float(src)
        src = None
    else:
        scalar = None
    it = np.nditer(idx, flags=["multi_index"])
    for v in it:
        mi = it.multi_index
        dst = list(mi)
        dst[dim] = int(v)
        val = scalar if scalar is not None else src[mi]
        if reduce == "add":
            out[tuple(dst)] += val
        elif reduce == "multiply":
            out[tuple(dst)] *= val
        else:
            out[tuple(dst)] = val
    return out


def np_scatter_reduce(base, dim, idx, src, reduce, include_self=True):
    """torch Tensor.scatter_reduce reference (sum/prod/amax/amin/mean)."""
    out = base.astype(np.float64).copy()
    # per-cell python lists of incoming src values
    buckets = {}
    it = np.nditer(idx, flags=["multi_index"])
    for v in it:
        mi = it.multi_index
        dst = list(mi); dst[dim] = int(v)
        buckets.setdefault(tuple(dst), []).append(float(src[mi]))
    for cell, vals in buckets.items():
        seed = [out[cell]] if include_self else []
        allv = seed + vals
        if reduce == "sum":
            out[cell] = sum(allv)
        elif reduce == "prod":
            p = 1.0
            for a in allv: p *= a
            out[cell] = p
        elif reduce in ("amax", "max"):
            out[cell] = max(allv)
        elif reduce in ("amin", "min"):
            out[cell] = min(allv)
        elif reduce == "mean":
            out[cell] = sum(allv) / len(allv)
    return out


def np_index_add(base, dim, index, source, alpha=1.0):
    """torch index_add: out[.., index[k], ..] += alpha*source[.., k, ..], ACCUMULATING dups."""
    out = base.astype(np.float64).copy()
    src = source.astype(np.float64) * alpha
    for k, j in enumerate(index):
        sl_dst = [slice(None)] * out.ndim; sl_dst[dim] = int(j)
        sl_src = [slice(None)] * src.ndim; sl_src[dim] = k
        out[tuple(sl_dst)] += src[tuple(sl_src)]
    return out


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_array_equal(g, r, err_msg=msg)


# ---------------------------------------------------------------------------
# scatter / scatter_  (overwrite + reduce)
# ---------------------------------------------------------------------------
class TestScatter(Base):
    def test_scatter_overwrite_dim1(self):
        base = np.zeros((3, 5), dtype="float32")
        # distinct destinations per column-row pair (overwrite is order-unspecified for dups)
        idx = np.array([[0, 1, 2], [0, 1, 4]], dtype="int64")
        src = np.arange(1, 7).reshape(2, 3).astype("float32")
        ref = np_scatter(base, 1, idx, src)
        def body(dev):
            self.ac(torch.scatter(t(base), 1, t(idx), t(src)).numpy(), ref,
                    msg=f"scatter overwrite d1 {dev}")
        both_devices(body)

    def test_scatter_overwrite_dim0(self):
        base = np.zeros((3, 5), dtype="float32")
        idx = np.array([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype="int64")
        src = np.arange(1, 11).reshape(2, 5).astype("float32")
        ref = np_scatter(base, 0, idx, src)
        def body(dev):
            self.ac(torch.scatter(t(base), 0, t(idx), t(src)).numpy(), ref,
                    msg=f"scatter overwrite d0 {dev}")
        both_devices(body)

    def test_scatter_scalar_src(self):
        # torch allows a scalar src: x.scatter_(dim, index, value). Used to mask
        # MoE logits with -inf (phimoe) etc.
        base = np.full((2, 4), 2.0, dtype="float32")
        idx = np.array([[2], [3]], dtype="int64")
        ref = np_scatter(base, 1, idx, 7.5)
        def body(dev):
            self.ac(torch.scatter(t(base), 1, t(idx), 7.5).numpy(), ref,
                    msg=f"scatter scalar {dev}")
        both_devices(body)

    def test_scatter_reduce_add_collision(self):
        base = np.zeros((3, 4), dtype="float32")
        idx = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype="int64")  # heavy collision
        src = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype="float32")
        ref = np_scatter(base, 0, idx, src, reduce="add")
        def body(dev):
            self.ac(torch.scatter(t(base), 0, t(idx), t(src), reduce="add").numpy(), ref,
                    msg=f"scatter reduce=add {dev}")
        both_devices(body)

    def test_scatter_reduce_multiply(self):
        base = np.full((2, 4), 2.0, dtype="float32")
        idx = np.array([[2, 3]], dtype="int64")
        src = np.array([[1.23, 1.23]], dtype="float32")
        ref = np_scatter(base, 1, idx, src, reduce="multiply")
        def body(dev):
            self.ac(torch.scatter(t(base), 1, t(idx), t(src), reduce="multiply").numpy(), ref,
                    atol=1e-4, msg=f"scatter reduce=multiply {dev}")
        both_devices(body)

    def test_scatter__inplace_mutates_and_returns(self):
        # scatter along dim 1 (size 3) so index values 0,1,2 are in range.
        idx = np.array([[0, 1, 2]], dtype="int64")
        src = np.array([[5., 6., 7.]], dtype="float32")
        def body(dev):
            x = jt.zeros((2, 3))
            r = x.scatter_(1, t(idx), t(src))
            ref = np_scatter(np.zeros((2, 3), "float32"), 1, idx, src)
            self.ac(x.numpy(), ref, msg=f"scatter_ mutated self {dev}")
            self.ac(r.numpy(), ref, msg=f"scatter_ returns self {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# scatter_add / scatter_add_   (out-of-place vs in-place; DUP-accumulate)
# ---------------------------------------------------------------------------
class TestScatterAdd(Base):
    def test_scatter_add_accumulates(self):
        base = np.array([[1., 1., 1., 1.]], dtype="float32")
        idx = np.array([[0, 0, 0, 0]], dtype="int64")          # all into col 0
        src = np.array([[1., 2., 3., 4.]], dtype="float32")
        ref = np_scatter(base, 1, idx, src, reduce="add")       # [[1+10, 1,1,1]]
        def body(dev):
            self.ac(t(base).scatter_add(1, t(idx), t(src)).numpy(), ref,
                    msg=f"scatter_add accumulate {dev}")
        both_devices(body)

    def test_scatter_add_out_of_place_keeps_base(self):
        base = np.zeros((2, 3), dtype="float32")
        idx = np.array([[0, 1, 2]], dtype="int64")
        src = np.array([[1., 2., 3.]], dtype="float32")
        def body(dev):
            x = t(base)
            _ = x.scatter_add(0, t(idx), t(src))
            self.ac(x.numpy(), base, msg=f"scatter_add base unchanged {dev}")
        both_devices(body)

    def test_scatter_add__in_place_mutates_base(self):
        base = np.zeros((2, 3), dtype="float32")
        idx = np.array([[0, 0, 0]], dtype="int64")
        src = np.array([[1., 2., 3.]], dtype="float32")
        ref = np_scatter(base, 0, idx, src, reduce="add")
        def body(dev):
            x = t(base)
            x.scatter_add_(0, t(idx), t(src))
            self.ac(x.numpy(), ref, msg=f"scatter_add_ in-place {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# scatter_reduce  (sum/prod/amax/amin/mean, include_self True/False)
# ---------------------------------------------------------------------------
class TestScatterReduce(Base):
    def setUp(self):
        self.base = np.array([[1., 2., 3., 4.]], dtype="float32")
        self.idx = np.array([[0, 0, 0, 0]], dtype="int64")
        self.src = np.array([[5., 1., 9., 2.]], dtype="float32")

    def _check(self, reduce, include_self, atol=1e-5):
        ref = np_scatter_reduce(self.base, 1, self.idx, self.src, reduce, include_self)
        def body(dev):
            g = t(self.base).scatter_reduce(1, t(self.idx), t(self.src), reduce,
                                            include_self=include_self).numpy()
            self.ac(g, ref, atol=atol, msg=f"scatter_reduce {reduce} incl={include_self} {dev}")
        both_devices(body)

    def test_sum_incl(self):     self._check("sum", True)
    def test_sum_excl(self):     self._check("sum", False)
    def test_prod_incl(self):    self._check("prod", True, atol=1e-3)
    def test_amax_incl(self):    self._check("amax", True)
    def test_amax_excl(self):    self._check("amax", False)
    def test_amin_incl(self):    self._check("amin", True)
    def test_mean_incl(self):    self._check("mean", True)
    def test_mean_excl(self):    self._check("mean", False)

    def test_scatter_reduce_non_receiving_cells_untouched(self):
        # include_self=False: cells that receive NOTHING keep their original self value.
        base = np.array([[10., 20., 30.]], dtype="float32")
        idx = np.array([[0]], dtype="int64")        # only col 0 receives
        src = np.array([[7.]], dtype="float32")
        ref = np_scatter_reduce(base, 1, idx, src, "amax", include_self=False)
        self.assertEqual(ref[0, 1], 20.0)           # untouched
        def body(dev):
            g = t(base).scatter_reduce(1, t(idx), t(src), "amax", include_self=False).numpy()
            self.ac(g, ref, msg=f"scatter_reduce keep non-recv {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------
class TestGather(Base):
    def test_gather_dim1(self):
        x = np.random.RandomState(12).randn(3, 4).astype("float32")
        idx = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1]], dtype="int64")
        def body(dev):
            self.ac(torch.gather(t(x), 1, t(idx)).numpy(),
                    np.take_along_axis(x, idx, axis=1), msg=f"gather d1 {dev}")
        both_devices(body)

    def test_gather_dim0(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        idx = np.array([[0, 1, 2, 1], [2, 2, 0, 0]], dtype="int64")
        def body(dev):
            self.ac(torch.gather(t(x), 0, t(idx)).numpy(),
                    np.take_along_axis(x, idx, axis=0), msg=f"gather d0 {dev}")
        both_devices(body)

    def test_gather_3d(self):
        x = np.random.RandomState(99).randn(2, 3, 4).astype("float32")
        idx = np.random.RandomState(100).randint(0, 4, size=(2, 3, 2)).astype("int64")
        def body(dev):
            self.ac(torch.gather(t(x), 2, t(idx)).numpy(),
                    np.take_along_axis(x, idx, axis=2), msg=f"gather 3d d2 {dev}")
        both_devices(body)

    def test_gather_method_form(self):
        x = np.arange(8).reshape(2, 4).astype("float32")
        idx = np.array([[3, 0], [1, 2]], dtype="int64")
        def body(dev):
            self.ac(t(x).gather(1, t(idx)).numpy(),
                    np.take_along_axis(x, idx, axis=1), msg=f"gather method {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# index_select  (every dim; method + functional)
# ---------------------------------------------------------------------------
class TestIndexSelect(Base):
    def setUp(self):
        # all dims >= 3 so idx=[0,2] is valid on every axis.
        self.x = np.arange(36).reshape(3, 3, 4).astype("float32")

    def test_index_select_all_dims(self):
        x = self.x
        idx = np.array([0, 2], dtype="int64")
        refs = {0: x[idx], 1: x[:, idx], 2: x[:, :, idx]}
        def body(dev):
            for d in (0, 1, 2):
                self.ac(torch.index_select(t(x), d, t(idx)).numpy(), refs[d],
                        msg=f"index_select func d{d} {dev}")
                self.ac(t(x).index_select(d, t(idx)).numpy(), refs[d],
                        msg=f"index_select method d{d} {dev}")
        both_devices(body)

    def test_index_select_repeats_allowed(self):
        x = np.arange(10).astype("float32")
        idx = np.array([3, 3, 0, 9, 9], dtype="int64")
        def body(dev):
            self.ac(torch.index_select(t(x), 0, t(idx)).numpy(), x[idx],
                    msg=f"index_select repeats {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# index_add / index_add_   (DUP-ACCUMULATE -- the torch contract)
# REGRESSION: the in-place index_add_ used `x[adv_idx] += t` (read-add-write) which is
# last-write-wins and DROPS earlier contributions at duplicate indices. Fixed in misc.py
# to route through the dup-correct scatter_add path.
# ---------------------------------------------------------------------------
class TestIndexAdd(Base):
    def test_index_add_out_of_place_dim0(self):
        base = np.ones((5, 3), dtype="float32")
        index = np.array([0, 4, 2], dtype="int64")
        source = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        ref = np_index_add(base, 0, index, source)
        def body(dev):
            self.ac(t(base).index_add(0, t(index), t(source)).numpy(), ref,
                    msg=f"index_add d0 {dev}")
        both_devices(body)

    def test_index_add_duplicate_indices_accumulate(self):
        # the bug magnet: index=[0,0,1] must add BOTH src rows 0,1 into out row 0.
        base = np.zeros((3, 2), dtype="float32")
        index = np.array([0, 0, 1], dtype="int64")
        source = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype="float32")
        ref = np_index_add(base, 0, index, source)        # row0=[3,3], row1=[3,3], row2=0
        self.ae(ref, np.array([[3., 3.], [3., 3.], [0., 0.]]))
        def body(dev):
            self.ac(t(base).index_add(0, t(index), t(source)).numpy(), ref,
                    msg=f"index_add dup out-of-place {dev}")
        both_devices(body)

    def test_index_add__in_place_duplicate_accumulate(self):
        base = np.zeros((3,), dtype="float32")
        index = np.array([0, 0, 1], dtype="int64")
        source = np.array([1., 2., 3.], dtype="float32")
        ref = np_index_add(base.reshape(3, 1), 0, index, source.reshape(3, 1)).reshape(3)
        self.ae(ref, np.array([3., 3., 0.]))
        def body(dev):
            x = t(base)
            x.index_add_(0, t(index), t(source))
            self.ac(x.numpy(), ref, msg=f"index_add_ in-place dup {dev}")
        both_devices(body)

    def test_index_add_alpha(self):
        base = np.ones((3, 2), dtype="float32")
        index = np.array([0, 0], dtype="int64")
        source = np.array([[1., 1.], [1., 1.]], dtype="float32")
        ref = np_index_add(base, 0, index, source, alpha=2.0)   # row0 = 1 + 2 + 2 = 5
        def body(dev):
            self.ac(t(base).index_add(0, t(index), t(source), alpha=2.0).numpy(), ref,
                    msg=f"index_add alpha {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# index_copy_   (overwrite along dim; distinct dst only)
# ---------------------------------------------------------------------------
class TestIndexCopy(Base):
    def test_index_copy__dim0(self):
        base = np.zeros((4, 3), dtype="float32")
        index = np.array([3, 1], dtype="int64")
        source = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")
        ref = base.copy(); ref[3] = source[0]; ref[1] = source[1]
        def body(dev):
            x = t(base)
            x.index_copy_(0, t(index), t(source))
            self.ac(x.numpy(), ref, msg=f"index_copy_ d0 {dev}")
        both_devices(body)

    def test_index_copy__dim1(self):
        base = np.zeros((2, 4), dtype="float32")
        index = np.array([0, 2], dtype="int64")
        source = np.array([[1, 2], [3, 4]], dtype="float32")
        ref = base.copy(); ref[:, 0] = source[:, 0]; ref[:, 2] = source[:, 1]
        def body(dev):
            x = t(base)
            x.index_copy_(1, t(index), t(source))
            self.ac(x.numpy(), ref, msg=f"index_copy_ d1 {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# index_fill / index_fill_
# ---------------------------------------------------------------------------
class TestIndexFill(Base):
    def test_index_fill_dim1(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        index = np.array([0, 2], dtype="int64")
        ref = x.copy(); ref[:, [0, 2]] = -5.0
        def body(dev):
            self.ac(t(x).index_fill(1, t(index), -5.0).numpy(), ref,
                    msg=f"index_fill d1 {dev}")
        both_devices(body)

    def test_index_fill__in_place(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        index = np.array([1], dtype="int64")
        ref = x.copy(); ref[1, :] = 9.0
        def body(dev):
            v = t(x)
            v.index_fill_(0, t(index), 9.0)
            self.ac(v.numpy(), ref, msg=f"index_fill_ d0 {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# masked_fill / masked_select
# ---------------------------------------------------------------------------
class TestMasked(Base):
    def test_masked_fill(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        m = (x % 2 == 0)
        ref = np.where(m, -1.0, x)
        def body(dev):
            self.ac(t(x).masked_fill(t(m), -1.0).numpy(), ref, msg=f"masked_fill {dev}")
        both_devices(body)

    def test_masked_fill_broadcast_mask(self):
        # torch broadcasts a lower-rank mask against x (row mask over (3,4)).
        x = np.arange(12).reshape(3, 4).astype("float32")
        m = np.array([True, False, True]).reshape(3, 1)
        ref = np.where(m, 99.0, x)
        def body(dev):
            self.ac(t(x).masked_fill(t(m), 99.0).numpy(), ref,
                    msg=f"masked_fill bcast {dev}")
        both_devices(body)

    def test_masked_select(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        m = (x % 3 == 0)
        def body(dev):
            self.ac(t(x).masked_select(t(m)).numpy(), x[m], msg=f"masked_select {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# take / take_along_dim
# ---------------------------------------------------------------------------
class TestTake(Base):
    def test_take_flat(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        idx = np.array([0, 5, 11, 2], dtype="int64")
        def body(dev):
            self.ac(torch.take(t(x), t(idx)).numpy(), np.take(x, idx),
                    msg=f"take flat {dev}")
        both_devices(body)

    def test_take_along_dim_argsort(self):
        a = np.random.RandomState(2).randn(3, 4).astype("float32")
        ti = np.argsort(a, axis=1).astype("int64")
        def body(dev):
            self.ac(torch.take_along_dim(t(a), t(ti), dim=1).numpy(),
                    np.take_along_axis(a, ti, axis=1), msg=f"take_along_dim d1 {dev}")
        both_devices(body)

    def test_take_along_dim_broadcast(self):
        # beam-search shape: input (2,3,4), index (2,1,1) -> torch broadcasts the
        # non-gather dims to input, gathering whole rows -> (2,1,4).
        inp = np.arange(24).reshape(2, 3, 4).astype("float32")
        ti = np.array([[[1]], [[2]]], dtype="int64")
        ref = np.take_along_axis(inp, np.broadcast_to(ti, (2, 1, 4)), axis=1)
        def body(dev):
            g = torch.take_along_dim(t(inp), t(ti), dim=1).numpy()
            self.ac(g, ref, msg=f"take_along_dim broadcast {dev}")
        both_devices(body)


# ---------------------------------------------------------------------------
# one_hot  (F.one_hot, torch.nn.functional level)
# ---------------------------------------------------------------------------
class TestOneHot(Base):
    def test_one_hot_explicit_classes(self):
        x = np.array([0, 2, 1, 2], dtype="int64")
        ref = np.eye(3, dtype="int32")[x]
        def body(dev):
            self.ae(F.one_hot(t(x), 3).numpy(), ref, msg=f"one_hot k=3 {dev}")
        both_devices(body)

    def test_one_hot_infer_classes(self):
        x = np.array([0, 1, 2, 1], dtype="int64")     # max 2 -> 3 classes
        ref = np.eye(3, dtype="int32")[x]
        def body(dev):
            self.ae(F.one_hot(t(x)).numpy(), ref, msg=f"one_hot infer {dev}")
        both_devices(body)

    def test_one_hot_2d_input(self):
        x = (np.arange(6).reshape(3, 2) % 3).astype("int64")
        ref = np.eye(3, dtype="int32")[x]
        def body(dev):
            self.ae(F.one_hot(t(x), 3).numpy(), ref, msg=f"one_hot 2d {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
