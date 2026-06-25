"""Torch-grade indexing/slicing-semantics regression tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (round 2). Like ``test_torch_compat_ops.py``
this is a structured ``unittest`` module: every check compares jittor-as-torch against an
INDEPENDENT numpy reference and runs on BOTH CPU and CUDA (when the build has it), so it
locks torch/numpy *indexing semantics* rather than jittor self-consistency.

Covered: basic slicing (incl. negative step ``x[::-1]``, ``x[1:3, ::2]``), integer-tensor
fancy indexing (``x[idx]`` / ``x[i0, i1]`` / negatives / slice+fancy mix), boolean mask
indexing (``x[x > 0]`` incl. lower-rank masks), ``index_select``, ``masked_select``,
``take`` / ``take_along_dim``, ``gather`` / ``scatter`` (incl. ``reduce='add'``), Ellipsis,
and ``None`` / newaxis.

Run:  python -m jittor.test.test_torch_compat_indexing
      python -m pytest python/jittor/test/test_torch_compat_indexing.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def t(a):
    """np array -> jittor Var (keep dtype, including int64/bool index tensors)."""
    return torch.array(a)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_array_equal(g, r, err_msg=msg)


class TestBasicSlicing(Base):
    def setUp(self):
        self.x1 = np.arange(10).astype("float32")
        self.x2 = np.arange(20).reshape(4, 5).astype("float32")
        self.x3 = np.arange(60).reshape(3, 4, 5).astype("float32")

    def test_simple_slices(self):
        x = self.x2
        def body(dev):
            tx = t(x)
            self.ac(tx[1:3].numpy(), x[1:3], msg=f"x[1:3] {dev}")
            self.ac(tx[:, 1:4].numpy(), x[:, 1:4], msg=f"x[:,1:4] {dev}")
            self.ac(tx[1:3, 2:].numpy(), x[1:3, 2:], msg=f"x[1:3,2:] {dev}")
            self.ac(tx[::2].numpy(), x[::2], msg=f"x[::2] {dev}")
            self.ac(tx[1:, ::2].numpy(), x[1:, ::2], msg=f"x[1:,::2] {dev}")
        both_devices(body)

    def test_negative_index_and_slice(self):
        x = self.x2
        def body(dev):
            tx = t(x)
            self.ac(tx[-1].numpy(), x[-1], msg=f"x[-1] {dev}")
            self.ac(tx[-2:].numpy(), x[-2:], msg=f"x[-2:] {dev}")
            self.ac(tx[:, -2:].numpy(), x[:, -2:], msg=f"x[:,-2:] {dev}")
            self.ac(tx[:-1, :-1].numpy(), x[:-1, :-1], msg=f"x[:-1,:-1] {dev}")
        both_devices(body)

    def test_negative_step_reverse(self):
        x1, x2 = self.x1, self.x2
        def body(dev):
            self.ac(t(x1)[::-1].numpy(), x1[::-1], msg=f"x[::-1] 1d {dev}")
            self.ac(t(x2)[::-1].numpy(), x2[::-1], msg=f"x[::-1] 2d {dev}")
            self.ac(t(x2)[1:3, ::2].numpy(), x2[1:3, ::2], msg=f"x[1:3,::2] {dev}")
            self.ac(t(x2)[:, ::-1].numpy(), x2[:, ::-1], msg=f"x[:,::-1] {dev}")
            self.ac(t(x2)[::-1, ::-1].numpy(), x2[::-1, ::-1], msg=f"x[::-1,::-1] {dev}")
            self.ac(t(x1)[8:2:-2].numpy(), x1[8:2:-2], msg=f"x[8:2:-2] {dev}")
        both_devices(body)

    def test_int_index_reduces_dim(self):
        x = self.x3
        def body(dev):
            tx = t(x)
            self.ac(tx[1].numpy(), x[1], msg=f"x[1] {dev}")
            self.ac(tx[1, 2].numpy(), x[1, 2], msg=f"x[1,2] {dev}")
            self.ac(tx[:, 0].numpy(), x[:, 0], msg=f"x[:,0] {dev}")
            self.ac(tx[:, :, -1].numpy(), x[:, :, -1], msg=f"x[:,:,-1] {dev}")
        both_devices(body)


class TestFancyIndexing(Base):
    def setUp(self):
        self.x = np.arange(20).reshape(4, 5).astype("float32")

    def test_int_tensor_rows(self):
        x = self.x
        idx = np.array([0, 2, 3], dtype="int64")
        def body(dev):
            self.ac(t(x)[t(idx)].numpy(), x[idx], msg=f"x[idx] {dev}")
        both_devices(body)

    def test_int_tensor_negative(self):
        x = self.x
        idx = np.array([-1, -2, 0, -1], dtype="int64")
        def body(dev):
            self.ac(t(x)[t(idx)].numpy(), x[idx], msg=f"x[neg idx] {dev}")
        both_devices(body)

    def test_two_index_arrays_pairs(self):
        # advanced indexing: x[i0, i1] gathers element pairs (i0[k], i1[k]).
        x = self.x
        i0 = np.array([0, 1, 2, 3], dtype="int64")
        i1 = np.array([1, 3, 4, 0], dtype="int64")
        def body(dev):
            self.ac(t(x)[t(i0), t(i1)].numpy(), x[i0, i1], msg=f"x[i0,i1] {dev}")
        both_devices(body)

    def test_slice_plus_fancy(self):
        x = self.x
        idx = np.array([0, 2], dtype="int64")
        def body(dev):
            self.ac(t(x)[1:3, t(idx)].numpy(), x[1:3, idx], msg=f"x[1:3,idx] {dev}")
        both_devices(body)


class TestBooleanMask(Base):
    def test_mask_same_rank(self):
        x = np.arange(20).reshape(4, 5).astype("float32")
        def body(dev):
            tx = t(x)
            self.ac(tx[tx > 10].numpy(), x[x > 10], msg=f"x[x>10] {dev}")
            self.ac(tx[(tx % 2) == 0].numpy(), x[(x % 2) == 0], msg=f"x[even] {dev}")
        both_devices(body)

    def test_mask_lower_rank(self):
        # mask of shape (2,3) applied to (2,3,4) selects whole rows along last dim.
        y = np.arange(24).reshape(2, 3, 4).astype("float32")
        m = (y.sum(2) > 20)
        def body(dev):
            self.ac(t(y)[t(m)].numpy(), y[m], msg=f"y[mask2d] {dev}")
        both_devices(body)


class TestIndexSelect(Base):
    def test_index_select_dim0(self):
        x = np.random.RandomState(13).randn(5, 4).astype("float32")
        idx = np.array([0, 2, 4], dtype="int64")
        def body(dev):
            g = torch.index_select(t(x), 0, t(idx)).numpy()
            self.ac(g, x[idx], msg=f"index_select d0 {dev}")
        both_devices(body)

    def test_index_select_dim_gt0(self):
        # regression: index_select for dim>0 used input[(None,)*dim+(indices,)]
        # (newaxis) instead of slice(None) -> wrong shape/values. Fixed in misc.py.
        x = np.arange(24).reshape(2, 3, 4).astype("float32")
        idx = np.array([0, 2], dtype="int64")
        def body(dev):
            self.ac(torch.index_select(t(x), 1, t(idx)).numpy(), x[:, idx],
                    msg=f"index_select d1 {dev}")
            self.ac(torch.index_select(t(x), 2, t(idx)).numpy(), x[:, :, idx],
                    msg=f"index_select d2 {dev}")
        both_devices(body)


class TestMaskedSelect(Base):
    def test_masked_select(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        m = (x % 2 == 0)
        def body(dev):
            # tensor-method form (torch.masked_select is not exposed at module level).
            self.ac(t(x).masked_select(t(m)).numpy(), x[m], msg=f"masked_select {dev}")
        both_devices(body)


class TestTake(Base):
    def test_take_flat(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        idx = np.array([0, 5, 11, 2], dtype="int64")
        def body(dev):
            self.ac(torch.take(t(x), t(idx)).numpy(), np.take(x, idx),
                    msg=f"take flat {dev}")
        both_devices(body)

    def test_take_preserves_index_shape(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        idx2 = np.array([[0, 5], [11, 2]], dtype="int64")
        def body(dev):
            self.ac(torch.take(t(x), t(idx2)).numpy(), np.take(x, idx2),
                    msg=f"take 2d-idx {dev}")
        both_devices(body)

    def test_take_along_dim(self):
        a = np.random.RandomState(2).randn(3, 4).astype("float32")
        ti = np.argsort(a, axis=1).astype("int64")
        def body(dev):
            self.ac(torch.take_along_dim(t(a), t(ti), dim=1).numpy(),
                    np.take_along_axis(a, ti, axis=1), msg=f"take_along_dim d1 {dev}")
        both_devices(body)

    def test_take_along_dim_dim0(self):
        a = np.random.RandomState(3).randn(4, 3).astype("float32")
        ti = np.argsort(a, axis=0).astype("int64")
        def body(dev):
            self.ac(torch.take_along_dim(t(a), t(ti), dim=0).numpy(),
                    np.take_along_axis(a, ti, axis=0), msg=f"take_along_dim d0 {dev}")
        both_devices(body)


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


class TestScatter(Base):
    def _np_scatter(self, base, dim, idx, src, reduce=None):
        out = base.copy()
        it = np.nditer(idx, flags=["multi_index"])
        for v in it:
            mi = list(it.multi_index)
            dst = list(mi)
            dst[dim] = int(v)
            if reduce == "add":
                out[tuple(dst)] += src[tuple(mi)]
            elif reduce == "multiply":
                out[tuple(dst)] *= src[tuple(mi)]
            else:
                out[tuple(dst)] = src[tuple(mi)]
        return out

    def test_scatter_overwrite_dim0(self):
        # distinct destinations only (torch leaves duplicate-write order unspecified).
        base = np.zeros((3, 5), dtype="float32")
        idx = np.array([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype="int64")
        src = np.arange(1, 11).reshape(2, 5).astype("float32")
        ref = self._np_scatter(base, 0, idx, src)
        def body(dev):
            self.ac(torch.scatter(t(base), 0, t(idx), t(src)).numpy(), ref,
                    msg=f"scatter overwrite d0 {dev}")
        both_devices(body)

    def test_scatter_reduce_add(self):
        base = np.zeros((3, 4), dtype="float32")
        idx = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype="int64")  # heavy collision
        src = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype="float32")
        ref = self._np_scatter(base, 0, idx, src, reduce="add")
        def body(dev):
            self.ac(torch.scatter(t(base), 0, t(idx), t(src), reduce="add").numpy(),
                    ref, msg=f"scatter reduce=add {dev}")
        both_devices(body)


class TestEllipsisNewaxis(Base):
    def setUp(self):
        self.y = np.arange(24).reshape(2, 3, 4).astype("float32")

    def test_ellipsis(self):
        y = self.y
        def body(dev):
            ty = t(y)
            self.ac(ty[..., 0].numpy(), y[..., 0], msg=f"y[...,0] {dev}")
            self.ac(ty[0, ...].numpy(), y[0, ...], msg=f"y[0,...] {dev}")
            self.ac(ty[..., 1:3].numpy(), y[..., 1:3], msg=f"y[...,1:3] {dev}")
            self.ac(ty[1, ..., 2].numpy(), y[1, ..., 2], msg=f"y[1,...,2] {dev}")
        both_devices(body)

    def test_newaxis_none(self):
        y = self.y
        def body(dev):
            ty = t(y)
            self.ac(ty[None].numpy(), y[None], msg=f"y[None] {dev}")
            self.ac(ty[:, None].numpy(), y[:, None], msg=f"y[:,None] {dev}")
            self.ac(ty[..., None].numpy(), y[..., None], msg=f"y[...,None] {dev}")
            self.ac(ty[:, None, :, None].numpy(), y[:, None, :, None],
                    msg=f"y[:,None,:,None] {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
