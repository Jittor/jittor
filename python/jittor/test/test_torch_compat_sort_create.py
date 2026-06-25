"""Torch-grade sort / search / creation op parity for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite. CPU+CUDA vs numpy references. Covers the
sort/select and tensor-construction surface not exercised by the other modules.

Run:  python -m jittor.test.test_torch_compat_sort_create
"""
import unittest
import numpy as np
import jittor as torch
import jittor as jt

_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref), err_msg=msg)


class TestSortSelect(Base):
    def test_sort(self):
        x = np.random.RandomState(0).randn(4, 5).astype("float32")
        def body(dev):
            r = torch.sort(torch.tensor(x), dim=-1)
            self.ac(r.values.numpy(), np.sort(x, axis=-1), msg=f"sort values {dev}")
            r2 = torch.sort(torch.tensor(x), dim=-1, descending=True)
            self.ac(r2.values.numpy(), np.sort(x, axis=-1)[:, ::-1], msg=f"sort desc {dev}")
        both_devices(body)

    def test_argsort(self):
        x = np.random.RandomState(1).randn(3, 6).astype("float32")
        def body(dev):
            # argsort indices must reproduce the sorted order
            idx = np.asarray(torch.argsort(torch.tensor(x), dim=-1).numpy())
            self.ac(np.take_along_axis(x, idx, axis=-1), np.sort(x, axis=-1),
                    msg=f"argsort {dev}")
        both_devices(body)

    def test_topk(self):
        x = np.random.RandomState(2).randn(4, 7).astype("float32")
        def body(dev):
            r = torch.topk(torch.tensor(x), 3, dim=-1)
            self.ac(r.values.numpy(), np.sort(x, axis=-1)[:, ::-1][:, :3],
                    msg=f"topk values {dev}")
            rs = torch.topk(torch.tensor(x), 2, dim=-1, largest=False)
            self.ac(rs.values.numpy(), np.sort(x, axis=-1)[:, :2],
                    msg=f"topk smallest {dev}")
        both_devices(body)

    def test_max_min_with_dim(self):
        x = np.random.RandomState(3).randn(3, 4).astype("float32")
        def body(dev):
            self.ac(torch.max(torch.tensor(x), dim=1).values.numpy(), x.max(1),
                    msg=f"max dim {dev}")
            self.ae(torch.argmax(torch.tensor(x), dim=1).numpy(), x.argmax(1),
                    msg=f"argmax dim {dev}")
        both_devices(body)


class TestSearch(Base):
    def test_nonzero(self):
        x = (np.random.RandomState(4).randn(5) > 0).astype("float32")
        def body(dev):
            nz = np.asarray(torch.nonzero(torch.tensor(x)).numpy()).reshape(-1)
            self.ae(np.sort(nz), np.nonzero(x)[0], msg=f"nonzero {dev}")
        both_devices(body)

    def test_searchsorted(self):
        seq = np.array([1., 3., 5., 7., 9.], dtype="float32")
        vals = np.array([0., 4., 6., 10.], dtype="float32")
        def body(dev):
            r = torch.searchsorted(torch.tensor(seq), torch.tensor(vals)).numpy()
            self.ae(np.asarray(r), np.searchsorted(seq, vals), msg=f"searchsorted {dev}")
        both_devices(body)


class TestCreation(Base):
    def test_arange_linspace(self):
        def body(dev):
            self.ac(torch.arange(0, 10, 2).numpy(), np.arange(0, 10, 2),
                    msg=f"arange {dev}")
            self.ac(torch.linspace(0, 1, 5).numpy(), np.linspace(0, 1, 5),
                    atol=1e-6, msg=f"linspace {dev}")
        both_devices(body)

    def test_eye_diag_tri(self):
        def body(dev):
            self.ac(torch.eye(4).numpy(), np.eye(4), msg=f"eye {dev}")
            x = np.random.RandomState(5).randn(4, 4).astype("float32")
            self.ac(torch.triu(torch.tensor(x)).numpy(), np.triu(x), msg=f"triu {dev}")
            self.ac(torch.tril(torch.tensor(x), -1).numpy(), np.tril(x, -1),
                    msg=f"tril diag {dev}")
            v = np.array([1., 2., 3.], dtype="float32")
            self.ac(torch.diag(torch.tensor(v)).numpy(), np.diag(v), msg=f"diag vec {dev}")
        both_devices(body)

    def test_meshgrid(self):
        a = np.array([1., 2., 3.], dtype="float32")
        b = np.array([4., 5.], dtype="float32")
        def body(dev):
            gx, gy = torch.meshgrid(torch.tensor(a), torch.tensor(b), indexing="ij")
            rx, ry = np.meshgrid(a, b, indexing="ij")
            self.ac(gx.numpy(), rx, msg=f"meshgrid x {dev}")
            self.ac(gy.numpy(), ry, msg=f"meshgrid y {dev}")
        both_devices(body)

    def test_like_constructors(self):
        x = np.random.RandomState(6).randn(2, 3).astype("float32")
        def body(dev):
            self.ac(torch.zeros_like(torch.tensor(x)).numpy(), np.zeros_like(x),
                    msg=f"zeros_like {dev}")
            self.ac(torch.ones_like(torch.tensor(x)).numpy(), np.ones_like(x),
                    msg=f"ones_like {dev}")
            self.ac(torch.full((2, 3), 1.5).numpy(), np.full((2, 3), 1.5, "float32"),
                    msg=f"full {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
