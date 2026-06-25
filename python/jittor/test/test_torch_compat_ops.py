"""Torch-grade op-semantics regression tests for the ``import jittor as torch`` layer.

Part of the torch-grade test-suite rewrite. Unlike the legacy ``test_torch_compat.py``
(a flat ``ok()`` script), this is a structured ``unittest`` module: every check compares
jittor-as-torch against an INDEPENDENT numpy reference, and runs on BOTH CPU and CUDA
(when available), so it locks torch-API *semantics* rather than jittor self-consistency.

Run:  python -m jittor.test.test_torch_compat_ops
      python -m pytest python/jittor/test/test_torch_compat_ops.py
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


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref), err_msg=msg)


class TestReductions(Base):
    def setUp(self):
        self.x = np.random.RandomState(0).randn(3, 4, 5).astype("float32")

    def test_sum_dim_keepdim(self):
        x = self.x
        for dim in [0, 1, 2, -1]:
            for keep in [False, True]:
                def body(dev, dim=dim, keep=keep):
                    g = torch.sum(torch.tensor(x), dim=dim, keepdim=keep).numpy()
                    self.ac(g, x.sum(axis=dim, keepdims=keep), atol=1e-4,
                            msg=f"sum dim={dim} keep={keep} {dev}")
                both_devices(body)

    def test_mean_dim(self):
        x = self.x
        for dim in [0, -1]:
            def body(dev, dim=dim):
                g = torch.mean(torch.tensor(x), dim=dim).numpy()
                self.ac(g, x.mean(axis=dim), atol=1e-5, msg=f"mean dim={dim} {dev}")
            both_devices(body)

    def test_max_min_dim_namedtuple(self):
        x = self.x
        def body(dev):
            mx = torch.max(torch.tensor(x), dim=1)
            self.ac(mx.values.numpy(), x.max(axis=1), msg=f"max.values {dev}")
            self.ae(mx.indices.numpy(), x.argmax(axis=1), msg=f"max.indices {dev}")
            mn = torch.min(torch.tensor(x), dim=2)
            self.ac(mn.values.numpy(), x.min(axis=2), msg=f"min.values {dev}")
        both_devices(body)

    def test_argmax_argmin(self):
        x = self.x
        def body(dev):
            self.ae(torch.argmax(torch.tensor(x), dim=-1).numpy(), x.argmax(axis=-1),
                    msg=f"argmax {dev}")
            self.ae(torch.argmin(torch.tensor(x), dim=0).numpy(), x.argmin(axis=0),
                    msg=f"argmin {dev}")
        both_devices(body)

    def test_argmax_argmin_all_dims(self):
        # regression: argmax/argmin with negative dims other than -1 used to crash
        # in arg_reduce's internal transpose (cutt_transpose "axes != xdim").
        x = self.x
        def body(dev):
            for d in [0, 1, 2, -1, -2, -3]:
                self.ae(torch.argmax(torch.tensor(x), dim=d).numpy(), x.argmax(axis=d),
                        msg=f"argmax dim={d} {dev}")
                self.ae(torch.argmin(torch.tensor(x), dim=d).numpy(), x.argmin(axis=d),
                        msg=f"argmin dim={d} {dev}")
        both_devices(body)

    def test_argmax_out_of_range_raises(self):
        x = self.x
        with self.assertRaises(IndexError):
            torch.argmax(torch.tensor(x), dim=7)

    def test_prod_dim(self):
        x = np.random.RandomState(1).rand(2, 3, 4).astype("float32") + 0.5
        def body(dev):
            self.ac(torch.prod(torch.tensor(x), dim=1).numpy(), x.prod(axis=1),
                    rtol=1e-4, msg=f"prod dim {dev}")
        both_devices(body)


class TestShapeOps(Base):
    def setUp(self):
        self.x = np.random.RandomState(2).randn(2, 3, 4).astype("float32")

    def test_permute_transpose(self):
        x = self.x
        def body(dev):
            self.ac(torch.permute(torch.tensor(x), (2, 0, 1)).numpy(),
                    np.transpose(x, (2, 0, 1)), msg=f"permute {dev}")
            self.ac(torch.transpose(torch.tensor(x), 0, 2).numpy(),
                    np.swapaxes(x, 0, 2), msg=f"transpose {dev}")
        both_devices(body)

    def test_reshape_flatten(self):
        x = self.x
        def body(dev):
            self.ac(torch.reshape(torch.tensor(x), (6, 4)).numpy(), x.reshape(6, 4),
                    msg=f"reshape {dev}")
            self.ac(torch.flatten(torch.tensor(x)).numpy(), x.reshape(-1),
                    msg=f"flatten {dev}")
            self.ac(torch.flatten(torch.tensor(x), 1).numpy(), x.reshape(2, -1),
                    msg=f"flatten start_dim {dev}")
        both_devices(body)

    def test_squeeze_unsqueeze(self):
        x = np.random.RandomState(3).randn(2, 1, 3).astype("float32")
        def body(dev):
            self.ac(torch.squeeze(torch.tensor(x)).numpy(), x.squeeze(),
                    msg=f"squeeze {dev}")
            self.ac(torch.unsqueeze(torch.tensor(x), 0).numpy(), x[None],
                    msg=f"unsqueeze {dev}")
        both_devices(body)

    def test_cat_stack(self):
        a = np.random.RandomState(4).randn(2, 3).astype("float32")
        b = np.random.RandomState(5).randn(2, 3).astype("float32")
        def body(dev):
            ta, tb = torch.tensor(a), torch.tensor(b)
            self.ac(torch.cat([ta, tb], dim=0).numpy(), np.concatenate([a, b], 0),
                    msg=f"cat0 {dev}")
            self.ac(torch.cat([ta, tb], dim=1).numpy(), np.concatenate([a, b], 1),
                    msg=f"cat1 {dev}")
            self.ac(torch.stack([ta, tb], dim=0).numpy(), np.stack([a, b], 0),
                    msg=f"stack0 {dev}")
            self.ac(torch.stack([ta, tb], dim=-1).numpy(), np.stack([a, b], -1),
                    msg=f"stack-1 {dev}")
        both_devices(body)


class TestComparisonWhere(Base):
    def test_where(self):
        a = np.random.RandomState(6).randn(4, 5).astype("float32")
        b = np.random.RandomState(7).randn(4, 5).astype("float32")
        def body(dev):
            ta, tb = torch.tensor(a), torch.tensor(b)
            self.ac(torch.where(ta > tb, ta, tb).numpy(), np.where(a > b, a, b),
                    msg=f"where {dev}")
        both_devices(body)

    def test_comparisons(self):
        a = np.random.RandomState(8).randn(3, 3).astype("float32")
        b = np.random.RandomState(9).randn(3, 3).astype("float32")
        def body(dev):
            ta, tb = torch.tensor(a), torch.tensor(b)
            self.ae((ta > tb).numpy(), a > b, msg=f"gt {dev}")
            self.ae((ta <= tb).numpy(), a <= b, msg=f"le {dev}")
            self.ac(torch.maximum(ta, tb).numpy(), np.maximum(a, b), msg=f"maximum {dev}")
            self.ac(torch.minimum(ta, tb).numpy(), np.minimum(a, b), msg=f"minimum {dev}")
        both_devices(body)


class TestCumulative(Base):
    def test_cumsum(self):
        x = np.random.RandomState(10).randn(3, 4).astype("float32")
        def body(dev):
            self.ac(torch.cumsum(torch.tensor(x), dim=1).numpy(), np.cumsum(x, axis=1),
                    rtol=1e-5, atol=1e-5, msg=f"cumsum {dev}")
        both_devices(body)

    def test_cumprod(self):
        x = np.random.RandomState(11).rand(3, 4).astype("float32") + 0.5
        def body(dev):
            self.ac(torch.cumprod(torch.tensor(x), dim=1).numpy(), np.cumprod(x, axis=1),
                    rtol=1e-4, msg=f"cumprod {dev}")
        both_devices(body)


class TestGather(Base):
    def test_gather(self):
        x = np.random.RandomState(12).randn(3, 4).astype("float32")
        idx = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1]], dtype="int64")
        def body(dev):
            g = torch.gather(torch.tensor(x), 1, torch.tensor(idx)).numpy()
            self.ac(g, np.take_along_axis(x, idx, axis=1), msg=f"gather {dev}")
        both_devices(body)

    def test_index_select(self):
        x = np.random.RandomState(13).randn(5, 4).astype("float32")
        idx = np.array([0, 2, 4], dtype="int64")
        def body(dev):
            g = torch.index_select(torch.tensor(x), 0, torch.tensor(idx)).numpy()
            self.ac(g, x[idx], msg=f"index_select {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
