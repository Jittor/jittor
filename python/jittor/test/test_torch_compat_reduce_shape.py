"""Torch-grade reduction + shape-boundary regression tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (round 2). Like ``test_torch_compat_ops.py``
this is a structured ``unittest`` module: every check compares jittor-as-torch against an
INDEPENDENT numpy reference and runs on BOTH CPU and CUDA (when the build has it).

Covered reductions: ``var`` / ``std`` (unbiased default + ``correction=`` / ``unbiased=``),
``norm`` (p = 1, 2, inf, -inf, 'fro'), ``logsumexp``, ``all`` / ``any``,
``count_nonzero``, ``cumsum`` / ``cumprod`` / ``cummax``.
Covered shape ops: ``repeat`` / ``repeat_interleave`` / ``tile``, ``roll``, ``flip``,
``unfold``, ``split`` / ``chunk`` / ``tensor_split``, ``movedim`` / ``moveaxis``,
``broadcast_to`` / ``expand``.

jittor has no 0-d scalars (a "scalar" is shape ``(1,)``); for full reductions we compare
values via ``.item()`` rather than asserting numpy's ``()`` shape.

Run:  python -m jittor.test.test_torch_compat_reduce_shape
      python -m pytest python/jittor/test/test_torch_compat_reduce_shape.py
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
    return torch.array(a)


def np_logsumexp(a, axis, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    r = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return r if keepdims else np.squeeze(r, axis=axis)


def torch_chunk_sizes(n, chunks):
    """Replicate torch.chunk sizing: ceil(n/chunks) per piece, last one shorter."""
    sz = -(-n // chunks)
    out, rem = [], n
    while rem > 0:
        out.append(min(sz, rem)); rem -= sz
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


# ----------------------------------------------------------------------------- reductions

class TestVarStd(Base):
    def setUp(self):
        self.x = np.random.RandomState(0).randn(4, 5).astype("float32")

    def test_var_unbiased_default(self):
        # torch default is unbiased (correction=1 / ddof=1).
        x = self.x
        for dim in [0, 1, -1]:
            def body(dev, dim=dim):
                self.ac(torch.var(t(x), dim=dim).numpy(), x.var(axis=dim, ddof=1),
                        rtol=1e-5, atol=1e-6, msg=f"var ddof1 dim={dim} {dev}")
            both_devices(body)

    def test_var_correction0(self):
        x = self.x
        def body(dev):
            self.ac(torch.var(t(x), dim=1, correction=0).numpy(), x.var(axis=1, ddof=0),
                    rtol=1e-5, atol=1e-6, msg=f"var correction=0 {dev}")
            self.ac(torch.var(t(x), dim=1, unbiased=False).numpy(), x.var(axis=1, ddof=0),
                    rtol=1e-5, atol=1e-6, msg=f"var unbiased=False {dev}")
        both_devices(body)

    def test_var_keepdim(self):
        x = self.x
        def body(dev):
            self.ac(torch.var(t(x), dim=1, keepdim=True).numpy(),
                    x.var(axis=1, ddof=1, keepdims=True), rtol=1e-5, atol=1e-6,
                    msg=f"var keepdim {dev}")
        both_devices(body)

    def test_var_full(self):
        x = self.x
        def body(dev):
            self.assertAlmostEqual(float(torch.var(t(x)).item()), float(x.var(ddof=1)),
                                   places=4, msg=f"var full {dev}")
        both_devices(body)

    def test_std(self):
        x = self.x
        def body(dev):
            self.ac(torch.std(t(x), dim=1).numpy(), x.std(axis=1, ddof=1),
                    rtol=1e-5, atol=1e-5, msg=f"std ddof1 {dev}")
            self.ac(torch.std(t(x), dim=1, correction=0).numpy(), x.std(axis=1, ddof=0),
                    rtol=1e-5, atol=1e-5, msg=f"std correction=0 {dev}")
        both_devices(body)


class TestNorm(Base):
    def setUp(self):
        self.x = np.random.RandomState(1).randn(3, 4).astype("float32")

    def test_norm_full_fro(self):
        x = self.x
        def body(dev):
            self.assertAlmostEqual(float(torch.norm(t(x)).item()),
                                   float(np.linalg.norm(x)), places=4,
                                   msg=f"norm fro full {dev}")
            self.assertAlmostEqual(float(torch.norm(t(x), p="fro").item()),
                                   float(np.linalg.norm(x)), places=4,
                                   msg=f"norm 'fro' full {dev}")
        both_devices(body)

    def test_norm_p_dim(self):
        x = self.x
        def body(dev):
            self.ac(torch.norm(t(x), p=1, dim=1).numpy(), np.abs(x).sum(1),
                    rtol=1e-5, atol=1e-5, msg=f"norm p1 d1 {dev}")
            self.ac(torch.norm(t(x), p=2, dim=1).numpy(), np.sqrt((x ** 2).sum(1)),
                    rtol=1e-5, atol=1e-5, msg=f"norm p2 d1 {dev}")
            self.ac(torch.norm(t(x), p=float("inf"), dim=1).numpy(), np.abs(x).max(1),
                    rtol=1e-5, atol=1e-5, msg=f"norm inf d1 {dev}")
            self.ac(torch.norm(t(x), p=float("-inf"), dim=1).numpy(), np.abs(x).min(1),
                    rtol=1e-5, atol=1e-5, msg=f"norm -inf d1 {dev}")
        both_devices(body)

    def test_norm_keepdim(self):
        x = self.x
        def body(dev):
            self.ac(torch.norm(t(x), p=2, dim=1, keepdim=True).numpy(),
                    np.sqrt((x ** 2).sum(1, keepdims=True)), rtol=1e-5, atol=1e-5,
                    msg=f"norm keepdim {dev}")
        both_devices(body)


class TestLogsumexp(Base):
    def setUp(self):
        self.x = np.random.RandomState(1).randn(3, 4).astype("float32")

    def test_logsumexp_dim(self):
        x = self.x
        for dim in [0, 1, -1]:
            def body(dev, dim=dim):
                self.ac(torch.logsumexp(t(x), dim=dim).numpy(), np_logsumexp(x, dim),
                        rtol=1e-5, atol=1e-5, msg=f"logsumexp dim={dim} {dev}")
            both_devices(body)

    def test_logsumexp_keepdim(self):
        x = self.x
        def body(dev):
            self.ac(torch.logsumexp(t(x), dim=1, keepdim=True).numpy(),
                    np_logsumexp(x, 1, keepdims=True), rtol=1e-5, atol=1e-5,
                    msg=f"logsumexp keepdim {dev}")
        both_devices(body)


class TestAllAny(Base):
    def setUp(self):
        self.b = np.array([[1, 0, 1], [1, 1, 1]], dtype="bool")

    def test_all_any_dim(self):
        b = self.b
        def body(dev):
            self.ae(torch.all(t(b), dim=1).numpy(), b.all(1), msg=f"all d1 {dev}")
            self.ae(torch.any(t(b), dim=0).numpy(), b.any(0), msg=f"any d0 {dev}")
            self.ae(torch.all(t(b), dim=1, keepdim=True).numpy(),
                    b.all(1, keepdims=True), msg=f"all keepdim {dev}")
        both_devices(body)

    def test_all_any_full(self):
        b = self.b
        def body(dev):
            self.assertEqual(bool(torch.all(t(b)).item()), bool(b.all()),
                             msg=f"all full {dev}")
            self.assertEqual(bool(torch.any(t(b)).item()), bool(b.any()),
                             msg=f"any full {dev}")
        both_devices(body)


class TestCountNonzero(Base):
    def setUp(self):
        self.x = np.array([[0, 1, 2], [0, 0, 3]], dtype="float32")

    def test_count_nonzero_full(self):
        x = self.x
        def body(dev):
            self.assertEqual(int(torch.count_nonzero(t(x)).item()),
                             int(np.count_nonzero(x)), msg=f"cnz full {dev}")
        both_devices(body)

    def test_count_nonzero_dim(self):
        x = self.x
        def body(dev):
            self.ae(torch.count_nonzero(t(x), dim=1).numpy(), np.count_nonzero(x, 1),
                    msg=f"cnz d1 {dev}")
            self.ae(torch.count_nonzero(t(x), dim=0).numpy(), np.count_nonzero(x, 0),
                    msg=f"cnz d0 {dev}")
        both_devices(body)


class TestCumulative(Base):
    def test_cumsum(self):
        x = np.random.RandomState(10).randn(3, 4).astype("float32")
        def body(dev):
            self.ac(torch.cumsum(t(x), dim=1).numpy(), np.cumsum(x, axis=1),
                    rtol=1e-5, atol=1e-5, msg=f"cumsum d1 {dev}")
            self.ac(torch.cumsum(t(x), dim=0).numpy(), np.cumsum(x, axis=0),
                    rtol=1e-5, atol=1e-5, msg=f"cumsum d0 {dev}")
        both_devices(body)

    def test_cumprod(self):
        x = np.random.RandomState(11).rand(3, 4).astype("float32") + 0.5
        def body(dev):
            self.ac(torch.cumprod(t(x), dim=1).numpy(), np.cumprod(x, axis=1),
                    rtol=1e-4, atol=1e-5, msg=f"cumprod d1 {dev}")
        both_devices(body)

    def test_cummax(self):
        x = np.array([[1, 3, 2, 5, 4], [5, 4, 3, 2, 1]], dtype="float32")
        def body(dev):
            r = torch.cummax(t(x), dim=1)
            self.ac(r.values.numpy(), np.maximum.accumulate(x, axis=1),
                    msg=f"cummax values {dev}")
            # argmax-of-running-max index reference
            ref_idx = np.zeros_like(x, dtype="int64")
            for i in range(x.shape[0]):
                best = 0
                for j in range(x.shape[1]):
                    if x[i, j] >= x[i, best]:
                        best = j
                    ref_idx[i, j] = best
            self.ae(r.indices.numpy(), ref_idx, msg=f"cummax indices {dev}")
        both_devices(body)


# --------------------------------------------------------------------------------- shapes

class TestRepeatTile(Base):
    def test_repeat_1d(self):
        y = np.array([1, 2, 3], dtype="float32")
        def body(dev):
            self.ac(t(y).repeat(2).numpy(), np.tile(y, 2), msg=f"repeat(2) {dev}")
            self.ac(t(y).repeat(2, 3).numpy(), np.tile(y, (2, 3)),
                    msg=f"repeat(2,3) {dev}")
        both_devices(body)

    def test_repeat_2d(self):
        y = np.arange(6).reshape(2, 3).astype("float32")
        def body(dev):
            self.ac(t(y).repeat(2, 1).numpy(), np.tile(y, (2, 1)),
                    msg=f"repeat(2,1) {dev}")
            self.ac(t(y).repeat(1, 2).numpy(), np.tile(y, (1, 2)),
                    msg=f"repeat(1,2) {dev}")
        both_devices(body)

    def test_tile(self):
        y = np.array([1, 2, 3], dtype="float32")
        def body(dev):
            self.ac(torch.tile(t(y), (2, 2)).numpy(), np.tile(y, (2, 2)),
                    msg=f"tile(2,2) {dev}")
        both_devices(body)

    def test_repeat_interleave_scalar(self):
        y = np.array([1, 2, 3], dtype="float32")
        def body(dev):
            self.ac(torch.repeat_interleave(t(y), 2).numpy(), np.repeat(y, 2),
                    msg=f"repeat_interleave(2) {dev}")
        both_devices(body)

    def test_repeat_interleave_dim(self):
        y = np.arange(6).reshape(2, 3).astype("float32")
        def body(dev):
            self.ac(torch.repeat_interleave(t(y), 2, dim=0).numpy(),
                    np.repeat(y, 2, axis=0), msg=f"repeat_interleave d0 {dev}")
            self.ac(torch.repeat_interleave(t(y), 3, dim=1).numpy(),
                    np.repeat(y, 3, axis=1), msg=f"repeat_interleave d1 {dev}")
        both_devices(body)


class TestRollFlip(Base):
    def test_roll_1d(self):
        z = np.arange(10).astype("float32")
        def body(dev):
            self.ac(torch.roll(t(z), 2).numpy(), np.roll(z, 2), msg=f"roll +2 {dev}")
            self.ac(torch.roll(t(z), -3).numpy(), np.roll(z, -3), msg=f"roll -3 {dev}")
        both_devices(body)

    def test_roll_dims(self):
        z = np.arange(12).reshape(3, 4).astype("float32")
        def body(dev):
            self.ac(torch.roll(t(z), (1, 1), (0, 1)).numpy(), np.roll(z, (1, 1), (0, 1)),
                    msg=f"roll (1,1) (0,1) {dev}")
            self.ac(torch.roll(t(z), 1, 0).numpy(), np.roll(z, 1, 0),
                    msg=f"roll 1 d0 {dev}")
        both_devices(body)

    def test_flip(self):
        z = np.arange(24).reshape(2, 3, 4).astype("float32")
        def body(dev):
            self.ac(torch.flip(t(z), [1]).numpy(), np.flip(z, 1), msg=f"flip [1] {dev}")
            self.ac(torch.flip(t(z), [0, 2]).numpy(), np.flip(z, (0, 2)),
                    msg=f"flip [0,2] {dev}")
        both_devices(body)


class TestUnfold(Base):
    def test_unfold_1d(self):
        x = np.arange(10).astype("float32")
        # numpy reference via explicit windows: unfold(0,3,2)
        ref = np.stack([x[i:i + 3] for i in range(0, 10 - 3 + 1, 2)])
        def body(dev):
            self.ac(t(x).unfold(0, 3, 2).numpy(), ref, msg=f"unfold 1d {dev}")
        both_devices(body)

    def test_unfold_2d(self):
        x = np.arange(12).reshape(3, 4).astype("float32")
        # unfold(1,2,1) -> windows along axis 1 appended as a trailing dim: (3,3,2)
        ref = np.stack([x[:, i:i + 2] for i in range(0, 4 - 2 + 1, 1)], axis=1)
        def body(dev):
            self.ac(t(x).unfold(1, 2, 1).numpy(), ref, msg=f"unfold 2d {dev}")
        both_devices(body)


class TestSplitChunk(Base):
    def test_split_int(self):
        x = np.arange(10).astype("float32")
        def body(dev):
            parts = torch.split(t(x), 3)
            ref = np.split(x, [3, 6, 9])
            self.assertEqual(len(parts), len(ref), msg=f"split int len {dev}")
            for p, r in zip(parts, ref):
                self.ac(p.numpy(), r, msg=f"split int {dev}")
        both_devices(body)

    def test_split_list(self):
        x = np.arange(10).astype("float32")
        def body(dev):
            parts = torch.split(t(x), [2, 3, 5])
            ref = np.split(x, [2, 5])
            for p, r in zip(parts, ref):
                self.ac(p.numpy(), r, msg=f"split list {dev}")
        both_devices(body)

    def test_chunk(self):
        for n, c in [(10, 3), (12, 4), (7, 3), (10, 4)]:
            x = np.arange(n).astype("float32")
            sizes = torch_chunk_sizes(n, c)
            bounds = list(np.cumsum(sizes)[:-1])
            ref = np.split(x, bounds)
            def body(dev, x=x, ref=ref, c=c):
                parts = torch.chunk(t(x), c)
                self.assertEqual(len(parts), len(ref),
                                 msg=f"chunk len n={len(x)} c={c} {dev}")
                for p, r in zip(parts, ref):
                    self.ac(p.numpy(), r, msg=f"chunk n={len(x)} c={c} {dev}")
            both_devices(body)

    def test_tensor_split_int(self):
        x = np.arange(10).astype("float32")
        def body(dev):
            parts = torch.tensor_split(t(x), 3)
            ref = np.array_split(x, 3)   # numpy array_split == torch.tensor_split sizing
            for p, r in zip(parts, ref):
                self.ac(p.numpy(), r, msg=f"tensor_split int {dev}")
        both_devices(body)

    def test_tensor_split_indices(self):
        x = np.arange(10).astype("float32")
        def body(dev):
            parts = torch.tensor_split(t(x), [2, 5])
            ref = np.split(x, [2, 5])
            for p, r in zip(parts, ref):
                self.ac(p.numpy(), r, msg=f"tensor_split idx {dev}")
        both_devices(body)


class TestMovedim(Base):
    def setUp(self):
        self.z = np.arange(24).reshape(2, 3, 4).astype("float32")

    def test_movedim_single(self):
        z = self.z
        def body(dev):
            self.ac(torch.movedim(t(z), 0, 2).numpy(), np.moveaxis(z, 0, 2),
                    msg=f"movedim 0->2 {dev}")
            self.ac(torch.movedim(t(z), 2, 0).numpy(), np.moveaxis(z, 2, 0),
                    msg=f"movedim 2->0 {dev}")
            self.ac(torch.moveaxis(t(z), -1, 0).numpy(), np.moveaxis(z, -1, 0),
                    msg=f"moveaxis -1->0 {dev}")
        both_devices(body)

    def test_movedim_multi(self):
        z = self.z
        def body(dev):
            self.ac(torch.moveaxis(t(z), (0, 1), (2, 0)).numpy(),
                    np.moveaxis(z, (0, 1), (2, 0)), msg=f"moveaxis (0,1)->(2,0) {dev}")
        both_devices(body)


class TestBroadcastExpand(Base):
    def test_broadcast_to(self):
        a = np.array([[1], [2], [3]], dtype="float32")
        b = np.array([1, 2, 3], dtype="float32")
        def body(dev):
            self.ac(t(a).broadcast_to((3, 4)).numpy(), np.broadcast_to(a, (3, 4)),
                    msg=f"broadcast_to col {dev}")
            self.ac(t(b).broadcast_to((2, 3)).numpy(), np.broadcast_to(b, (2, 3)),
                    msg=f"broadcast_to row {dev}")
        both_devices(body)

    def test_expand(self):
        a = np.array([[1], [2], [3]], dtype="float32")
        def body(dev):
            self.ac(t(a).expand(3, 4).numpy(), np.broadcast_to(a, (3, 4)),
                    msg=f"expand(3,4) {dev}")
            self.ac(t(a).expand(-1, 4).numpy(), np.broadcast_to(a, (3, 4)),
                    msg=f"expand(-1,4) {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
