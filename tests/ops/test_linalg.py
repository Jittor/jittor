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
import jittor as jt
import numpy as np
import unittest

from _helpers.torch_runtime import import_torch_modules, modules_available


torch = None
Variable = None
anp = None
jacobian = None
has_autograd = modules_available("torch", "autograd")


def setUpModule():
    global torch, Variable, anp, jacobian
    if has_autograd:
        torch, anp, autograd = import_torch_modules(
            "torch", "autograd.numpy", "autograd"
        )
        Variable = torch.autograd.Variable
        jacobian = autograd.jacobian


@unittest.skipIf(not has_autograd, "No independent Torch or autograd found.")
class TestLinalgOp(unittest.TestCase):
    def test_svd(self):
        def check_svd(a):
            u, s, v = anp.linalg.svd(a, full_matrices=0)
            return u, s, v

        def check_u(a):
            u, s, v = anp.linalg.svd(a, full_matrices=0)
            return u

        def check_s(a):
            u, s, v = anp.linalg.svd(a, full_matrices=0)
            return s

        def check_v(a):
            u, s, v = anp.linalg.svd(a, full_matrices=0)
            return v

        for i in range(50):
            # not for full-matrices!
            a = jt.random((2, 2, 5, 4))
            c_a = anp.array(a.data)
            u, s, v = jt.linalg.svd(a)
            tu, ts, tv = check_svd(c_a)
            assert np.allclose(tu, u.data)
            assert np.allclose(ts, s.data)
            assert np.allclose(tv, v.data)
            ju = jt.grad(u, a)
            js = jt.grad(s, a)
            jv = jt.grad(v, a)
            grad_u = jacobian(check_u)
            gu = grad_u(c_a)
            gu = np.sum(gu, 4)
            gu = np.sum(gu, 4)
            gu = np.sum(gu, 2)
            gu = np.sum(gu, 2)
            grad_s = jacobian(check_s)
            gs = grad_s(c_a)
            gs = np.sum(gs, 4)
            gs = np.sum(gs, 2)
            gs = np.sum(gs, 2)
            grad_v = jacobian(check_v)
            gv = grad_v(c_a)
            gv = np.sum(gv, 4)
            gv = np.sum(gv, 4)
            gv = np.sum(gv, 2)
            gv = np.sum(gv, 2)
            try:
                assert np.allclose(ju.data, gu, atol=1e-5)
            except AssertionError:
                print(ju.data)
                print(gu)
            try:
                assert np.allclose(js.data, gs, atol=1e-5)
            except AssertionError:
                print(js.data)
                print(gs)
            try:
                assert np.allclose(jv.data, gv, atol=1e-5)
            except AssertionError:
                print(jv.data)
                print(gv)

    def test_eigh(self):
        def check_eigh(a, UPLO='L'):
            w, v = anp.linalg.eigh(a, UPLO)
            return w, v

        def check_w(a, UPLO='L'):
            w, v = anp.linalg.eigh(a, UPLO)
            return w

        def check_v(a, UPLO='L'):
            w, v = anp.linalg.eigh(a, UPLO)
            return v

        for i in range(50):
            a = jt.random((2, 2, 3, 3))
            c_a = a.data
            w, v = jt.linalg.eigh(a)
            tw, tv = check_eigh(c_a)
            assert np.allclose(w.data, tw)
            assert np.allclose(v.data, tv)
            jw = jt.grad(w, a)
            jv = jt.grad(v, a)
            check_gw = jacobian(check_w)
            check_gv = jacobian(check_v)
            gw = check_gw(c_a)
            gw = np.sum(gw, 4)
            gw = np.sum(gw, 2)
            gw = np.sum(gw, 2)
            assert np.allclose(gw, jw.data, rtol=1, atol=5e-8)
            gv = check_gv(c_a)
            gv = np.sum(gv, 4)
            gv = np.sum(gv, 4)
            gv = np.sum(gv, 2)
            gv = np.sum(gv, 2)
            assert np.allclose(gv, jv.data, rtol=1, atol=5e-8)

    def test_pinv(self):
        def check_pinv(a):
            w = anp.linalg.pinv(a)
            return w

        for i in range(50):
            x = jt.random((2, 2, 4, 3))
            c_a = x.data
            mx = jt.linalg.pinv(x)
            tx = check_pinv(c_a)
            np.allclose(mx.data, tx)
            jx = jt.grad(mx, x)
            check_grad = jacobian(check_pinv)
            gx = check_grad(c_a)
            np.allclose(gx, jx.data)

    def test_inv(self):
        def check_inv(a):
            w = anp.linalg.inv(a)
            return w

        for i in range(50):
            tn = np.random.randn(4, 4).astype('float32') * 5
            while np.allclose(np.linalg.det(tn), 0):
                tn = np.random.randn((4, 4)).astype('float32') * 5
            x = jt.array(tn)
            x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            c_a = x.data
            mx = jt.linalg.inv(x)
            tx = check_inv(c_a)
            np.allclose(mx.data, tx)
            jx = jt.grad(mx, x)
            check_grad = jacobian(check_inv)
            gx = check_grad(c_a)
            np.allclose(gx, jx.data)

    def test_slogdet(self):
        def check_ans(a):
            s, w = anp.linalg.slogdet(a)
            return s, w

        def check_slogdet(a):
            s, w = anp.linalg.slogdet(a)
            return w

        for i in range(50):
            tn = np.random.randn(4, 4).astype('float32') * 10
            while np.allclose(np.linalg.det(tn), 0):
                tn = np.random.randn((4, 4)).astype('float32') * 10
            x = jt.array(tn)
            x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            s = list(x.shape)
            det_s = s[:-2]
            if len(det_s) == 0:
                det_s.append(1)
            sign, mx = jt.linalg.slogdet(x)
            ts, ta = check_ans(x.data)
            assert np.allclose(sign.data, ts)
            assert np.allclose(mx.data, ta)
            jx = jt.grad(mx, x)
            check_sgrad = jacobian(check_slogdet)
            gx = check_sgrad(x.data)
            gx = np.sum(gx, 2)
            gx = np.sum(gx, 2)
            assert np.allclose(gx, jx.data)

    def test_cholesky(self):
        def check_cholesky(a):
            L = anp.linalg.cholesky(a)
            return L

        for i in range(50):
            x = jt.array(np.diag((np.random.rand(3) + 1) * 2))
            x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            tx = x.data
            L = jt.linalg.cholesky(x)
            tL = check_cholesky(tx)
            assert np.allclose(tL, L.data)
            jx = jt.grad(L, x)
            check_grad = jacobian(check_cholesky)
            gx = check_grad(tx)
            gx = np.sum(gx, 0)
            gx = np.sum(gx, 0)
            gx = np.sum(gx, 0)
            gx = np.sum(gx, 0)
            assert np.allclose(jx.data, gx)

    def test_solve(self):
        def check_solve(a, b):
            ans = anp.linalg.solve(a, b)
            return ans

        for i in range(50):
            a = jt.random((2, 2, 3, 3))
            b = jt.random((2, 2, 3))
            ans = jt.linalg.solve(a, b)
            ta = check_solve(a.data, b.data)
            assert np.allclose(ans.data, ta)
            jx = jt.grad(ans, a)
            check_sgrad = jacobian(check_solve)
            gx = check_sgrad(a.data, b.data)
            gx = np.sum(gx, 0)
            gx = np.sum(gx, 0)
            gx = np.sum(gx, 0)
            try:
                assert np.allclose(gx, jx.data, rtol=1)
            except AssertionError:
                print(gx)
                print(jx.data)

    def test_det(self):
        def check_det(a):
            de = anp.linalg.det(a)
            return de

        for i in range(50):
            tn = np.random.randn(3, 3).astype('float32') * 5
            while np.allclose(np.linalg.det(tn), 0):
                tn = np.random.randn((3, 3)).astype('float32') * 5
            x = jt.array(tn)
            x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            s = list(x.shape)
            x_s = s[:-2]
            if len(s) == 2:
                x_s.append(1)
            det = jt.linalg.det(x)
            ta = check_det(x.data)
            assert np.allclose(det.data, ta)
            jx = jt.grad(det, x)
            check_sgrad = jacobian(check_det)
            gx = check_sgrad(x.data)
            gx = np.sum(gx, 2)
            gx = np.sum(gx, 2)
            assert np.allclose(gx, jx.data)

    def test_qr(self):
        for i in range(50):
            tn = np.random.randn(3, 3).astype('float32')
            while np.allclose(np.linalg.det(tn), 0):
                tn = np.random.randn((3, 3)).astype('float32')
            x = jt.array(tn)
            # x = x.reindex([2, 2, x.shape[0], x.shape[1]], ["i2", "i3"])
            t_x = torch.from_numpy(tn)
            t_x = Variable(t_x, requires_grad=True)
            jq, jr = jt.linalg.qr(x)
            tq, tr = torch.qr(t_x)
            try:
                assert np.allclose(jq.data, tq.detach().numpy(), rtol=1e-4, atol=1e-6)
                assert np.allclose(jr.data, tr.detach().numpy(), rtol=1e-4, atol=1e-6)
            except AssertionError:
                print("ours' qr results:")
                print(jq)
                print(jr)
                print("pytorch's qr results:")
                print(tq)
                print(tr)
            gq = jt.grad(jq, x).data
            gr = jt.grad(jr, x).data
            tgq = torch.autograd.grad(tq, t_x, torch.ones_like(tq), retain_graph=True)
            tgr = torch.autograd.grad(tr, t_x, torch.ones_like(tr), retain_graph=True)
            try:
                assert np.allclose(gq, tgq[0].numpy(), rtol=1e-4, atol=1e-6)
                assert np.allclose(gr, tgr[0].numpy(), rtol=1e-4, atol=1e-6)
            except AssertionError:
                print("ours' qr grad results:")
                print(gq)
                print(gr)
                print("pytorch's qr grad result")
                print(tgq[0])
                print(tgr[0])

@unittest.skipIf(not jt.has_cuda, "No cuda found.")
class TestBUG4_2Op(unittest.TestCase):
    # flag_scope, not a bare assignment: `jt.flags.use_cuda = 1` here used to
    # leak CUDA into every test that ran after this one in the same process,
    # so tests written and read as CPU tests were silently exercising a
    # different backend. 0.12.
    @jt.flag_scope(use_cuda=1)
    def test(self):
        x = jt.randn(32, 50, 2)
        y = jt.rand(32, 1, 2)

        # MLE
        mean = x.mean(dim=1, keepdims=True)# [batch_size, 1, n_feature]
        mup = jt.transpose((x - mean), [0, 2, 1])# [batch_size, n_feature, n_particles]
        cov = jt.nn.bmm_transpose(mup, mup) / (50 - 1)# [batch_size, n_feature, n_feature]
        prec = jt.linalg.inv(cov)# [batch_size, n_feature, n_feature]
        # print(prec)
        # log_prob
        dst = y - mean
        log_prob = -1/2 * jt.bmm(dst, jt.bmm_transpose(prec, dst))
        grad = jt.grad(log_prob, x)
        grad.sync()

class TestEighZeroEigenvectorGrad(unittest.TestCase):
    """``eigh``'s backward must write its output buffer unconditionally.

    ``jt.numpy_code`` hands the backward a freshly allocated, *uninitialized*
    output array.  The eigenvector branch skipped the write when ``dout`` was
    all zeros (a loss that only reads the eigenvalues, or one that multiplies
    the eigenvectors by a runtime zero), so whatever the allocator had recycled
    into that memory was returned as the gradient and summed into the result.

    Needs no torch: the eigenvalue gradient of a symmetric matrix is
    ``V diag(dw) V^T``, which numpy computes directly.
    """

    SIZE = 5

    def setUp(self):
        # Pin the device: under ``use_cuda`` a ``numpy_code`` callback is handed
        # *cupy*, so ``np.linalg.eigh`` becomes cuSOLVER's, whose eigenvectors
        # carry different (equally valid) column signs than LAPACK's.  The
        # numpy closed form below fixes LAPACK's convention, so it is only a
        # valid oracle on the host.  ``TestEighCrossDevice`` covers CUDA with
        # sign-invariant assertions instead.
        self._saved_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 0

    def tearDown(self):
        jt.flags.use_cuda = self._saved_use_cuda

    @staticmethod
    def _poison(shape, value):
        """Fill and release buffers so the allocator hands back dirty memory."""
        junk = [jt.ones(shape) * value for _ in range(128)]
        for block in junk:
            block.sync()
        del junk

    def _symmetric(self, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((self.SIZE, self.SIZE))
        return ((a + a.T) / 2).astype("float32"), rng

    def _grad(self, x, w_seed, with_zero_eigenvector_term):
        xv = jt.array(x)
        w, v = jt.linalg.eigh(xv)
        loss = (w * jt.array(w_seed)).sum()
        if with_zero_eigenvector_term:
            zeros = np.zeros((self.SIZE, self.SIZE), dtype="float32")
            loss = loss + (v * jt.array(zeros)).sum()
        grad, = jt.grad(loss, [xv])
        return grad.numpy().copy()

    def test_zero_dout_does_not_leak_recycled_memory(self):
        x, rng = self._symmetric(0)
        w_seed = rng.standard_normal(self.SIZE).astype("float32")
        _, vectors = np.linalg.eigh(x.astype("float64"), UPLO="L")
        expected = vectors @ np.diag(w_seed.astype("float64")) @ vectors.T
        for trial in range(8):
            with self.subTest(trial=trial):
                self._poison((self.SIZE, self.SIZE), 98765.0 + trial)
                got = self._grad(x, w_seed, with_zero_eigenvector_term=True)
                np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

    def test_zero_eigenvector_grad_contributes_exactly_zero(self):
        """Not "close to zero" -- adding it must leave the sum bit-identical."""
        x, rng = self._symmetric(1)
        w_seed = rng.standard_normal(self.SIZE).astype("float32")
        for trial in range(8):
            with self.subTest(trial=trial):
                self._poison((self.SIZE, self.SIZE), -4321.0 - trial)
                with_term = self._grad(x, w_seed, with_zero_eigenvector_term=True)
                without_term = self._grad(x, w_seed, with_zero_eigenvector_term=False)
                np.testing.assert_array_equal(with_term, without_term)

    def test_zero_dout_batched(self):
        rng = np.random.default_rng(2)
        a = rng.standard_normal((3, 4, 4))
        x = ((a + np.swapaxes(a, -1, -2)) / 2).astype("float32")
        w_seed = rng.standard_normal((3, 4)).astype("float32")
        _, vectors = np.linalg.eigh(x.astype("float64"), UPLO="L")
        expected = np.einsum(
            "bij,bj,bkj->bik", vectors, w_seed.astype("float64"), vectors
        )
        for trial in range(4):
            with self.subTest(trial=trial):
                self._poison((3, 4, 4), 5555.0 + trial)
                xv = jt.array(x)
                w, v = jt.linalg.eigh(xv)
                zeros = np.zeros((3, 4, 4), dtype="float32")
                loss = (w * jt.array(w_seed)).sum() + (v * jt.array(zeros)).sum()
                grad, = jt.grad(loss, [xv])
                np.testing.assert_allclose(
                    grad.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_nonzero_dout_still_uses_the_eigenvector_formula(self):
        """The fix must not disturb the ordinary (non-zero ``dout``) path."""
        x, rng = self._symmetric(3)
        v_seed = rng.standard_normal((self.SIZE, self.SIZE)).astype("float32")
        values, vectors = np.linalg.eigh(x.astype("float64"), UPLO="L")
        off = np.ones((self.SIZE, self.SIZE)) - np.eye(self.SIZE)
        repeated = np.repeat(values[..., None], self.SIZE, axis=-1)
        f = off / (repeated.T - repeated + np.eye(self.SIZE))
        expected = vectors @ (f * (vectors.T @ v_seed.astype("float64"))) @ vectors.T
        xv = jt.array(x)
        _, v = jt.linalg.eigh(xv)
        grad, = jt.grad((v * jt.array(v_seed)).sum(), [xv])
        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-4, atol=1e-4)


class TestEighCrossDevice(unittest.TestCase):
    """What ``eigh`` does and does not promise across CPU and CUDA.

    ``jt.numpy_code`` hands its callback the ``cupy`` module instead of
    ``numpy`` whenever ``use_cuda`` is on (see ``pyjt/py_converter.h`` and
    ``init_cupy.py``), so ``jt.linalg.eigh`` is LAPACK on the host and cuSOLVER
    on the device.  Eigenvectors are only defined up to a per-column sign, and
    the two libraries do not agree on it -- exactly as ``torch.linalg.eigh``
    documents.

    So the following are *not* bugs and must not be "fixed":
      - ``v`` differing from ``numpy.linalg.eigh``'s ``v`` by column signs on CUDA;
      - the gradient of a sign-dependent loss such as ``(v * seed).sum()``
        differing between devices.

    These are the invariants that do hold, and this class pins them:
      - eigenvalues match numpy on both devices;
      - ``v diag(w) v^T`` reconstructs the input on both devices;
      - the gradient is the closed form evaluated with the eigenvectors that
        *this* device returned (self-consistency);
      - a sign-invariant loss gives the same gradient on both devices.
    """

    SIZE = 5

    def setUp(self):
        self._saved_use_cuda = jt.flags.use_cuda
        rng = np.random.default_rng(31)
        a = rng.standard_normal((self.SIZE, self.SIZE))
        self.x = ((a + a.T) / 2).astype("float32")
        self.seed = rng.standard_normal((self.SIZE, self.SIZE)).astype("float32")

    def tearDown(self):
        jt.flags.use_cuda = self._saved_use_cuda

    def _closed_form_vector_grad(self, values, vectors, dout):
        size = self.SIZE
        off_diag = np.ones((size, size)) - np.eye(size)
        repeated = np.repeat(values[..., None], size, axis=-1)
        f = off_diag / (repeated.T - repeated + np.eye(size))
        return vectors @ (f * (vectors.T @ dout)) @ vectors.T

    def _devices(self):
        return (0, 1) if jt.compiler.has_cuda else (0,)

    def test_eigenvalues_and_reconstruction_match_on_every_device(self):
        expected_values = np.linalg.eigvalsh(self.x.astype("float64"), UPLO="L")
        for use_cuda in self._devices():
            with self.subTest(use_cuda=use_cuda):
                jt.flags.use_cuda = use_cuda
                w, v = jt.linalg.eigh(jt.array(self.x))
                np.testing.assert_allclose(
                    w.numpy(), expected_values, rtol=1e-4, atol=1e-4)
                values = w.numpy().astype("float64")
                vectors = v.numpy().astype("float64")
                np.testing.assert_allclose(
                    vectors @ np.diag(values) @ vectors.T,
                    self.x.astype("float64"), rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(
                    vectors.T @ vectors, np.eye(self.SIZE), rtol=1e-4, atol=1e-4)

    def test_vector_gradient_is_self_consistent_on_every_device(self):
        for use_cuda in self._devices():
            with self.subTest(use_cuda=use_cuda):
                jt.flags.use_cuda = use_cuda
                xv = jt.array(self.x)
                _, v = jt.linalg.eigh(xv)
                grad, = jt.grad((v * jt.array(self.seed)).sum(), [xv])
                w2, v2 = jt.linalg.eigh(jt.array(self.x))
                expected = self._closed_form_vector_grad(
                    w2.numpy().astype("float64"),
                    v2.numpy().astype("float64"),
                    self.seed.astype("float64"),
                )
                np.testing.assert_allclose(
                    grad.numpy(), expected, rtol=1e-3, atol=1e-3)

    def test_sign_invariant_loss_gives_the_same_gradient_on_every_device(self):
        """``v diag(w) v^T`` does not depend on the eigenvector sign convention."""
        results = []
        for use_cuda in self._devices():
            jt.flags.use_cuda = use_cuda
            xv = jt.array(self.x)
            w, v = jt.linalg.eigh(xv)
            reconstruction = jt.matmul(
                v * w.broadcast(v.shape, [0]), v.transpose(1, 0))
            grad, = jt.grad((reconstruction * jt.array(self.seed)).sum(), [xv])
            results.append(grad.numpy())
        for other in results[1:]:
            np.testing.assert_allclose(other, results[0], rtol=1e-3, atol=1e-3)

    def test_zero_eigenvector_grad_writes_zero_on_every_device(self):
        """The 6.P07 ``else: copyto(out, 0)`` branch also runs under cupy."""
        rng = np.random.default_rng(41)
        value_seed = rng.standard_normal(self.SIZE).astype("float32")
        zeros = np.zeros((self.SIZE, self.SIZE), dtype="float32")
        for use_cuda in self._devices():
            with self.subTest(use_cuda=use_cuda):
                jt.flags.use_cuda = use_cuda
                xv = jt.array(self.x)
                w, v = jt.linalg.eigh(xv)
                seeded = (w * jt.array(value_seed)).sum()
                with_term, = jt.grad(seeded + (v * jt.array(zeros)).sum(), [xv])
                xv2 = jt.array(self.x)
                w2, _ = jt.linalg.eigh(xv2)
                without_term, = jt.grad(
                    (w2 * jt.array(value_seed)).sum(), [xv2])
                np.testing.assert_array_equal(
                    with_term.numpy(), without_term.numpy())


if __name__ == "__main__":
    unittest.main()
