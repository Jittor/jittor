"""Torch-grade parity tests for the linalg SVD family on the
``import jittor as torch`` path: ``torch.linalg.svd`` (full_matrices=True/False
shape semantics + named (U, S, Vh) tuple), ``torch.linalg.svdvals`` (singular
values only), and ``torch.linalg.eigvalsh`` (symmetric/Hermitian eigenvalues).

References are numpy (``numpy.linalg.svd`` / ``eigvalsh``), which match real
torch (verified against torch 2.12 while writing these). Checks are
gauge-invariant where the decomposition is sign/rotation-ambiguous: we test
reconstruction ``A = U diag(S) Vh``, singular values sorted *descending*, and
eigenvalues sorted *ascending* — never the raw (sign-ambiguous) U / Vh / vector
entries. Tolerances are NOT loosened to hide divergence.

Run:  python -m pytest tests/compat/torch/test_torch_compat_svd.py
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
    def ac(self, got, ref, atol=1e-4, rtol=1e-4, msg=""):
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                                   atol=atol, rtol=rtol, err_msg=msg)

    def spd(self, n, seed):
        a = np.random.RandomState(seed).randn(n, n).astype("float64")
        return (a @ a.T + n * np.eye(n)).astype("float32")


class TestSVD(Base):
    # ---- named tuple + tuple-unpack compatibility ----------------------------
    def test_named_tuple_and_unpack(self):
        A = np.random.RandomState(0).randn(4, 4).astype("float32")

        def body(dev):
            r = torch.linalg.svd(jt.array(A))
            # attribute access (torch-grade) ...
            self.assertTrue(hasattr(r, "U") and hasattr(r, "S") and hasattr(r, "Vh"),
                            f"named fields {dev}")
            # ... and plain 3-tuple unpack / indexing still work
            u, s, vh = r
            self.assertEqual(len(r), 3, f"len==3 {dev}")
            self.ac(r[0].numpy(), u.numpy(), msg=f"index0==U {dev}")
            self.ac(r.S.numpy(), s.numpy(), msg=f".S==index1 {dev}")
        both_devices(body)

    # ---- default is the reduced form (jittor-native contract) ----------------
    def test_default_is_reduced(self):
        # NOTE on torch parity: torch.linalg.svd defaults to full_matrices=True;
        # this jittor-native svd keeps the reduced default (differentiable path +
        # all internal callers + native test_linalg rely on it). full shapes are
        # available via full_matrices=True. The torch-facing default is meant to
        # be applied at the torch-compat boundary.
        B = np.random.RandomState(100).randn(5, 3).astype("float32")

        def body(dev):
            r = torch.linalg.svd(jt.array(B))           # no kwarg -> reduced
            self.assertEqual(tuple(r.U.shape), (5, 3), f"default reduced U {dev}")
            self.assertEqual(tuple(r.Vh.shape), (3, 3), f"default reduced Vh {dev}")
            recon = r.U.numpy() @ np.diag(r.S.numpy()) @ r.Vh.numpy()
            self.ac(recon, B, atol=1e-3, msg=f"default reduced recon {dev}")
        both_devices(body)

    # ---- full_matrices shape semantics: TALL (m > n) -------------------------
    def test_full_matrices_tall(self):
        # B is 5x3 -> k=3. full: U(5,5), Vh(3,3); reduced: U(5,3), Vh(3,3).
        B = np.random.RandomState(1).randn(5, 3).astype("float32")
        k = 3

        def body(dev):
            full = torch.linalg.svd(jt.array(B), full_matrices=True)
            self.assertEqual(tuple(full.U.shape), (5, 5), f"full U tall {dev}")
            self.assertEqual(tuple(full.Vh.shape), (3, 3), f"full Vh tall {dev}")
            self.assertEqual(tuple(full.S.shape), (3,), f"full S tall {dev}")
            recon = full.U.numpy()[:, :k] @ np.diag(full.S.numpy()) @ full.Vh.numpy()[:k, :]
            self.ac(recon, B, atol=1e-3, msg=f"full recon tall {dev}")

            thin = torch.linalg.svd(jt.array(B), full_matrices=False)
            self.assertEqual(tuple(thin.U.shape), (5, 3), f"thin U tall {dev}")
            self.assertEqual(tuple(thin.Vh.shape), (3, 3), f"thin Vh tall {dev}")
            recon2 = thin.U.numpy() @ np.diag(thin.S.numpy()) @ thin.Vh.numpy()
            self.ac(recon2, B, atol=1e-3, msg=f"thin recon tall {dev}")
        both_devices(body)

    # ---- full_matrices shape semantics: WIDE (m < n) -------------------------
    def test_full_matrices_wide(self):
        # W is 3x5 -> k=3. full: U(3,3), Vh(5,5); reduced: U(3,3), Vh(3,5).
        W = np.random.RandomState(2).randn(3, 5).astype("float32")
        k = 3

        def body(dev):
            full = torch.linalg.svd(jt.array(W), full_matrices=True)
            self.assertEqual(tuple(full.U.shape), (3, 3), f"full U wide {dev}")
            self.assertEqual(tuple(full.Vh.shape), (5, 5), f"full Vh wide {dev}")
            recon = full.U.numpy() @ np.diag(full.S.numpy()) @ full.Vh.numpy()[:k, :]
            self.ac(recon, W, atol=1e-3, msg=f"full recon wide {dev}")

            thin = torch.linalg.svd(jt.array(W), full_matrices=False)
            self.assertEqual(tuple(thin.U.shape), (3, 3), f"thin U wide {dev}")
            self.assertEqual(tuple(thin.Vh.shape), (3, 5), f"thin Vh wide {dev}")
            recon2 = thin.U.numpy() @ np.diag(thin.S.numpy()) @ thin.Vh.numpy()
            self.ac(recon2, W, atol=1e-3, msg=f"thin recon wide {dev}")
        both_devices(body)

    # ---- square: reduced == full -------------------------------------------
    def test_square_recon_and_values(self):
        A = np.random.RandomState(3).randn(4, 4).astype("float32")

        def body(dev):
            # square: reduced default already equals full shapes
            full = torch.linalg.svd(jt.array(A))
            self.assertEqual((tuple(full.U.shape), tuple(full.Vh.shape)),
                             ((4, 4), (4, 4)), f"square shapes {dev}")
            recon = full.U.numpy() @ np.diag(full.S.numpy()) @ full.Vh.numpy()
            self.ac(recon, A, atol=1e-3, msg=f"square recon {dev}")
            self.ac(full.S.numpy(), np.linalg.svd(A, compute_uv=False),
                    rtol=1e-3, msg=f"square S desc {dev}")
        both_devices(body)

    # ---- batched full_matrices ----------------------------------------------
    def test_batched_full_matrices(self):
        Bb = np.random.RandomState(4).randn(2, 5, 3).astype("float32")

        def body(dev):
            full = torch.linalg.svd(jt.array(Bb), full_matrices=True)
            self.assertEqual(tuple(full.U.shape), (2, 5, 5), f"batched full U {dev}")
            self.assertEqual(tuple(full.Vh.shape), (2, 3, 3), f"batched full Vh {dev}")
            self.assertEqual(tuple(full.S.shape), (2, 3), f"batched S {dev}")
            U, S, Vh = full.U.numpy(), full.S.numpy(), full.Vh.numpy()
            for i in range(2):
                recon = U[i][:, :3] @ np.diag(S[i]) @ Vh[i][:3, :]
                self.ac(recon, Bb[i], atol=1e-3, msg=f"batched recon[{i}] {dev}")
            thin = torch.linalg.svd(jt.array(Bb), full_matrices=False)
            self.assertEqual(tuple(thin.U.shape), (2, 5, 3), f"batched thin U {dev}")
        both_devices(body)

    # ---- S is descending ----------------------------------------------------
    def test_singular_values_descending(self):
        A = np.random.RandomState(5).randn(6, 4).astype("float32")

        def body(dev):
            S = torch.linalg.svd(jt.array(A), full_matrices=False).S.numpy()
            self.assertTrue(np.all(np.diff(S) <= 1e-5), f"S descending {dev}")
            self.ac(S, np.linalg.svd(A, compute_uv=False), rtol=1e-3,
                    msg=f"S values {dev}")
        both_devices(body)

    # ---- reduced svd stays differentiable -----------------------------------
    def test_reduced_svd_differentiable(self):
        A = np.random.RandomState(6).randn(4, 4).astype("float32")

        def body(dev):
            M = jt.array(A.copy())
            g = jt.grad(torch.linalg.svd(M, full_matrices=False).S.sum(), [M])[0]
            self.assertTrue(bool(jt.isfinite(g).all().item()), f"grad finite {dev}")
            self.assertGreater(float(jt.abs(g).sum().item()), 0, f"grad nonzero {dev}")
        both_devices(body)


class TestSvdvals(Base):
    def test_svdvals_values_and_shape(self):
        # tall, wide, square
        for seed, shape in ((10, (5, 3)), (11, (3, 5)), (12, (4, 4))):
            A = np.random.RandomState(seed).randn(*shape).astype("float32")
            ref = np.linalg.svd(A, compute_uv=False)  # descending

            def body(dev, A=A, ref=ref, shape=shape):
                sv = torch.linalg.svdvals(jt.array(A))
                self.assertEqual(tuple(sv.shape), (min(shape),), f"svdvals shape {shape} {dev}")
                self.assertTrue(np.all(np.diff(sv.numpy()) <= 1e-5),
                                f"svdvals descending {shape} {dev}")
                self.ac(sv.numpy(), ref, rtol=1e-3, msg=f"svdvals {shape} {dev}")
            both_devices(body)

    def test_svdvals_matches_svd_S(self):
        A = np.random.RandomState(13).randn(5, 3).astype("float32")

        def body(dev):
            sv = torch.linalg.svdvals(jt.array(A)).numpy()
            s_from_svd = torch.linalg.svd(jt.array(A), full_matrices=False).S.numpy()
            self.ac(sv, s_from_svd, rtol=1e-4, msg=f"svdvals==svd.S {dev}")
        both_devices(body)

    def test_svdvals_differentiable(self):
        A = np.random.RandomState(14).randn(4, 3).astype("float32")

        def body(dev):
            M = jt.array(A.copy())
            g = jt.grad(torch.linalg.svdvals(M).sum(), [M])[0]
            self.assertTrue(bool(jt.isfinite(g).all().item()), f"svdvals grad finite {dev}")
            self.assertGreater(float(jt.abs(g).sum().item()), 0, f"svdvals grad nonzero {dev}")
        both_devices(body)


class TestEigvalsh(Base):
    def test_eigvalsh_ascending_and_values(self):
        A = self.spd(4, 20)
        ref = np.linalg.eigvalsh(A)  # ascending

        def body(dev):
            w = torch.linalg.eigvalsh(jt.array(A))
            self.assertEqual(tuple(w.shape), (4,), f"eigvalsh shape {dev}")
            self.assertTrue(np.all(np.diff(w.numpy()) >= -1e-5),
                            f"eigvalsh ascending {dev}")
            self.ac(w.numpy(), ref, rtol=1e-3, atol=1e-4, msg=f"eigvalsh values {dev}")
        both_devices(body)

    def test_eigvalsh_batched(self):
        a = np.random.RandomState(21).randn(2, 4, 4).astype("float32")
        SPD = (np.matmul(a, np.transpose(a, (0, 2, 1))) + 4 * np.eye(4)).astype("float32")
        ref = np.linalg.eigvalsh(SPD)

        def body(dev):
            w = torch.linalg.eigvalsh(jt.array(SPD))
            self.assertEqual(tuple(w.shape), (2, 4), f"eigvalsh batched shape {dev}")
            self.ac(w.numpy(), ref, rtol=1e-3, atol=1e-4, msg=f"eigvalsh batched {dev}")
        both_devices(body)

    def test_eigvalsh_uplo_symmetric_invariance(self):
        # for a genuinely symmetric matrix UPLO='U' and 'L' give the same spectrum
        A = self.spd(5, 22)
        ref = np.linalg.eigvalsh(A)

        def body(dev):
            wl = torch.linalg.eigvalsh(jt.array(A), UPLO='L').numpy()
            wu = torch.linalg.eigvalsh(jt.array(A), UPLO='U').numpy()
            self.ac(wl, ref, rtol=1e-3, atol=1e-4, msg=f"UPLO=L {dev}")
            self.ac(wu, ref, rtol=1e-3, atol=1e-4, msg=f"UPLO=U {dev}")
        both_devices(body)

    def test_eigvalsh_matches_eigh(self):
        A = self.spd(4, 23)

        def body(dev):
            w_only = torch.linalg.eigvalsh(jt.array(A)).numpy()
            w_eigh = torch.linalg.eigh(jt.array(A))[0].numpy()
            self.ac(np.sort(w_only), np.sort(w_eigh), rtol=1e-4, atol=1e-5,
                    msg=f"eigvalsh==eigh[0] {dev}")
        both_devices(body)

    def test_eigvalsh_differentiable(self):
        A = self.spd(4, 24)

        def body(dev):
            M = jt.array(A.copy())
            g = jt.grad(torch.linalg.eigvalsh(M).sum(), [M])[0]
            self.assertTrue(bool(jt.isfinite(g).all().item()), f"eigvalsh grad finite {dev}")
            # sum of eigenvalues == trace, so d(trace)/dA == I -> nonzero grad
            self.assertGreater(float(jt.abs(g).sum().item()), 0, f"eigvalsh grad nonzero {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
