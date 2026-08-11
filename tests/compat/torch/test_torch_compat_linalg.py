"""Torch-grade linalg parity tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite. Compares jittor-as-torch linalg ops against
numpy references. Uses gauge-invariant checks (reconstructions / singular values / |det|)
where the decomposition is sign/rotation-ambiguous (svd, qr, eig).

Run:  python -m pytest tests/compat/torch/test_torch_compat_linalg.py
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
        g = np.asarray(got); r = np.asarray(ref)
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def spd(self, n, seed):
        # a symmetric positive-definite matrix (well-conditioned)
        a = np.random.RandomState(seed).randn(n, n).astype("float64")
        return (a @ a.T + n * np.eye(n)).astype("float32")


class TestLinalg(Base):
    def test_mv(self):
        A = np.random.RandomState(11).randn(3, 4).astype("float32")
        v = np.random.RandomState(12).randn(4).astype("float32")
        ref = A @ v
        def body(dev):
            matrix, vector = torch.tensor(A), torch.tensor(v)
            self.ac(torch.mv(matrix, vector).numpy(), ref, msg=f"torch.mv {dev}")
            self.ac(matrix.mv(vector).numpy(), ref, msg=f"Tensor.mv {dev}")
            out = torch.empty((3,), dtype=torch.float32)
            self.assertIs(torch.mv(matrix, vector, out=out), out, f"torch.mv out identity {dev}")
            self.ac(out.numpy(), ref, msg=f"torch.mv out value {dev}")
            with self.assertRaisesRegex(RuntimeError, "2-D matrix"):
                torch.mv(matrix.reshape(1, 3, 4), vector)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                matrix.mv(torch.ones((3,)))
        both_devices(body)

    def test_det_slogdet(self):
        A = self.spd(4, 0)
        def body(dev):
            self.ac(float(torch.linalg.det(jt.array(A)).item()), np.linalg.det(A),
                    rtol=1e-3, msg=f"det {dev}")
            s, ld = torch.linalg.slogdet(jt.array(A))
            rs, rld = np.linalg.slogdet(A)
            self.ac(float(s.item()), rs, msg=f"slogdet sign {dev}")
            self.ac(float(ld.item()), rld, rtol=1e-3, msg=f"slogdet logabs {dev}")
        both_devices(body)

    def test_inv(self):
        A = self.spd(5, 1)
        def body(dev):
            inv = torch.linalg.inv(jt.array(A)).numpy()
            self.ac(inv @ A, np.eye(5), atol=1e-3, msg=f"inv@A==I {dev}")
        both_devices(body)

    def test_solve(self):
        A = self.spd(4, 2); b = np.random.RandomState(3).randn(4, 2).astype("float32")
        def body(dev):
            x = torch.linalg.solve(jt.array(A), jt.array(b)).numpy()
            self.ac(A @ x, b, atol=1e-3, msg=f"A@x==b {dev}")
        both_devices(body)

    def test_cholesky(self):
        A = self.spd(4, 4)
        def body(dev):
            L = torch.linalg.cholesky(jt.array(A)).numpy()
            self.ac(L @ L.T, A, atol=1e-3, msg=f"L@L.T==A {dev}")
            self.assertTrue(np.allclose(np.triu(L, 1), 0, atol=1e-4), f"L lower {dev}")
        both_devices(body)

    def test_svd_values_and_recon(self):
        A = np.random.RandomState(5).randn(5, 3).astype("float32")
        def body(dev):
            # Jittor's canonical torch.linalg path returns the reduced (U, S, Vh)
            # decomposition, equivalent to torch full_matrices=False.
            U, S, Vh = torch.linalg.svd(jt.array(A))
            self.ac(np.sort(S.numpy())[::-1], np.linalg.svd(A, compute_uv=False),
                    rtol=1e-3, msg=f"singular values {dev}")
            recon = U.numpy() @ np.diag(S.numpy()) @ Vh.numpy()
            self.ac(recon, A, atol=1e-3, msg=f"U@S@Vh==A {dev}")
        both_devices(body)

    def test_singular_values_via_svd(self):
        # Exercise the singular values through the shared canonical SVD path.
        A = np.random.RandomState(6).randn(4, 6).astype("float32")
        def body(dev):
            S = torch.linalg.svd(jt.array(A))[1]
            self.ac(np.sort(S.numpy())[::-1], np.linalg.svd(A, compute_uv=False),
                    rtol=1e-3, msg=f"singular values via svd {dev}")
        both_devices(body)

    def test_eigh_values(self):
        # eigh returns eigenvalues and eigenvectors; this test consumes the values.
        A = self.spd(4, 7)
        def body(dev):
            w = torch.linalg.eigh(jt.array(A))[0].numpy()
            self.ac(np.sort(w), np.sort(np.linalg.eigvalsh(A)), rtol=1e-3,
                    msg=f"eigh values {dev}")
        both_devices(body)

    def test_qr_recon(self):
        A = np.random.RandomState(8).randn(5, 4).astype("float32")
        def body(dev):
            Q, R = torch.linalg.qr(jt.array(A))
            self.ac(Q.numpy() @ R.numpy(), A, atol=1e-3, msg=f"Q@R==A {dev}")
            self.ac(Q.numpy().T @ Q.numpy(), np.eye(4), atol=1e-3, msg=f"Q orthonormal {dev}")
        both_devices(body)

    def test_matrix_rank(self):
        A = np.random.RandomState(9).randn(5, 3).astype("float32")
        A = np.concatenate([A, A[:, :1]], axis=1)  # rank 3, 4 cols
        def body(dev):
            self.assertEqual(int(torch.linalg.matrix_rank(jt.array(A)).item()), 3,
                             f"matrix_rank {dev}")
        both_devices(body)

    def test_norm(self):
        x = np.random.RandomState(10).randn(3, 4).astype("float32")
        def body(dev):
            self.ac(float(torch.linalg.norm(jt.array(x)).item()),
                    np.linalg.norm(x), rtol=1e-4, msg=f"fro norm {dev}")
            self.ac(float(torch.linalg.norm(jt.array(x[0]), ord=1).item()),
                    np.linalg.norm(x[0], ord=1), rtol=1e-4, msg=f"1-norm {dev}")
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
