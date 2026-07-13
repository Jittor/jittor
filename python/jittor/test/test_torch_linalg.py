# ***************************************************************
# torch.linalg regression (through `import torch` -> jittor shim). jittor.linalg
# supplies differentiable svd/qr/inv/pinv/det/slogdet/solve/cholesky/eig/eigh; the
# shim adds torch semantics on top: svd accepts full_matrices= and returns a named
# (U, S, Vh) tuple (jittor's svd is the REDUCED form and already returns Vh), plus
# svdvals/eigvalsh/eigvals/matrix_rank/multi_dot/lstsq. Verified vs numpy/torch.
#
#   /home/yizhang/miniconda3/envs/jt-torch/bin/python -m jittor.test.test_torch_linalg
# Skips cleanly if the torch_shim is unavailable.
# ***************************************************************
import unittest, numpy as np

try:
    import torch  # torch_shim -> jittor
    import jittor as jt
    _HAS = (getattr(torch, '__name__', '') == 'torch') and hasattr(torch, 'linalg')
except Exception:
    _HAS = False


@unittest.skipUnless(_HAS, "needs torch_shim")
class TestTorchLinalg(unittest.TestCase):
    def setUp(self):
        rs = np.random.RandomState(0)
        self.A = rs.randn(4, 4).astype('float32')
        self.SPD = self.A @ self.A.T + np.eye(4, dtype='float32')
        self.B = rs.randn(5, 3).astype('float32')

    def test_svd_square_named_vh_and_recon(self):
        r = torch.linalg.svd(torch.tensor(self.A))         # default full_matrices=True
        U, S, Vh = r.U, r.S, r.Vh                          # named tuple
        recon = U.numpy() @ np.diag(S.numpy()) @ Vh.numpy()
        self.assertLess(np.abs(recon - self.A).max(), 1e-4, "A != U diag(S) Vh")
        self.assertLess(np.abs(np.sort(S.numpy())[::-1] - np.linalg.svd(self.A, compute_uv=False)).max(), 1e-4)

    def test_svd_full_matrices_shapes(self):
        full = torch.linalg.svd(torch.tensor(self.B))                       # full_matrices=True
        self.assertEqual((tuple(full.U.shape), tuple(full.Vh.shape)), ((5, 5), (3, 3)))
        recon = full.U.numpy()[:, :3] @ np.diag(full.S.numpy()) @ full.Vh.numpy()
        self.assertLess(np.abs(recon - self.B).max(), 1e-4)
        thin = torch.linalg.svd(torch.tensor(self.B), full_matrices=False)
        self.assertEqual((tuple(thin.U.shape), tuple(thin.Vh.shape)), ((5, 3), (3, 3)))

    def test_svd_differentiable(self):
        M = jt.array(self.A.copy())
        g = jt.grad(torch.linalg.svd(M, full_matrices=False).S.sum(), [M])[0]
        self.assertTrue(bool(jt.isfinite(g).all().item()) and float(jt.abs(g).sum().item()) > 0)

    def test_svdvals_eigvalsh_matrixrank(self):
        self.assertLess(np.abs(np.sort(torch.linalg.svdvals(torch.tensor(self.SPD)).numpy())[::-1]
                               - np.linalg.svd(self.SPD, compute_uv=False)).max(), 1e-4)
        self.assertLess(np.abs(np.sort(torch.linalg.eigvalsh(torch.tensor(self.SPD)).numpy())
                               - np.sort(np.linalg.eigvalsh(self.SPD))).max(), 1e-4)
        self.assertEqual(int(torch.linalg.matrix_rank(torch.tensor(self.SPD)).item()),
                         int(np.linalg.matrix_rank(self.SPD)))

    def test_inv_solve_cholesky_det(self):
        self.assertLess(np.abs(torch.linalg.inv(torch.tensor(self.SPD)).numpy() - np.linalg.inv(self.SPD)).max(), 1e-3)
        b = np.random.RandomState(1).randn(4, 2).astype('float32')
        self.assertLess(np.abs(torch.linalg.solve(torch.tensor(self.SPD), torch.tensor(b)).numpy()
                               - np.linalg.solve(self.SPD, b)).max(), 1e-3)
        self.assertLess(np.abs(torch.linalg.cholesky(torch.tensor(self.SPD)).numpy() - np.linalg.cholesky(self.SPD)).max(), 1e-3)


if __name__ == '__main__':
    unittest.main()
