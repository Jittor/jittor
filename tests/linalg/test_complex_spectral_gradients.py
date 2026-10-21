# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# complex64 SVD backward (`jittor.linalg.complex.complex_svd`): a joint
# reconstruction loss that depends on U, S, and Vh together exercises the
# per-column complex phase-gauge term that a real-valued SVD backward has no
# analogue for (real orthonormal U makes U^T dU exactly skew-symmetric --
# diagonal forced to zero; complex unitary U only makes U^H dU
# skew-HERMITIAN -- diagonal forced purely imaginary, not zero). Testing S
# alone, or |U|**2/|Vh|**2 alone, does not exercise this term (both are
# phase-invariant and never see the coupling); only a loss that mixes U, S,
# and Vh catches a formula that drops it incorrectly.
#
# Ground truth for `loss = Re<(U*S)@Vh, P>` is exact: d(U diag(S) Vh)/dA is
# the identity map in that direction for a reconstruction-style loss, so the
# correct gradient is `P` itself. Verified three independent ways: (1) an
# exact match to `P`, (2) a numpy central-finite-difference oracle on the
# same `np.linalg.svd`-based forward jittor's numpy_code backend uses, and
# (3) an independent real-PyTorch child process (no jittor import) computing
# the same loss via `torch.linalg.svd` autograd.
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import linalg

from _helpers import capability as _test_capability
from _helpers.cupy_bridge import cuda_numpy_code_available
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy


def _to_complex64_var(a):
    return jt.array(a.astype("complex64"))


def _reconstruction_loss_var(A):
    u, s, vh = linalg.svd(A)
    recon = jt.matmul(u * s.unsqueeze(-2), vh)
    return recon, u, s, vh


def _finite_diff_grad(loss_fn, a, eps=1e-3):
    """Central-difference dL/dA (Wirtinger cotangent), perturbing Re/Im
    independently, on a numpy complex128 array. Ground-truth oracle,
    independent of the analytic backward formula under test."""
    g = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        orig = a[idx]
        a[idx] = orig + eps
        lp = loss_fn(a)
        a[idx] = orig - eps
        lm = loss_fn(a)
        re = (lp - lm) / (2 * eps)
        a[idx] = orig + eps * 1j
        lp = loss_fn(a)
        a[idx] = orig - eps * 1j
        lm = loss_fn(a)
        im = (lp - lm) / (2 * eps)
        a[idx] = orig
        g[idx] = re + 1j * im
    return g


def _torch_reconstruction_grad(a, p):
    """Independent oracle: a real PyTorch process (no jittor import) computing
    dL/dA for the same `loss = Re<(U*S)@Vh, P>` via `torch.linalg.svd`
    autograd. Arrays are passed through temp .npy files, not inlined as
    source, so this scales to any shape/batch.

    Runs under `_real_torch_python()`'s interpreter, NOT `sys.executable`:
    jittor's own compat layer installs a top-level `torch` package (the
    Torch shim, `jittor-torch`), so a plain `sys.executable -c "import
    torch"` in this repo's own interpreter silently resolves to jittor's
    shim (same site-packages), not upstream PyTorch -- defeating the whole
    point of an "independent" oracle. This needs a genuinely separate
    interpreter/site-packages with real `torch` installed and no jittor on
    its path.
    """
    python = _real_torch_python()
    with tempfile.TemporaryDirectory() as td:
        a_path = Path(td) / "a.npy"
        p_path = Path(td) / "p.npy"
        out_path = Path(td) / "g.npy"
        np.save(a_path, a)
        np.save(p_path, p)
        script = f"""
import numpy as np, torch
assert not hasattr(torch, "compat"), "resolved jittor's torch shim, not real PyTorch"
a = np.load({str(a_path)!r})
p = np.load({str(p_path)!r})
A = torch.tensor(a, requires_grad=True)
P = torch.tensor(p)
u, s, vh = torch.linalg.svd(A, full_matrices=False)
recon = (u * s.unsqueeze(-2)) @ vh
loss = (recon * P.conj()).real.sum()
loss.backward()
np.save({str(out_path)!r}, A.grad.numpy())
"""
        subprocess.run([python, "-c", script], check=True,
                        capture_output=True, timeout=120, env=_oracle_env())
        return np.load(out_path)


def _oracle_env():
    """The oracle's own environment: the suite's PYTHONPATH puts Jittor's
    torch shim first, which would make the "real" interpreter import it."""
    return {k: v for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "JITTOR_TORCH_SHIM")}


def _real_torch_python():
    """Locate a Python interpreter with real (non-shim) PyTorch installed,
    isolated from this repo's own `jittor`/`jittor-torch`. Override with
    ``JITTOR_TEST_REAL_TORCH_PYTHON``; falls back to a conventional dev venv
    path. Raises if neither is usable -- callers should catch and skip."""
    candidates = []
    override = os.environ.get("JITTOR_TEST_REAL_TORCH_PYTHON")
    if override:
        candidates.append(override)
    candidates.append("/opt/real-torch-venv/bin/python3")
    for python in candidates:
        if not Path(python).exists():
            continue
        probe = subprocess.run(
            [python, "-c", "import torch; assert not hasattr(torch, 'compat')"],
            capture_output=True, timeout=60, env=_oracle_env())
        if probe.returncode == 0:
            return python
    raise RuntimeError(
        "no interpreter with real (non-shim) PyTorch found; set "
        "JITTOR_TEST_REAL_TORCH_PYTHON to one, or skip this test")


@_test_preserve_policy(jt, 'use_cuda')
class _Mixin:
    use_cuda = 0

    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self.use_cuda))

    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

    # -------------------------------------------------- joint reconstruction

    def _check_joint_reconstruction(self, m, n, batch, seed):
        rng = np.random.RandomState(seed)
        shape = ((batch,) if batch else ()) + (m, n)
        a = (rng.randn(*shape) + 1j * rng.randn(*shape))
        p = (rng.randn(*shape) + 1j * rng.randn(*shape))

        A = _to_complex64_var(a)
        P = _to_complex64_var(p)
        recon, u, s, vh = _reconstruction_loss_var(A)
        self.assertTrue("complex" in str(u.dtype))
        loss = (recon * jt.conj(P)).real.sum()
        gA = jt.grad(loss, A).numpy().astype("complex128")

        p64 = p.astype("complex64").astype("complex128")
        np.testing.assert_allclose(gA, p64, atol=2e-4, rtol=2e-4,
                                    err_msg="dL/dA should equal P exactly for "
                                            "a reconstruction loss")

    def test_joint_reconstruction_square(self):
        self._check_joint_reconstruction(4, 4, 0, seed=0)

    def test_joint_reconstruction_tall(self):
        self._check_joint_reconstruction(5, 3, 0, seed=1)

    def test_joint_reconstruction_wide(self):
        self._check_joint_reconstruction(3, 5, 0, seed=2)

    def test_joint_reconstruction_batched(self):
        self._check_joint_reconstruction(4, 3, 2, seed=3)

    def test_joint_reconstruction_matches_finite_difference(self):
        rng = np.random.RandomState(4)
        m, n = 4, 3
        a64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")
        p64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")

        A = _to_complex64_var(a64)
        P = _to_complex64_var(p64)
        recon, u, s, vh = _reconstruction_loss_var(A)
        loss = (recon * jt.conj(P)).real.sum()
        gA = jt.grad(loss, A).numpy().astype("complex128")

        def loss_fn(a_np):
            u_np, s_np, vh_np = np.linalg.svd(a_np, full_matrices=False)
            recon_np = (u_np * s_np[..., np.newaxis, :]) @ vh_np
            return float(np.real(np.sum(np.conj(recon_np) * p64.astype("complex128"))))

        g_fd = _finite_diff_grad(loss_fn, a64.astype("complex128").copy(), eps=1e-4)
        np.testing.assert_allclose(gA, g_fd, atol=5e-3, rtol=5e-3)

    def test_joint_reconstruction_matches_independent_pytorch(self):
        try:
            _real_torch_python()
        except RuntimeError as e:
            self.skipTest(str(e))
        rng = np.random.RandomState(5)
        m, n = 4, 3
        a64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")
        p64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")

        A = _to_complex64_var(a64)
        P = _to_complex64_var(p64)
        recon, u, s, vh = _reconstruction_loss_var(A)
        loss = (recon * jt.conj(P)).real.sum()
        gA = jt.grad(loss, A).numpy().astype("complex128")

        g_torch = _torch_reconstruction_grad(
            a64.astype("complex128"), p64.astype("complex128"))
        np.testing.assert_allclose(gA, g_torch, atol=5e-3, rtol=5e-3)

    # ------------------------------------------- isolated branches (cheap
    # regression; the doc's own warning is that these ALONE don't cover the
    # joint bug above, since each is phase-invariant on its own)

    def test_s_only_matches_finite_difference(self):
        rng = np.random.RandomState(6)
        m, n = 4, 3
        a64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")
        A = _to_complex64_var(a64)
        u, s, vh = linalg.svd(A)
        loss = (s.real ** 2).sum()
        gA = jt.grad(loss, A).numpy().astype("complex128")

        def loss_fn(a_np):
            s_np = np.linalg.svd(a_np, full_matrices=False, compute_uv=False)
            return float(np.sum(s_np ** 2))

        g_fd = _finite_diff_grad(loss_fn, a64.astype("complex128").copy(), eps=1e-4)
        np.testing.assert_allclose(gA, g_fd, atol=5e-3, rtol=5e-3)

    def test_u_only_matches_finite_difference(self):
        rng = np.random.RandomState(7)
        m, n = 4, 3
        a64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")
        A = _to_complex64_var(a64)
        u, s, vh = linalg.svd(A)
        loss = (jt.abs(u) ** 2).sum()
        gA = jt.grad(loss, A).numpy().astype("complex128")

        def loss_fn(a_np):
            u_np = np.linalg.svd(a_np, full_matrices=False)[0]
            return float(np.sum(np.abs(u_np) ** 2))

        g_fd = _finite_diff_grad(loss_fn, a64.astype("complex128").copy(), eps=1e-4)
        np.testing.assert_allclose(gA, g_fd, atol=5e-3, rtol=5e-3)

    def test_v_only_matches_finite_difference(self):
        rng = np.random.RandomState(8)
        m, n = 4, 3
        a64 = (rng.randn(m, n) + 1j * rng.randn(m, n)).astype("complex64")
        A = _to_complex64_var(a64)
        u, s, vh = linalg.svd(A)
        loss = (jt.abs(vh) ** 2).sum()
        gA = jt.grad(loss, A).numpy().astype("complex128")

        def loss_fn(a_np):
            vh_np = np.linalg.svd(a_np, full_matrices=False)[2]
            return float(np.sum(np.abs(vh_np) ** 2))

        g_fd = _finite_diff_grad(loss_fn, a64.astype("complex128").copy(), eps=1e-4)
        np.testing.assert_allclose(gA, g_fd, atol=5e-3, rtol=5e-3)


# complex64 linalg runs through jt.numpy_code, and py_converter hands that
# callback `cupy` instead of `numpy` when use_cuda is on -- without CuPy the
# operator raises from inside execution (see test_complex64_linalg.py for the
# full rationale on why this must be a collection-time skip, not a per-test
# try/except).
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "no cuda found")
@unittest.skipIf(not cuda_numpy_code_available(),
                 "CUDA numpy-code operators need CuPy; it is not installed")
class TestComplexSpectralGradientsCUDA(_Mixin, unittest.TestCase):
    use_cuda = 1


class TestComplexSpectralGradientsCPU(_Mixin, unittest.TestCase):
    use_cuda = 0


if __name__ == "__main__":
    unittest.main()
