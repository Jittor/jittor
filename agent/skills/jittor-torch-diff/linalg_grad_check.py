"""Finite-difference checker for jittor.linalg backward passes.

Catches silent-wrong / crashing gradients in svd/eigh/qr/solve/cholesky/inv/...
Ran on a CUDA box (cscg104) it found 3 distinct bugs in one sweep:
  - svd  V-gradient: wrong-axis contraction (silent) + m<n crash   (2c85c570)
  - solve d/db: gradient wrt RHS was a stub writing 0 (silent)     (52d71415)
  - qr: hardcoded square; non-square forward shape + tall backward (52d71415)

KEY GOTCHAS baked in (each cost a debugging cycle to discover):
  1. jittor's default dtype is float32. `jt.array(float64_ndarray)` DOWNCASTS to
     float32, and `.float64()` afterwards can't recover the lost precision.
     => never FD inside jittor. Use a pure-numpy float64 reference for FD and
        compare it to jittor's analytic grad (float32, rel ~1e-6 is fine).
  2. SVD/eigh eigenvectors have a GAUGE (sign/rotation) freedom. The backward
     formula deliberately computes only the gauge-INVARIANT gradient (torch does
     too), so a gauge-VARIANT loss like sum(W*U) "fails" FD even when correct.
     => test U/V via gauge-invariant projectors: U Uᵀ, Vᵀ V, V diag(g) Vᵀ.
        Singular values / eigenvalues are gauge-free -> test directly.
  3. cholesky/eigh take a SYMMETRIC input; their backward returns a symmetric
     gradient. Per-entry FD (independent perturbation) breaks symmetry and
     mismatches by ~2x on off-diagonals. => use a symmetric directional
     derivative: compare sum(grad*H) to (L(x+εH)-L(x-εH))/2ε with H=Hᵀ.

Run on a box with jittor importable (CPU is fine; svd etc. use numpy_code on
host). use_cuda is irrelevant for these (the backward_code runs in numpy/cupy).

    PYTHONPATH=<dev>/python python linalg_grad_check.py
"""
import numpy as np
import jittor as jt
from jittor import linalg
from jittor.nn import ComplexNumber

rng = np.random.RandomState(0)
TOL = 2e-4
_n_fail = 0

def _unwrap(g):
    return g[0] if isinstance(g, (list, tuple)) else g

def _report(name, rel, extra=""):
    global _n_fail
    ok = rel < TOL
    _n_fail += (not ok)
    print(f"[{name}] {'PASS' if ok else 'FAIL'} rel={rel:.2e} {extra}")

def check(name, jt_fn, np_fn, x0):
    """Per-entry FD (for single-valued / gauge-free functions)."""
    try:
        x = jt.array(x0); g = _unwrap(jt.grad(jt_fn(x), [x])).numpy()
    except Exception as e:
        print(f"[{name}] CRASH {type(e).__name__}: {e}"); globals().__setitem__('_n_fail', _n_fail+1); return
    gf = np.zeros_like(x0, dtype=np.float64); it = np.nditer(x0, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index
        xp = x0.astype(np.float64).copy(); xp[i] += 1e-6
        xm = x0.astype(np.float64).copy(); xm[i] -= 1e-6
        gf[i] = (np_fn(xp) - np_fn(xm)) / 2e-6; it.iternext()
    err = float(np.abs(g - gf).max()); _report(name, err / (float(np.abs(gf).max()) + 1e-12))

def check_dir(name, jt_fn, np_fn, x0, H):
    """Directional derivative along H (use H=Hᵀ for symmetric-input fns)."""
    try:
        x = jt.array(x0); g = _unwrap(jt.grad(jt_fn(x), [x])).numpy()
    except Exception as e:
        print(f"[{name}] CRASH {type(e).__name__}: {e}"); globals().__setitem__('_n_fail', _n_fail+1); return
    ad = float((g * H).sum()); xb = x0.astype(np.float64)
    fd = (np_fn(xb + 1e-6 * H) - np_fn(xb - 1e-6 * H)) / 2e-6
    _report(name, abs(ad - fd) / (abs(fd) + 1e-9), f"analytic={ad:.4f} fd={fd:.4f}")

def _f32(a): return a.astype(np.float32)

def check_cplx(name, jt_L, np_L, R0, I0):
    """Convention-free complex backward check: ComplexNumber is stored as a real
    (...,2) tensor and autodiff runs on that storage, so FD a REAL scalar loss
    over the real/imag leaves and compare to jt.grad on each."""
    try:
        r = jt.array(R0); im = jt.array(I0); L = jt_L(r, im)
        gr = _unwrap(jt.grad(L, [r])).numpy(); gi = _unwrap(jt.grad(L, [im])).numpy()
    except Exception as e:
        print(f"[{name}] CRASH {type(e).__name__}: {e}"); globals().__setitem__('_n_fail', _n_fail+1); return
    def fd(which):
        g = np.zeros_like(R0, dtype=np.float64); it = np.nditer(R0, flags=['multi_index'])
        while not it.finished:
            i = it.multi_index
            Rp, Ip = R0.astype(np.float64).copy(), I0.astype(np.float64).copy()
            Rm, Im = R0.astype(np.float64).copy(), I0.astype(np.float64).copy()
            if which == 'r': Rp[i] += 1e-6; Rm[i] -= 1e-6
            else:            Ip[i] += 1e-6; Im[i] -= 1e-6
            g[i] = (np_L(Rp, Ip) - np_L(Rm, Im)) / 2e-6; it.iternext()
        return g
    fdr, fdi = fd('r'), fd('i')
    er = float(np.abs(gr - fdr).max()); ei = float(np.abs(gi - fdi).max())
    den = max(float(np.abs(fdr).max()), float(np.abs(fdi).max())) + 1e-12
    _report(name, max(er, ei) / den, f"er={er:.1e} ei={ei:.1e}")

def main():
    # ---- gauge-free, per-entry FD ----
    A = _f32(rng.randn(3, 3)) + 3 * np.eye(3, dtype=np.float32)
    W = rng.randn(3, 3)
    check("inv", lambda x: (jt.array(_f32(W)) * linalg.inv(x)).sum(),
                 lambda x: float((W * np.linalg.inv(x)).sum()), A)
    for (m, n) in [(3, 3), (4, 2), (2, 4)]:
        Ar = _f32(rng.randn(m, n)); Wp = rng.randn(n, m)
        check(f"pinv {m}x{n}", lambda x, Wp=Wp: (jt.array(_f32(Wp)) * linalg.pinv(x)).sum(),
                               lambda x, Wp=Wp: float((Wp * np.linalg.pinv(x)).sum()), Ar)
    check("det", lambda x: (2.0 * linalg.det(x)).sum(), lambda x: float(2.0 * np.linalg.det(x)), A)
    def jt_slog(x):
        r = linalg.slogdet(x); return (1.0 * (r[1] if isinstance(r, (list, tuple)) else r.logabsdet)).sum()
    check("slogdet", jt_slog, lambda x: float(np.linalg.slogdet(x)[1]), A)

    b0 = _f32(rng.randn(3, 2)); Ws = rng.randn(3, 2)
    check("solve d/da", lambda x: (jt.array(_f32(Ws)) * linalg.solve(x, jt.array(b0))).sum(),
                        lambda x: float((Ws * np.linalg.solve(x, b0)).sum()), A)
    check("solve d/db", lambda x: (jt.array(_f32(Ws)) * linalg.solve(jt.array(A), x)).sum(),
                        lambda x: float((Ws * np.linalg.solve(A, x)).sum()), b0)

    # ---- symmetric-input: directional derivative along symmetric H ----
    M = _f32(rng.randn(4, 4)); S = _f32(M @ M.T + 4 * np.eye(4)); Wc = rng.randn(4, 4)
    H = rng.randn(4, 4); H = (H + H.T) / 2
    check_dir("cholesky", lambda x: (jt.array(_f32(Wc)) * linalg.cholesky(x)).sum(),
                          lambda x: float((Wc * np.linalg.cholesky(x)).sum()), S, H)
    Sy = _f32((M + M.T) / 2); cw = rng.randn(4); g_ = rng.randn(4); WP = rng.randn(4, 4)
    check_dir("eigh eigenvalues", lambda x: (jt.array(_f32(cw)) * linalg.eigh(x)[0]).sum(),
                                  lambda x: float((cw * np.linalg.eigh(x)[0]).sum()), Sy, H)
    def jt_eigv(x):
        w, v = linalg.eigh(x); return (jt.array(_f32(WP)) * ((v * jt.array(_f32(g_))) @ v.transpose())).sum()
    def np_eigv(x):
        w, v = np.linalg.eigh(x); return float((WP * ((v * g_) @ v.T)).sum())
    check_dir("eigh eigenvec-proj", jt_eigv, np_eigv, Sy, H)

    # ---- svd: gauge-invariant projector losses (s, U Uᵀ, Vᵀ V) ----
    def svd_proj(m, n):
        x0 = _f32(rng.randn(m, n)); WU = rng.randn(m, m); WV = rng.randn(n, n); ws = rng.randn(min(m, n))
        def jt_fn(x):
            u, s, v = linalg.svd(x)
            return ((jt.array(_f32(WU)) * (u @ u.transpose())).sum()
                    + (jt.array(_f32(WV)) * (v.transpose() @ v)).sum()
                    + (jt.array(_f32(ws)) * s).sum())
        def np_fn(x):
            u, s, vh = np.linalg.svd(x.astype(np.float64), full_matrices=0)
            return float((WU * (u @ u.T)).sum() + (WV * (vh.T @ vh)).sum() + (ws * s).sum())
        check(f"svd {m}x{n} (proj)", jt_fn, np_fn, x0)
    for (m, n) in [(5, 3), (4, 4), (3, 5)]:
        svd_proj(m, n)

    # ---- qr: reconstruction-style loss, tall/square (m>=n) ----
    for (m, n) in [(3, 3), (4, 3), (6, 4)]:
        Aq = _f32(rng.randn(m, n)); k = min(m, n); Wq = rng.randn(m, k); Wr = rng.randn(k, n)
        check(f"qr {m}x{n}",
              lambda x, Wq=Wq, Wr=Wr: (jt.array(_f32(Wq)) * linalg.qr(x)[0]).sum() + (jt.array(_f32(Wr)) * linalg.qr(x)[1]).sum(),
              lambda x, Wq=Wq, Wr=Wr: float((Wq * np.linalg.qr(x)[0]).sum() + (Wr * np.linalg.qr(x)[1]).sum()), Aq)

    # ---- complex paths (ComplexNumber). eig/svd backward honestly raise; check inv/qr ----
    N = 4
    Ri = _f32(rng.randn(N, N)) + 3 * np.eye(N, dtype=np.float32); Ii = _f32(rng.randn(N, N))
    Wr = rng.randn(N, N); Wi = rng.randn(N, N)
    check_cplx("complex_inv",
        lambda r, im: (jt.array(_f32(Wr)) * linalg.complex_inv(ComplexNumber(r, im)).real
                       + jt.array(_f32(Wi)) * linalg.complex_inv(ComplexNumber(r, im)).imag).sum(),
        lambda R, I: float((Wr * np.linalg.inv(R + 1j * I).real + Wi * np.linalg.inv(R + 1j * I).imag).sum()),
        Ri, Ii)
    Rq = _f32(rng.randn(N, N)); Iq = _f32(rng.randn(N, N))
    Wqr, Wqi, Wrr, Wri = rng.randn(N, N), rng.randn(N, N), rng.randn(N, N), rng.randn(N, N)
    def jt_cqr(r, im):
        q, rr = linalg.complex_qr(ComplexNumber(r, im))
        return (jt.array(_f32(Wqr)) * q.real + jt.array(_f32(Wqi)) * q.imag
                + jt.array(_f32(Wrr)) * rr.real + jt.array(_f32(Wri)) * rr.imag).sum()
    def np_cqr(R, I):
        Q, RR = np.linalg.qr(R + 1j * I)
        return float((Wqr * Q.real + Wqi * Q.imag + Wrr * RR.real + Wri * RR.imag).sum())
    check_cplx("complex_qr", jt_cqr, np_cqr, Rq, Iq)

    print(f"\n{'ALL PASS' if _n_fail == 0 else str(_n_fail) + ' FAIL'}")
    return _n_fail

if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
