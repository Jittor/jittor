"""Native Jittor numerics against independent NumPy references.

Never Jittor-vs-Jittor: every expected value is computed by NumPy (or an
explicit closed form), so a shared defect cannot make both sides agree.
float64 cases are chosen to fail if the work is silently done in float32.
"""
import argparse, json
import numpy as np
import jittor as jt

R = np.random.RandomState(20260909)
CASES = []
def case(name, fn, ref, tol=2e-6):
    CASES.append((name, fn, ref, tol))

A = R.rand(4, 5).astype("float32") + 0.5
B = R.rand(4, 5).astype("float32") + 0.5
M = R.rand(6, 7).astype("float32")
N = R.rand(7, 5).astype("float32")
C3 = R.rand(3, 4, 5).astype("float32")
SPD = (lambda x: x @ x.T + 5 * np.eye(5))(R.rand(5, 5)).astype("float32")

# --- elementwise unary -------------------------------------------------------
def resolve(name):
    """Public spelling first; record where it actually lives."""
    for holder in (jt, jt.ops, jt.misc, jt.nn):
        fn = getattr(holder, name, None)
        if callable(fn):
            return fn, holder.__name__
    return None, None

for nm, nf in [("exp", np.exp), ("log", np.log), ("sqrt", np.sqrt),
               ("abs", np.abs), ("sin", np.sin), ("cos", np.cos),
               ("tan", np.tan), ("tanh", np.tanh),
               ("floor", np.floor), ("ceil", np.ceil), ("sign", np.sign)]:
    jf, where = resolve(nm)
    if jf is None:
        CASES.append(("unary/"+nm, None, None, 0.0)); continue
    case("unary/%s[%s]" % (nm, where), (lambda jf=jf: jf(jt.array(A))), (lambda nf=nf: nf(A)))
case("unary/sigmoid", lambda: jt.sigmoid(jt.array(A)), lambda: 1/(1+np.exp(-A)))
case("unary/relu", lambda: jt.nn.relu(jt.array(A-0.8)), lambda: np.maximum(A-0.8, 0))

# --- binary ------------------------------------------------------------------
for nm, op in [("add", lambda x,y: x+y), ("sub", lambda x,y: x-y),
               ("mul", lambda x,y: x*y), ("div", lambda x,y: x/y)]:
    case("binary/"+nm, (lambda op=op: op(jt.array(A), jt.array(B))),
         (lambda op=op: op(A, B)))
case("binary/pow", lambda: jt.array(A)**2.5, lambda: A**2.5)
case("binary/maximum", lambda: jt.maximum(jt.array(A), jt.array(B)), lambda: np.maximum(A,B))
case("binary/minimum", lambda: jt.minimum(jt.array(A), jt.array(B)), lambda: np.minimum(A,B))

# --- reductions --------------------------------------------------------------
for nm, jf, nf in [("sum", jt.sum, np.sum), ("mean", jt.mean, np.mean),
                   ("max", jt.max, np.max), ("min", jt.min, np.min),
                   ("prod", jt.prod, np.prod)]:
    case("reduce/%s-all" % nm, (lambda jf=jf: jf(jt.array(A))), (lambda nf=nf: nf(A)))
    for ax in (0, 1):
        case("reduce/%s-ax%d" % (nm, ax),
             (lambda jf=jf, ax=ax: jf(jt.array(A), ax)),
             (lambda nf=nf, ax=ax: nf(A, axis=ax)))
case("reduce/argmax-ax1", lambda: jt.argmax(jt.array(A), 1)[0], lambda: np.argmax(A, 1))
case("reduce/argmin-ax1", lambda: jt.argmin(jt.array(A), 1)[0], lambda: np.argmin(A, 1))
case("reduce/cumsum", lambda: jt.cumsum(jt.array(A), 1), lambda: np.cumsum(A, 1))
case("reduce/std", lambda: jt.array(A).std(), lambda: A.std(ddof=1), 1e-5)
case("reduce/norm", lambda: jt.norm(jt.array(A)), lambda: np.linalg.norm(A), 1e-5)

# --- matmul ------------------------------------------------------------------
case("matmul/2d", lambda: jt.matmul(jt.array(M), jt.array(N)), lambda: M @ N, 1e-5)
case("matmul/bmm", lambda: jt.bmm(jt.array(C3), jt.array(C3.transpose(0,2,1))),
     lambda: C3 @ C3.transpose(0,2,1), 1e-5)

# --- structural --------------------------------------------------------------
case("shape/transpose", lambda: jt.transpose(jt.array(C3), (1,0,2)), lambda: C3.transpose(1,0,2))
case("shape/reshape", lambda: jt.reshape(jt.array(A), (5,4)), lambda: A.reshape(5,4))
case("shape/concat", lambda: jt.concat([jt.array(A), jt.array(B)], 0), lambda: np.concatenate([A,B],0))
case("shape/stack", lambda: jt.stack([jt.array(A), jt.array(B)], 0), lambda: np.stack([A,B],0))
case("shape/flip", lambda: jt.flip(jt.array(A), 1), lambda: A[:, ::-1])
case("shape/slice", lambda: jt.array(C3)[:, 1:3, ::2], lambda: C3[:, 1:3, ::2])
case("shape/where", lambda: jt.ternary(jt.array(A) > 0.9, jt.array(A), jt.array(B)),
     lambda: np.where(A > 0.9, A, B))

# --- linalg ------------------------------------------------------------------
case("linalg/inv", lambda: jt.linalg.inv(jt.array(SPD)), lambda: np.linalg.inv(SPD), 1e-4)
case("linalg/det", lambda: jt.linalg.det(jt.array(SPD)), lambda: np.linalg.det(SPD), 1e-3)
case("linalg/solve", lambda: jt.linalg.solve(jt.array(SPD), jt.array(SPD[:, :1])),
     lambda: np.linalg.solve(SPD, SPD[:, :1]), 1e-4)
case("linalg/cholesky-recon",
     lambda: (lambda L: jt.matmul(L, L.transpose()))(jt.linalg.cholesky(jt.array(SPD))),
     lambda: SPD, 1e-4)
case("linalg/eigh-values", lambda: jt.linalg.eigh(jt.array(SPD))[0],
     lambda: np.linalg.eigh(SPD)[0], 1e-4)
case("linalg/svd-values", lambda: jt.linalg.svd(jt.array(M))[1],
     lambda: np.linalg.svd(M, compute_uv=False), 1e-4)

# --- fft ---------------------------------------------------------------------
F = R.rand(4, 8).astype("float32")
case("fft/rfft-real", lambda: jt.fft.rfft(jt.array(F)).real, lambda: np.fft.rfft(F).real, 1e-4)
case("fft/rfft-imag", lambda: jt.fft.rfft(jt.array(F)).imag, lambda: np.fft.rfft(F).imag, 1e-4)
case("fft/irfft-roundtrip", lambda: jt.fft.irfft(jt.fft.rfft(jt.array(F)), 8), lambda: F, 1e-4)

# --- nn functional -----------------------------------------------------------
X = R.rand(2, 3, 8, 8).astype("float32")
W = R.rand(4, 3, 3, 3).astype("float32")
def np_conv2d(x, w):
    n, c, h, ww = x.shape; o, _, kh, kw = w.shape
    oh, ow = h - kh + 1, ww - kw + 1
    out = np.zeros((n, o, oh, ow), dtype="float64")
    for b in range(n):
        for f in range(o):
            for i in range(oh):
                for j in range(ow):
                    out[b,f,i,j] = (x[b,:,i:i+kh,j:j+kw].astype("float64")
                                    * w[f].astype("float64")).sum()
    return out
case("nn/conv2d", lambda: jt.nn.conv2d(jt.array(X), jt.array(W)),
     lambda: np_conv2d(X, W), 1e-4)
case("nn/softmax", lambda: jt.nn.softmax(jt.array(A), 1),
     lambda: (lambda e: e/e.sum(1, keepdims=True))(np.exp(A - A.max(1, keepdims=True))))
case("nn/log_softmax", lambda: jt.nn.log_softmax(jt.array(A), 1),
     lambda: A - A.max(1,keepdims=True) - np.log(np.exp(A-A.max(1,keepdims=True)).sum(1,keepdims=True)), 1e-5)
def np_layernorm(x, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)
case("nn/layer_norm", lambda: jt.nn.layer_norm(jt.array(A), (5,)),
     lambda: np_layernorm(A.astype("float64")), 1e-4)

# --- autograd ----------------------------------------------------------------
def grad_exp():
    x = jt.array(A); y = (jt.exp(x) * 2).sum()
    return jt.grad(y, x)
case("grad/exp", grad_exp, lambda: 2*np.exp(A), 1e-5)
def grad_matmul():
    x = jt.array(M); y = jt.matmul(x, jt.array(N)).sum()
    return jt.grad(y, x)
case("grad/matmul", grad_matmul, lambda: np.ones((6,5)) @ N.T, 1e-5)

# --- float64 must not be silently narrowed ----------------------------------
D = np.array([[1/3, 1e-15], [1 + 2**-40, np.pi]], dtype="float64")
case("f64/identity", lambda: jt.array(D) * 1.0, lambda: D, 1e-15)
case("f64/sum", lambda: jt.array(D).sum(), lambda: D.sum(), 1e-15)
DM = R.rand(64, 64).astype("float64")
case("f64/matmul", lambda: jt.matmul(jt.array(DM), jt.array(DM)),
     lambda: DM @ DM, 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", required=True)
    opt = ap.parse_args()
    jt.flags.use_cuda = 1 if opt.device == "cuda" else 0
    rows = []
    for name, fn, ref, tol in CASES:
        if fn is None:
            rows.append({"name": name, "status": "MISSING-PUBLIC-NAME"}); continue
        try:
            got = fn()
            got = got.numpy() if isinstance(got, jt.Var) else np.asarray(got)
            exp = np.asarray(ref())
            if got.shape != exp.shape:
                rows.append({"name": name, "status": "shape",
                             "detail": "%s vs %s" % (got.shape, exp.shape)}); continue
            scale = max(float(np.abs(exp).max()), 1e-6)
            err = float(np.abs(got.astype("float64") - exp.astype("float64")).max() / scale)
            rows.append({"name": name, "status": "ok" if err <= tol else "PRECISION",
                         "err": err, "tol": tol,
                         "dtype": str(got.dtype) + "/" + str(exp.dtype)})
        except Exception as exc:               # noqa: BLE001 -- classification
            rows.append({"name": name, "status": "raised",
                         "detail": type(exc).__name__ + ": " + str(exc).splitlines()[0][:200]})
    with open(opt.out, "w") as fh: json.dump(rows, fh, indent=1)
    from collections import Counter
    print(opt.device, dict(Counter(r["status"] for r in rows)))
    for r in rows:
        if r["status"] != "ok": print("  ", r["name"], r["status"], r.get("detail", r.get("err")))
main()
