#!/usr/bin/env python3
"""Gradient-parity for the non-trivial ops added to jittor-as-torch: confirms the
BACKWARD pass (not just forward) matches real torch. Each op is differentiated w.r.t.
a float input; we compare d(sum(w*out))/d(input) on jittor vs real torch on identical
seeded inputs. Catches scatter-add/gather/reindex backward bugs that forward-only
testing misses.

  grad_ops.py jt  <outdir>     # jittor-as-torch: compute grads, save
  grad_ops.py rt  <outdir>     # real torch: compute grads, save
  grad_ops.py cmp <outdir>     # compare
"""
import os, sys, json
import numpy as np

def make_inputs():
    rng = np.random.RandomState(0)
    return {
        "a":    rng.randn(4, 6).astype("float32"),
        "src":  rng.randn(4, 6).astype("float32"),
        "w":    rng.randn(4, 6).astype("float32"),       # output weighting for a scalar loss
        "idx2": rng.randint(0, 6, size=(4, 6)).astype("int64"),
        "idx1": rng.randint(0, 4, size=(5,)).astype("int64"),
        "vec":  rng.randn(8).astype("float32"),
        "wvec": rng.randn(6).astype("float32"),
    }

# (name, fn) -> fn(T, lib) returns an output tensor differentiable w.r.t. T["a"]
# (or T["vec"]); we backprop sum(out * matching_weight) and read the input grad.
def battery():
    B = []
    add = lambda n, f, src: B.append((n, f, src))
    add("scatter_add",    lambda T, lib: T["a"].scatter_add(1, T["idx2"], T["src"]), "a")
    add("scatter_reduce_sum",  lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "sum"), "a")
    add("scatter_reduce_mean", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "mean"), "a")
    add("scatter_reduce_amax", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "amax"), "a")
    add("gather",         lambda T, lib: T["a"].gather(1, T["idx2"]), "a")
    add("index_add",      lambda T, lib: T["a"].index_add(0, T["idx1"][:4], T["src"]), "a")
    add("diagonal",       lambda T, lib: T["a"].diagonal(), "a")
    add("masked_select",  lambda T, lib: T["a"].masked_select(T["a"] > 0), "a")
    add("sort_vals",      lambda T, lib: T["a"].sort(dim=1)[0], "a")
    add("topk_vals",      lambda T, lib: T["a"].topk(3, dim=1)[0], "a")
    add("flip",           lambda T, lib: T["a"].flip(dims=[0, 1]), "a")
    add("roll",           lambda T, lib: T["a"].roll(2, dims=1), "a")
    add("cumsum",         lambda T, lib: T["a"].cumsum(dim=1), "a")
    add("softmax",        lambda T, lib: lib.nn.functional.softmax(T["a"], dim=1), "a")
    add("logsumexp",      lambda T, lib: T["a"].logsumexp(dim=1), "a")
    add("var",            lambda T, lib: T["a"].var(dim=1), "a")
    add("unfold",         lambda T, lib: T["vec"].unfold(0, 3, 2), "vec")
    add("as_strided",     lambda T, lib: T["vec"].as_strided([3, 3], [2, 1]), "vec")
    return B


def run(side, outdir):
    os.makedirs(outdir, exist_ok=True)
    npz = os.path.join(outdir, "inputs.npz")
    if side == "jt":
        inp = make_inputs(); np.savez(npz, **inp)
    else:
        inp = dict(np.load(npz))
    import torch
    lib = torch
    grads, errors = {}, {}
    for name, fn, srckey in battery():
        try:
            T = {}
            for k, v in inp.items():
                if v.dtype == np.int64:
                    T[k] = torch.from_numpy(v)
                else:
                    t = torch.from_numpy(v.copy())
                    if hasattr(t, "requires_grad_") and k == srckey:
                        t.requires_grad_(True)
                    T[k] = t
            out = fn(T, lib)
            # self-contained scalar loss (no external weight to size-match): pow2 sum
            # exercises the full backward (grad = 2*out * d(out)/d(in)).
            flat = out.reshape(-1)
            loss = (flat * flat).sum()
            xin = T[srckey]
            if hasattr(torch, "jittor"):     # jittor side: use jt.grad
                import jittor as jt
                g = jt.grad(loss, xin)
            else:
                loss.backward(); g = xin.grad
            grads[name] = np.asarray(g.detach().cpu().numpy() if hasattr(g, "detach") else g.numpy(), np.float32)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {str(e).splitlines()[-1][:120]}"
    np.savez(os.path.join(outdir, f"grad_{side}.npz"), **grads)
    json.dump(errors, open(os.path.join(outdir, f"err_{side}.json"), "w"))
    print(f"{side}: grads for {len(grads)} ops, {len(errors)} errored")
    for k, v in errors.items():
        print(f"   ERR {k}: {v}")


def cmp(outdir):
    jt = dict(np.load(os.path.join(outdir, "grad_jt.npz")))
    rt = dict(np.load(os.path.join(outdir, "grad_rt.npz")))
    ej = json.load(open(os.path.join(outdir, "err_jt.json")))
    er = json.load(open(os.path.join(outdir, "err_rt.json")))
    nfail = 0
    for n, _, _ in battery():
        if n in ej or n in er:
            print(f"  {'JT' if n in ej else 'RT'}-ERR {n}: {ej.get(n) or er.get(n)}")
            if n in ej:
                nfail += 1
            continue
        a, b = jt[n], rt[n]
        if a.shape != b.shape:
            print(f"  FAIL {n}: grad shape {a.shape} vs {b.shape}"); nfail += 1; continue
        d = np.abs(a - b).max(); sc = np.abs(b).max() + 1e-9
        tag = "PASS " if d / sc < 1e-4 else "FAIL "
        if d / sc >= 1e-4:
            nfail += 1
        print(f"  {tag} {n:22s} grad max|d|={d:.2e} rel={d/sc:.2e}")
    print(f"\n{'ALL GRADS MATCH' if nfail == 0 else str(nfail)+' GRAD(S) DIVERGE'}")


if __name__ == "__main__":
    mode, outdir = sys.argv[1], sys.argv[2]
    (cmp if mode == "cmp" else lambda o: run(mode, o))(outdir)
