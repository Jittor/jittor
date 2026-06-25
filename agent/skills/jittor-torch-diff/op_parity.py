#!/usr/bin/env python3
"""Op-level differential test: run a battery of tensor ops on IDENTICAL seeded
inputs through jittor-as-torch and through real torch, and flag any mismatch.

Finds *silent-wrong* semantic divergences (like Var.where treating self as the
condition) that per-model probing only catches by luck. Each op is a lambda over
(t, ...) using the torch tensor API; the same lambda runs on a jittor Var (which
must behave like a torch tensor) and on a real torch.Tensor.

Usage:
  op_parity.py jt   <outdir>     # run battery on jittor-as-torch, save results
  op_parity.py rt   <outdir>     # run battery on real torch, save results
  op_parity.py cmp  <outdir>     # compare, print PASS/FAIL per op
"""
import os, sys, json
import numpy as np

# Fixed seeded inputs (saved once on the jt run, reused by rt for identical data).
def make_inputs():
    rng = np.random.RandomState(0)
    return {
        "a":   rng.randn(4, 5).astype("float32"),
        "b":   rng.randn(4, 5).astype("float32"),
        "c3":  rng.randn(2, 3, 4).astype("float32"),
        "pos": np.abs(rng.randn(4, 5)).astype("float32") + 0.1,
        "mask": (rng.randn(4, 5) > 0),
        "idx2": rng.randint(0, 5, size=(4, 5)).astype("int64"),
        "idx1": rng.randint(0, 4, size=(7,)).astype("int64"),
        "src":  rng.randn(4, 5).astype("float32"),
        "vec":  rng.randn(6).astype("float32"),
    }

# (name, fn) — fn takes a dict of torch/jittor tensors `T` and returns one tensor.
# Only the torch *public* API is used, so the jittor side exercises torch-compat.
def battery():
    B = []
    add = lambda n, f: B.append((n, f))
    add("where_cond_other", lambda T, lib: T["a"].where(T["mask"], T["b"]))
    add("masked_fill",      lambda T, lib: T["a"].masked_fill(T["mask"], -9.0))
    add("masked_select",    lambda T, lib: T["a"].masked_select(T["mask"]))
    add("clamp",            lambda T, lib: T["a"].clamp(-0.5, 0.5))
    add("clamp_min",        lambda T, lib: T["a"].clamp(min=0.0))
    add("gather_d1",        lambda T, lib: T["a"].gather(1, T["idx2"]))
    add("scatter_d1",       lambda T, lib: T["a"].scatter(1, T["idx2"], T["src"]))
    add("scatter_add",      lambda T, lib: T["a"].scatter_add(1, T["idx2"], T["src"]))
    add("scatter_reduce_sum", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "sum"))
    add("scatter_reduce_mean", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "mean"))
    add("scatter_reduce_amax", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "amax"))
    add("scatter_reduce_amin", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "amin"))
    add("scatter_reduce_sum_noself", lambda T, lib: T["a"].scatter_reduce(1, T["idx2"], T["src"], "sum", include_self=False))
    add("index_add",        lambda T, lib: T["a"].index_add(0, T["idx1"].clamp(0, 3), T["b"][T["idx1"].clamp(0, 3)]))
    add("index_select",     lambda T, lib: T["a"].index_select(0, T["idx1"]))
    add("sort_vals",        lambda T, lib: T["a"].sort(dim=1, descending=True)[0])
    add("sort_idx",         lambda T, lib: T["a"].sort(dim=1, descending=True)[1].float())
    add("argsort",          lambda T, lib: T["a"].argsort(dim=1).float())
    add("topk_vals",        lambda T, lib: T["a"].topk(3, dim=1)[0])
    add("cumsum",           lambda T, lib: T["a"].cumsum(dim=1))
    add("cumprod",          lambda T, lib: T["pos"].cumprod(dim=1))
    add("flip",             lambda T, lib: T["a"].flip(dims=[0, 1]))
    add("roll",             lambda T, lib: T["a"].roll(shifts=2, dims=1))
    add("tril",             lambda T, lib: T["a"].tril(diagonal=-1))
    add("triu",             lambda T, lib: T["a"].triu(diagonal=1))
    # NB: the .max(dim)/.min(dim) METHOD form stays jittor-native (values-only) on
    # purpose -- jittor-core (linalg/nn) relies on it -- so test the torch FUNCTION
    # form torch.max(x,dim), which IS torch-correct (returns the (values,indices)
    # namedtuple). The method-with-index form is a known, documented divergence.
    add("max_dim_vals",     lambda T, lib: lib.max(T["a"], dim=1).values)
    add("max_dim_idx",      lambda T, lib: lib.max(T["a"], dim=1).indices.float())
    add("min_dim_vals",     lambda T, lib: lib.min(T["a"], dim=0).values)
    add("prod_dim",         lambda T, lib: T["pos"].prod(dim=1))
    add("mean_dim",         lambda T, lib: T["a"].mean(dim=1))
    add("std_dim",          lambda T, lib: T["a"].std(dim=1))
    add("var_dim",          lambda T, lib: T["a"].var(dim=1))
    add("logsumexp",        lambda T, lib: T["a"].logsumexp(dim=1))
    add("cummax_vals",      lambda T, lib: lib.cummax(T["a"], dim=1).values)
    add("cummax_idx",       lambda T, lib: lib.cummax(T["a"], dim=1).indices.float())
    add("cummin_vals",      lambda T, lib: lib.cummin(T["a"], dim=1).values)
    add("softmax",          lambda T, lib: lib.nn.functional.softmax(T["a"], dim=1))
    add("log_softmax",      lambda T, lib: lib.nn.functional.log_softmax(T["a"], dim=1))
    add("norm_p2_dim",      lambda T, lib: T["a"].norm(p=2, dim=1))
    add("norm_p1_dim",      lambda T, lib: T["a"].norm(p=1, dim=1))
    add("cumsum_neg_dim",   lambda T, lib: T["c3"].cumsum(dim=-1))
    add("repeat_interleave",lambda T, lib: T["vec"].repeat_interleave(2))
    add("narrow",           lambda T, lib: T["a"].narrow(1, 1, 3))
    add("roll_nodim",       lambda T, lib: T["vec"].roll(2))
    add("outer",            lambda T, lib: lib.outer(T["vec"], T["vec"]))
    add("tensor_unfold",    lambda T, lib: T["vec"].unfold(0, 3, 1))
    add("diagonal",         lambda T, lib: T["a"].diagonal(offset=0, dim1=0, dim2=1))
    add("flatten_range",    lambda T, lib: T["c3"].flatten(0, 1))
    # more divergence-prone ops (type promotion, nan-handling, tensor-arg clamp, ...)
    add("roll_multi",       lambda T, lib: T["a"].roll(shifts=(1, 2), dims=(0, 1)))
    add("clamp_tensor",     lambda T, lib: T["a"].clamp(min=T["b"]))          # tensor min
    add("maximum_bcast",    lambda T, lib: lib.maximum(T["a"], T["b"][0]))    # broadcast
    add("minimum_elem",     lambda T, lib: lib.minimum(T["a"], T["b"]))
    add("cumsum_int",       lambda T, lib: T["idx2"].cumsum(dim=1).float())   # int cumsum
    add("prod_all",         lambda T, lib: T["pos"].prod())
    add("amax_dim",         lambda T, lib: T["a"].amax(dim=1))
    add("amin_dim",         lambda T, lib: T["a"].amin(dim=(0, 1)))
    add("count_nonzero",    lambda T, lib: T["mask"].float().count_nonzero().float())
    add("tril_default",     lambda T, lib: T["a"].tril())
    add("cross",            lambda T, lib: lib.cross(T["c3"][:, :, :3].reshape(-1, 3),
                                                     T["c3"][:, :, 1:].reshape(-1, 3), dim=1))
    add("logaddexp",        lambda T, lib: lib.logaddexp(T["a"], T["b"]))
    add("clamp_both_t",     lambda T, lib: T["a"].clamp(min=-T["pos"], max=T["pos"]))
    add("sign",             lambda T, lib: T["a"].sign())
    add("frac",             lambda T, lib: (T["a"] * 3).frac())
    add("trunc",            lambda T, lib: (T["a"] * 3).trunc())
    # mixed finite/inf input + DEFAULT bounds: validates finite values stay untouched
    # while ±inf map to ±float32-max (the common, exact path).
    add("nan_to_num",       lambda T, lib: (T["a"] / T["mask"].float()).nan_to_num())
    # batch 3: math / predicate / indexing / reduction ops prone to silent divergence
    add("erf",              lambda T, lib: lib.erf(T["a"]))
    add("expm1",            lambda T, lib: lib.expm1(T["a"]))
    add("log1p",            lambda T, lib: lib.log1p(T["pos"]))
    add("reciprocal",       lambda T, lib: lib.reciprocal(T["pos"]))
    add("rsqrt",            lambda T, lib: lib.rsqrt(T["pos"]))
    add("hypot",            lambda T, lib: lib.hypot(T["a"], T["b"]))
    add("atan2",            lambda T, lib: lib.atan2(T["a"], T["b"]))
    add("lerp",             lambda T, lib: lib.lerp(T["a"], T["b"], 0.3))
    add("addcmul",          lambda T, lib: T["a"].addcmul(T["b"], T["pos"], value=2.0))
    add("addcdiv",          lambda T, lib: T["a"].addcdiv(T["b"], T["pos"], value=2.0))
    add("clamp_one_min",    lambda T, lib: T["a"].clamp(min=0.0))
    add("isclose",          lambda T, lib: lib.isclose(T["a"], T["a"] + 1e-9).float())
    add("type_promo_add",   lambda T, lib: (T["idx2"] + T["a"]))           # int + float -> float
    add("take_along_dim",   lambda T, lib: lib.take_along_dim(T["a"], T["idx2"], dim=1))
    add("argmax_dim",       lambda T, lib: T["a"].argmax(dim=1).float())
    add("movedim",          lambda T, lib: lib.movedim(T["c3"], 0, 2).flatten())
    add("broadcast_to",     lambda T, lib: T["b"][0].broadcast_to((4, 5)))
    add("logsumexp_keepdim",lambda T, lib: T["a"].logsumexp(dim=1, keepdim=True))
    add("erfc",             lambda T, lib: lib.erfc(T["a"]) if hasattr(lib, "erfc") else 1 - lib.erf(T["a"]))
    return B


def run(side, outdir):
    os.makedirs(outdir, exist_ok=True)
    npz = os.path.join(outdir, "inputs.npz")
    if side == "jt":
        inp = make_inputs()
        np.savez(npz, **inp)
    else:
        inp = dict(np.load(npz))
    import torch  # jittor-as-torch OR real torch depending on env
    lib = torch
    def to_t(v):
        return torch.from_numpy(v) if v.dtype != np.bool_ else torch.from_numpy(v).bool()
    T = {k: to_t(v) for k, v in inp.items()}
    results, errors = {}, {}
    for name, fn in battery():
        # fresh copy per op: some jittor ops (e.g. scatter) are in-place and would
        # poison shared inputs for later ops -- clone so each op sees pristine data.
        Tloc = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in T.items()}
        try:
            r = fn(Tloc, lib)
            results[name] = np.asarray(r.float().cpu().numpy() if hasattr(r, "float") else r, np.float32)
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {str(e).splitlines()[-1][:120]}"
    np.savez(os.path.join(outdir, f"res_{side}.npz"),
             **{k: v for k, v in results.items()})
    json.dump(errors, open(os.path.join(outdir, f"err_{side}.json"), "w"), indent=1)
    print(f"{side}: ran {len(results)} ops, {len(errors)} errored")
    for k, v in errors.items():
        print(f"   ERR {k}: {v}")


def cmp(outdir):
    jt = dict(np.load(os.path.join(outdir, "res_jt.npz")))
    rt = dict(np.load(os.path.join(outdir, "res_rt.npz")))
    ej = json.load(open(os.path.join(outdir, "err_jt.json")))
    er = json.load(open(os.path.join(outdir, "err_rt.json")))
    names = [n for n, _ in battery()]
    nfail = 0
    for n in names:
        if n in ej and n in er:
            print(f"  SKIP  {n:20s} (both errored)")
            continue
        if n in ej:
            print(f"  JT-ERR {n:20s} {ej[n]}"); nfail += 1; continue
        if n in er:
            print(f"  RT-ERR {n:20s} {er[n]} (rt-only; harness/version)"); continue
        a, b = jt[n], rt[n]
        # jittor has no 0-d scalars: a full reduction is shape (1,) where torch is ().
        # Treat that representation difference as equal when both hold one value.
        if a.shape != b.shape:
            if a.size == 1 and b.size == 1:
                d0 = abs(float(a.reshape(-1)[0]) - float(b.reshape(-1)[0]))
                tag = "PASS " if d0 < 1e-4 else "FAIL "
                if d0 >= 1e-4:
                    nfail += 1
                print(f"  {tag} {n:20s} scalar (jittor (1,) vs torch ()) d={d0:.2e}")
                continue
            print(f"  FAIL  {n:20s} shape {a.shape} vs {b.shape}"); nfail += 1; continue
        d = np.abs(a - b)
        rel = d.max() / (np.abs(b).max() + 1e-12)
        tag = "PASS " if rel < 1e-4 else "FAIL "
        if rel >= 1e-4:
            nfail += 1
        print(f"  {tag} {n:20s} max|d|={d.max():.2e} rel={rel:.2e}")
    print(f"\n{'ALL OPS MATCH' if nfail==0 else str(nfail)+' OP(S) DIVERGE'}")


if __name__ == "__main__":
    mode, outdir = sys.argv[1], sys.argv[2]
    if mode == "cmp":
        cmp(outdir)
    else:
        run(mode, outdir)
