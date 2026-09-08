"""Jittor side: load weights.npz (name-keyed) + input, run the GPT-2, dump loss
and per-name grad L2.  Usage: python run_jt.py [acl|cuda]"""
import sys, json, math, numpy as np, jittor as jt
backend = sys.argv[1] if len(sys.argv) > 1 else "acl"
if backend == "acl":
    jt.flags.use_acl = 1; jt.flags.use_cuda = 1
else:
    jt.flags.use_cuda = 1
from model_jt import GPT2
B, T, V, L, H, E = 2, 64, 512, 2, 4, 128
m = GPT2(V, T, L, H, E)
data = np.load("weights.npz"); idx = data["_idx"]
named = dict(m.named_parameters())
wkeys = set(k for k in data.files if k != "_idx")
if set(named) != wkeys:
    print("NAME MISMATCH jt-only=%s w-only=%s" % (sorted(set(named) - wkeys), sorted(wkeys - set(named))))
    sys.exit(2)
for n, p in named.items():
    p.assign(jt.array(data[n].astype(np.float32)))
jt.sync_all(True)
x = jt.array(idx[:, :-1].astype(np.int32)); y = jt.array(idx[:, 1:].astype(np.int32))
loss = m(x, y)
names = sorted(named); grads = jt.grad(loss, [named[n] for n in names])
jt.sync_all(True)
def l2(a): v = a.astype(np.float64); return float(math.sqrt((v * v).sum()))
gl = {n: l2(g.numpy()) for n, g in zip(names, grads)}
tot = math.sqrt(sum(v * v for v in gl.values()))
json.dump({"backend": "jt_" + backend, "loss": float(loss.numpy().reshape(-1)[0]),
           "total_grad_l2": tot, "grads": gl}, open("xcheck_jt_%s.json" % backend, "w"), indent=2)
print("[jt_%s] loss=%.6f total_grad_l2=%.6f" % (backend, float(loss.numpy().reshape(-1)[0]), tot))
