"""Compare xcheck JSONs against a reference: loss + per-param grad L2 rel diff.
Usage: python compare.py [xcheck_torch.json]"""
import json, sys, glob
ref_name = sys.argv[1] if len(sys.argv) > 1 else "xcheck_torch.json"
ref = json.load(open(ref_name))
print("reference: %s  loss=%.6f  grad_l2=%.6f\n" % (ref["backend"], ref["loss"], ref["total_grad_l2"]))
for f in sorted(glob.glob("xcheck_*.json")):
    if f == ref_name: continue
    d = json.load(open(f))
    dl = abs(d["loss"] - ref["loss"]) / (abs(ref["loss"]) + 1e-12)
    dg = abs(d["total_grad_l2"] - ref["total_grad_l2"]) / (abs(ref["total_grad_l2"]) + 1e-12)
    worst = ("", 0.0)
    for k in ref.get("grads", {}):
        if k in d.get("grads", {}):
            r = abs(d["grads"][k] - ref["grads"][k]) / (abs(ref["grads"][k]) + 1e-12)
            if r > worst[1]: worst = (k, r)
    print("%-14s loss=%.6f (rel %.2e)  grad_l2=%.6f (rel %.2e)  worst=%s (%.2e)"
          % (d["backend"], d["loss"], dl, d["total_grad_l2"], dg, worst[0], worst[1]))
