#!/usr/bin/env python
"""Isolate autograd *computation* vs *exposure* bugs on jittor-as-torch.

Run in the JITTOR env:  grad_probe.py <arch>   (arch from parity.py:make_config)

For a fresh tiny model it reports, after loss.backward():
  - how many params have `.grad is None`  (exposure)
  - the leaf-registry length `jt._torch_leaf_params`
  - whether `jt.grad(loss, params)` returns correct nonzero grads (computation)
  - top/bottom grad magnitudes (sanity: e.g. BERT pooler grad == 0 is correct)
If grads compute but `.grad` is None -> exposure bug (registry/backward bridge).
If grads are wrong/NaN -> computation bug (op kernel / autodiff).
"""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch, jittor as jt
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity import make_config, fixed_ids

arch = sys.argv[1] if len(sys.argv) > 1 else "bert"
model = AutoModel.from_config(make_config(arch)); model.eval()
named = list(model.named_parameters())
out = model(input_ids=torch.from_numpy(fixed_ids()))
loss = out.last_hidden_state.float().pow(2).sum()

# computation: jt.grad over all params
g = jt.grad(loss, [p for _, p in named], retain_graph=True)
mags = sorted(((n, float(np.max(np.abs(gi.numpy())))) for (n, _), gi in zip(named, g)),
              key=lambda x: -x[1])

# exposure: torch-idiomatic .backward() then read .grad
loss.backward()
none_after = [n for n, p in named if p.grad is None]
reg = getattr(jt, "_torch_leaf_params", {})

print(f"arch={arch}  nparams={len(named)}")
print(f"leaf_registry_len = {len(reg)}")
print(f".grad is None after backward: {len(none_after)}/{len(named)}")
if none_after:
    print(f"  e.g. {none_after[:6]}")
print(f"jt.grad computed: {sum(1 for _, m in mags if m > 0)}/{len(mags)} nonzero")
print("top grad magnitudes:")
for n, m in mags[:4]:
    print(f"   {n:50s} {m:.4e}")
print("bottom grad magnitudes (near-zero may be correct, e.g. unused heads):")
for n, m in mags[-4:]:
    print(f"   {n:50s} {m:.4e}")

if none_after and all(m > 0 for _, m in mags[:1]):
    print(">> DIAGNOSIS: gradients COMPUTE but .grad EXPOSURE is broken "
          "(leaf registry / backward bridge in torch_compat.py).")
elif any(np.isnan(m) for _, m in mags):
    print(">> DIAGNOSIS: NaN gradients -> op kernel / autodiff COMPUTATION bug.")
else:
    print(">> .grad exposure OK.")
