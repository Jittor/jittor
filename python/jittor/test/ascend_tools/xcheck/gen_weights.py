"""Generate deterministic name-keyed weights + input once (pure numpy), so the
torch and jittor runs consume IDENTICAL data. Run with the jittor env."""
import numpy as np
from model_jt import GPT2
B, T, V, L, H, E = 2, 64, 512, 2, 4, 128
SEED = 1234
m = GPT2(V, T, L, H, E)
named = dict(m.named_parameters())
rng = np.random.RandomState(SEED)
wd = {}
for name in sorted(named):
    wd[name] = (rng.standard_normal(tuple(named[name].shape)) * 0.02).astype(np.float32)
wd["_idx"] = rng.randint(0, V, size=(B, T + 1)).astype(np.int64)
np.savez("weights.npz", **wd)
print("saved %d params; e.g. %s" % (len(wd) - 1, sorted(named)[:3]))
