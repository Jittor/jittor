"""Real-PyTorch reference: faithful GPT-2 (same arch as model_jt.py), load the
SAME weights.npz + input, run fwd+bwd, dump loss + per-name grad L2."""
import math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B, T, V, L, H, E = 2, 64, 512, 2, 4, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"

class CSA(nn.Module):
    def __init__(s, E, H):
        super().__init__(); s.n_head = H; s.c_attn = nn.Linear(E, 3 * E); s.c_proj = nn.Linear(E, E)
    def forward(s, x):
        b, t, c = x.shape
        q, k, v = s.c_attn(x).chunk(3, dim=2)
        q = q.reshape(b, t, s.n_head, c // s.n_head).permute(0, 2, 1, 3)
        k = k.reshape(b, t, s.n_head, c // s.n_head).permute(0, 2, 1, 3)
        v = v.reshape(b, t, s.n_head, c // s.n_head).permute(0, 2, 1, 3)
        att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(k.shape[-1]))
        mask = torch.tril(torch.ones(t, t, device=x.device)).reshape(1, 1, t, t)
        att = F.softmax(att * mask + (1.0 - mask) * (-1e9), dim=-1)
        y = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b, t, c)
        return s.c_proj(y)
class Block(nn.Module):
    def __init__(s, E, H):
        super().__init__(); s.ln_1 = nn.LayerNorm(E); s.attn = CSA(E, H); s.ln_2 = nn.LayerNorm(E)
        s.mlp_fc = nn.Linear(E, 4 * E); s.mlp_proj = nn.Linear(4 * E, E)
    def forward(s, x):
        x = x + s.attn(s.ln_1(x)); x = x + s.mlp_proj(F.gelu(s.mlp_fc(s.ln_2(x)))); return x
class GPT2(nn.Module):
    def __init__(s, V, Tb, L, H, E):
        super().__init__(); s.wte = nn.Embedding(V, E); s.wpe = nn.Embedding(Tb, E)
        s.blocks = nn.ModuleList([Block(E, H) for _ in range(L)]); s.ln_f = nn.LayerNorm(E)
        s.head = nn.Linear(E, V, bias=False)
    def forward(s, idx, tgt):
        b, t = idx.shape; pos = torch.arange(t, device=idx.device).reshape(1, t)
        x = s.wte(idx) + s.wpe(pos)
        for bl in s.blocks: x = bl(x)
        return F.cross_entropy(s.head(s.ln_f(x)).reshape(-1, V), tgt.reshape(-1))

data = np.load("weights.npz"); idx = data["_idx"]
m = GPT2(V, T, L, H, E).to(dev)
named = dict(m.named_parameters())
assert set(named) == set(k for k in data.files if k != "_idx"), "NAME MISMATCH"
with torch.no_grad():
    for n, p in named.items(): p.copy_(torch.from_numpy(data[n]).to(dev))
x = torch.from_numpy(idx[:, :-1]).to(dev); y = torch.from_numpy(idx[:, 1:]).to(dev)
loss = m(x, y); loss.backward()
def l2(t): v = t.detach().double(); return float((v * v).sum().sqrt().cpu())
gl = {n: l2(p.grad) for n, p in named.items()}
tot = math.sqrt(sum(g * g for g in gl.values()))
json.dump({"backend": "torch_" + dev, "loss": float(loss.item()), "total_grad_l2": tot, "grads": gl},
          open("xcheck_torch.json", "w"), indent=2)
print("[torch_%s] loss=%.6f total_grad_l2=%.6f" % (dev, float(loss.item()), tot))
