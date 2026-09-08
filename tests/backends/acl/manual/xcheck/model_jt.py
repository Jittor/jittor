"""Jittor GPT-2 used by the cross-backend / vs-torch numerical parity check.
Architecture is kept bit-faithful to the torch model in run_torch.py (Linear
[out,in], exact-erf gelu, LayerNorm eps=1e-5, explicit -1e9 causal mask)."""
import math
import jittor as jt
from jittor import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def execute(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=2)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        att = jt.matmul(q, k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(k.shape[-1]))
        mask = jt.tril(jt.ones((T, T))).reshape(1, 1, T, T)
        att = att * mask + (1.0 - mask) * (-1e9)
        att = nn.softmax(att, dim=-1)
        y = jt.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp_fc = nn.Linear(n_embd, 4 * n_embd)
        self.mlp_proj = nn.Linear(4 * n_embd, n_embd)

    def execute(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp_proj(nn.gelu(self.mlp_fc(self.ln_2(x))))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def execute(self, idx, targets):
        B, T = idx.shape
        pos = jt.arange(T).reshape(1, T)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        return nn.cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
