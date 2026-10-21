"""Jittor ACL timings, matching bench_torch.py model for model. See models.md."""
import argparse, json, time
import numpy as np
import jittor as jt
from jittor import nn

assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
jt.flags.use_acl = 1
jt.flags.use_cuda = 1


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nn.Linear(1024, 4096), nn.Linear(4096, 4096)
        self.l3, self.l4 = nn.Linear(4096, 4096), nn.Linear(4096, 1000)

    def execute(self, x):
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        x = nn.relu(self.l3(x))
        return self.l4(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 64, 64, 128, 128, 256, 256]
        self.convs = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, padding=1) for i in range(6)])
        self.pool = nn.Pool(2, 2, op="maximum")
        self.h1, self.h2 = nn.Linear(256 * 4 * 4, 512), nn.Linear(512, 1000)

    def execute(self, x):
        for i, conv in enumerate(self.convs):
            x = nn.relu(conv(x))
            if i % 2 == 1:
                x = self.pool(x)
        x = x.reshape([x.shape[0], -1])
        return self.h2(nn.relu(self.h1(x)))


class Block(nn.Module):
    def __init__(self, d=512, h=8, ff=2048):
        super().__init__()
        self.h, self.d = h, d
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def execute(self, x):
        B, L, D = x.shape
        H, hd = self.h, self.d // self.h
        y = self.n1(x)
        qkv = self.qkv(y).reshape([B, L, 3, H, hd]).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = nn.softmax(jt.matmul(q, k.transpose(0, 1, 3, 2)) / hd ** 0.5, dim=-1)
        x = x + self.proj(jt.matmul(att, v).transpose(0, 2, 1, 3).reshape([B, L, D]))
        return x + self.f2(nn.gelu(self.f1(self.n2(x))))


class Transformer(nn.Module):
    def __init__(self, vocab=8192, d=512, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def execute(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build(name):
    rng = np.random.RandomState(0)
    if name == "mlp":
        return (MLP(), jt.array(rng.rand(256, 1024).astype(np.float32)),
                jt.array(rng.randint(0, 1000, 256).astype(np.int32)), lambda o: o)
    if name == "cnn":
        return (CNN(), jt.array(rng.rand(64, 3, 32, 32).astype(np.float32)),
                jt.array(rng.randint(0, 1000, 64).astype(np.int32)), lambda o: o)
    if name == "transformer":
        return (Transformer(), jt.array(rng.randint(0, 8192, (8, 256)).astype(np.int32)),
                jt.array(rng.randint(0, 8192, 8 * 256).astype(np.int32)),
                lambda o: o.reshape([-1, o.shape[-1]]))
    raise SystemExit("unknown model " + name)


def measure(name, mode, warmup, iters):
    model, x, y, flat = build(name)
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train_step():
        optimizer.step(nn.cross_entropy_loss(flat(model(x)), y))

    kept = []

    def infer_step():
        with jt.no_grad():
            # A discarded lazy output is never computed. Keeping a scalar alive
            # forces the forward and matches what the torch script measures.
            kept.append(model(x).sum())

    step = train_step if mode == "train" else infer_step
    if mode == "infer":
        model.eval()
    for _ in range(warmup):
        step()
    jt.sync_all(True)
    start = time.perf_counter()
    for _ in range(iters):
        step()
    jt.sync_all(True)
    return (time.perf_counter() - start) / iters * 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mlp,cnn,transformer")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--hf32", action="store_true",
                        help="allow HF32 cube math for matmul and convolution")
    args = parser.parse_args()
    jt.acl_allow_hf32 = bool(args.hf32)
    out = {"framework": "jittor", "hf32": bool(args.hf32), "jittor": jt.__version__, "results": {}}
    for name in args.models.split(","):
        for mode in ("train", "infer"):
            ms = measure(name, mode, args.warmup, args.iters)
            out["results"]["%s/%s" % (name, mode)] = round(ms, 3)
            print("%-22s %8.3f ms" % (name + "/" + mode, ms), flush=True)
    print("JSON " + json.dumps(out))
