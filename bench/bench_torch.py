"""PyTorch + torch_npu reference timings. See models.md for the shared spec."""
import argparse, json, time
import numpy as np
import torch
import torch_npu  # noqa: F401  (registers the npu device)
import torch.nn as nn

DEV = "npu"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1000))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 64, 64, 128, 128, 256, 256]
        layers = []
        for i in range(6):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, padding=1), nn.ReLU()]
            if i % 2 == 1:
                layers += [nn.MaxPool2d(2, 2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Linear(512, 1000))

    def forward(self, x):
        x = self.features(x)
        return self.head(x.reshape(x.shape[0], -1))


class Block(nn.Module):
    def __init__(self, d=512, h=8, ff=2048):
        super().__init__()
        self.h, self.d = h, d
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.act = nn.GELU()

    def forward(self, x):
        B, L, D = x.shape
        H, hd = self.h, self.d // self.h
        y = self.n1(x)
        qkv = self.qkv(y).reshape(B, L, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5, dim=-1)
        x = x + self.proj((att @ v).permute(0, 2, 1, 3).reshape(B, L, D))
        return x + self.f2(self.act(self.f1(self.n2(x))))


class Transformer(nn.Module):
    def __init__(self, vocab=8192, d=512, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build(name):
    rng = np.random.RandomState(0)
    if name == "mlp":
        model = MLP()
        x = torch.from_numpy(rng.rand(256, 1024).astype(np.float32)).to(DEV)
        y = torch.from_numpy(rng.randint(0, 1000, 256).astype(np.int64)).to(DEV)
        return model.to(DEV), x, y, (lambda o: o)
    if name == "cnn":
        model = CNN()
        x = torch.from_numpy(rng.rand(64, 3, 32, 32).astype(np.float32)).to(DEV)
        y = torch.from_numpy(rng.randint(0, 1000, 64).astype(np.int64)).to(DEV)
        return model.to(DEV), x, y, (lambda o: o)
    if name == "transformer":
        model = Transformer()
        x = torch.from_numpy(rng.randint(0, 8192, (8, 256)).astype(np.int64)).to(DEV)
        y = torch.from_numpy(rng.randint(0, 8192, 8 * 256).astype(np.int64)).to(DEV)
        return model.to(DEV), x, y, (lambda o: o.reshape(-1, o.shape[-1]))
    raise SystemExit("unknown model " + name)


def measure(name, mode, warmup, iters):
    model, x, y, flat = build(name)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(flat(model(x)), y)
        loss.backward()
        optimizer.step()

    kept = []

    def infer_step():
        with torch.no_grad():
            # Keep a scalar off the output. Jittor is lazy, so its counterpart
            # must have a live consumer or the whole forward is dead code; the
            # reduction is added on both sides to keep the work identical.
            kept.append(model(x).sum())

    step = train_step if mode == "train" else infer_step
    if mode == "infer":
        model.eval()
    for _ in range(warmup):
        step()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mlp,cnn,transformer")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--hf32", action="store_true",
                        help="allow HF32 cube math for matmul and convolution")
    args = parser.parse_args()
    torch.npu.matmul.allow_hf32 = bool(args.hf32)
    torch.npu.conv.allow_hf32 = bool(args.hf32)
    out = {"framework": "torch", "hf32": bool(args.hf32), "torch": torch.__version__,
           "torch_npu": torch_npu.__version__,
           "device": torch.npu.get_device_name(0), "results": {}}
    for name in args.models.split(","):
        for mode in ("train", "infer"):
            ms = measure(name, mode, args.warmup, args.iters)
            out["results"]["%s/%s" % (name, mode)] = round(ms, 3)
            print("%-22s %8.3f ms" % (name + "/" + mode, ms), flush=True)
    print("JSON " + json.dumps(out))
