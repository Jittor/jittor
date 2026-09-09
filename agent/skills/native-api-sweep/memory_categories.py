"""Device-memory behaviour across the native operator categories.

Two things are checked per category: repeated execution must not grow device
memory (a leak), and an explicit release must actually return it. Measurement
forces materialisation first -- Jittor factories are lazy, so a reading taken
before sync attributes the source allocation to whatever ran next.
"""
import argparse, json, subprocess
import numpy as np
import jittor as jt

def smi(dev):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(dev)], capture_output=True, text=True).stdout.strip()
    return int(out)

R = np.random.RandomState(7)
def mk(*shape): return jt.array(R.rand(*shape).astype("float32"))

def elementwise():
    a, b = mk(512, 512), mk(512, 512)
    return (jt.exp(a) + jt.tanh(b) * jt.sigmoid(a)).sum()
def reduction():
    a = mk(512, 512)
    return a.sum() + a.mean() + a.max() + jt.cumsum(a, 1).sum()
def matmul():
    return jt.matmul(mk(256, 256), mk(256, 256)).sum()
def conv():
    return jt.nn.conv2d(mk(4, 8, 32, 32), mk(8, 8, 3, 3)).sum()
def reduce_grad():
    x = mk(256, 256); y = (jt.exp(x) * 2).sum()
    return jt.grad(y, x).sum()
def structural():
    a = mk(128, 128)
    return jt.concat([a, a], 0).transpose().reshape(-1).sum()
def fft_case():
    return jt.fft.rfft(mk(64, 128)).real.sum()
def linalg_case():
    m = R.rand(64, 64).astype("float32"); spd = m @ m.T + 64 * np.eye(64)
    return jt.linalg.inv(jt.array(spd.astype("float32"))).sum()
def transfer():
    a = mk(512, 512)
    return jt.array(a.cpu().numpy()).sum()

CATEGORIES = [
    ("elementwise", elementwise), ("reduction", reduction), ("matmul", matmul),
    ("conv2d", conv), ("autograd", reduce_grad), ("structural", structural),
    ("fft", fft_case), ("linalg", linalg_case), ("transfer", transfer),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smi-index", type=int, default=0,
                    help="physical index queried by nvidia-smi; jittor itself\n                          runs on whatever CUDA_VISIBLE_DEVICES exposes as 0")
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--out", required=True)
    opt = ap.parse_args()
    jt.flags.use_cuda = 1
    jt.ones(1).sync()                       # bring the CUDA context up first
    rows = []
    for name, fn in CATEGORIES:
        try:
            fn().sync()                     # warm: compile + first alloc
            jt.clean()
            base = smi(opt.smi_index)
            for _ in range(opt.repeats):
                fn().sync()
            after = smi(opt.smi_index)
            jt.clean()
            released = smi(opt.smi_index)
            rows.append({"category": name, "growth_MiB": after - base,
                         "after_release_MiB": released - base,
                         "status": "ok" if (after - base) <= 8 and (released - base) <= 8
                                   else "LEAK"})
        except Exception as exc:            # noqa: BLE001 -- classification
            rows.append({"category": name, "status": "raised",
                         "detail": type(exc).__name__ + ": " + str(exc).splitlines()[0][:180]})
    with open(opt.out, "w") as fh: json.dump(rows, fh, indent=1)
    for r in rows:
        print("%-12s %s" % (r["category"], {k: v for k, v in r.items() if k != "category"}))
main()
