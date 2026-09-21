"""The real H3 decode, run from two Python threads.

Five synthetic reproductions failed to trigger the video corruption, each with
tens of thousands of samples. What they all lacked is the decode itself: a
handful of elementwise ops on a 64x64 tensor is not a VAE with real weights,
real shapes, fp16 autocast and a fused graph of hundreds of thousands of nodes.
This keeps the decode exactly as `probe_decode_base_repeat.py` runs it -- same
checkpoint, same captured latent, same autocast -- and adds the one thing the
server does that that probe does not: a second Python thread fetching the result
while the first keeps decoding.

The consumer compares against a reference captured on the first, quiet decode.
Any later decode of the same latent must match it: same input, same weights, no
sampling. A mismatch is the corruption.

    python3 h3_decode_race.py [rounds] [consumer_threads]
"""
import json
import sys
import threading
import time
import queue

import numpy as np
import torch

COMP = "/root/jittor-lab/_state/h3/models/MiniMax-H3/FL2VA/video_vae"
NPZ = "/root/jittor-lab/comfyui/server_latent512.npz"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NCONS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

cfg = json.load(open("%s/config.json" % COMP))
from transformers.dynamic_module_utils import get_class_from_dynamic_module

cls = get_class_from_dynamic_module(cfg["auto_map"]["AutoModel"], COMP)
model = cls.from_pretrained(COMP).eval().to(torch.device("cuda:0"))
# `torch.from_numpy(...).to("cuda:0")` goes through a shim path that calls
# `_owner._is_index` (method_api.py:602), and the tensor installer in this
# deployment does not define it -- the stock decode probe fails the same way,
# so this is the tree's own shim inconsistency, not this script's. `as_tensor`
# with an explicit device avoids that path.
_lat_np = np.load(NPZ)["latent"]
latent = torch.as_tensor(_lat_np, device="cuda:0")
mean = torch.tensor(cfg["latents_mean"], device="cuda:0",
                    dtype=torch.float32).view(1, -1, 1, 1, 1)
std = torch.tensor(cfg["latents_std"], device="cuda:0",
                   dtype=torch.float32).view(1, -1, 1, 1, 1)


import os
SYNC_PRODUCER = int(os.environ.get("H3DR_SYNC_PRODUCER", "0"))
_SYNC_REPORTED = False


def decode():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                enabled=True):
        out = model.decode_base(latent * std + mean)
    if SYNC_PRODUCER == 1:
        # `varsync` syncs *this Var*, not the world. The two are not the same
        # here: on the server, `jt.sync_all(True)` at the same position behaved
        # differently from `video.sync(True)`, which is what the w_all arm was
        # for. Reaching the underlying jittor Var through the shim's tensor.
        # Say once what was actually reached and whether the sync ran. Every
        # change in this investigation that was scored without that check
        # turned out not to have taken effect.
        # `torch.Tensor` in this shim *inherits from* jittor_core.Var, so the
        # sync is on the tensor itself -- exactly what the deployment's
        # `varsync` does. Looking for `_jt_var` found None and silently fell
        # back to `out.float().sum().item()`, which is a different operation;
        # two arms were scored against that before the self-report caught it.
        global _SYNC_REPORTED
        did = "tensor.sync"
        out.sync(True)
        if not _SYNC_REPORTED:
            _SYNC_REPORTED = True
            print("sync path=%s out_type=%s has_jt_var=%s"
                  % (did, type(out).__name__, True), flush=True)
    elif SYNC_PRODUCER == 2:
        import jittor as jt
        jt.sync_all(True)
    return out


print("START rounds=%d consumers=%d" % (ROUNDS, NCONS), flush=True)
t0 = time.time()
ref = decode().float().cpu().numpy()     # quiet reference, nothing else running
print("reference captured at %.1fs shape=%s mean=%.4f"
      % (time.time() - t0, ref.shape, float(ref.mean())), flush=True)

# The server does one decode per request, seconds apart. This producer decodes
# flat out, which is two or three orders of magnitude more pressure and may open
# paths the server never reaches -- the workaround that fixes the server does
# not fix this reproduction, and that is the first thing to test. PACED=1 makes
# the producer wait for the consumer to finish each item before starting the
# next, which is the server's rhythm.
PACED = int(os.environ.get("H3DR_PACED", "0"))
done_one = threading.Event()
work = queue.Queue(maxsize=1 if PACED else 3)
stop = threading.Event()
results = []
lock = threading.Lock()


def consumer():
    while not stop.is_set():
        try:
            i, out = work.get(timeout=1)
        except queue.Empty:
            continue
        got = out.float().cpu().numpy()   # the cross-thread fetch
        # NaN in the output makes `np.abs(got-ref).max()` NaN, and `nan > tol`
        # is False -- so a NaN decode scored as a pass. That is a checker that
        # turns a failure into a success, which is worse than one that misses
        # a failure, and it is the eighth scoring hole found in this
        # investigation. NaN is now infinite error.
        nan_frac = float(np.isnan(got).mean())
        if nan_frac > 0:
            rel = float("inf")
        else:
            d = float(np.abs(got - ref).max())
            rel = d / max(1e-9, float(np.abs(ref).max()))
        with lock:
            results.append((i, rel))
        del out, got
        if PACED:
            done_one.set()


# NCONS=0 is the control: the main thread fetches each decode itself, so no
# second Python thread is ever inside jittor. If that is clean while NCONS>=1
# corrupts, concurrency is the variable -- which is the whole claim.
cons = [threading.Thread(target=consumer, daemon=True, name="fetch%d" % k)
        for k in range(NCONS)]
for c in cons:
    c.start()

for i in range(ROUNDS):
    out = decode()                        # producer keeps decoding, does not wait
    if NCONS == 0:
        got = out.float().cpu().numpy()
        rel = (float("inf") if np.isnan(got).any()
               else float(np.abs(got - ref).max()) / max(1e-9, float(np.abs(ref).max())))
        with lock:
            results.append((i, rel))
        del got
    else:
        try:
            if PACED:
                done_one.clear()
            work.put((i, out), timeout=5)
            if PACED:
                done_one.wait(timeout=30)   # the server's rhythm: one at a time
        except queue.Full:
            pass
    del out

# let the consumers drain
deadline = time.time() + 60
while time.time() < deadline:
    with lock:
        n = len(results)
    if n >= ROUNDS or work.empty():
        time.sleep(2)
        with lock:
            n = len(results)
        if n >= ROUNDS:
            break
    time.sleep(1)
stop.set()
for c in cons:
    c.join(timeout=5)

with lock:
    checked = len(results)
    # Threshold from the single-threaded control, not guessed: twelve decodes
    # with no second thread land in 0.0017-0.0023, which is what fp16 and
    # cuDNN's own nondeterminism cost. Anything above 0.01 is four times the
    # widest of those and three orders below what the two-thread arm produces
    # (0.84-69.6), so the two populations do not overlap anywhere near it.
    bad = [(i, r) for i, r in results if r > 1e-2]
for i, r in bad[:5]:
    print("  MISMATCH round=%d rel=%.4g" % (i, r), flush=True)
finite = [r for _, r in results if np.isfinite(r)]
n_inf = sum(1 for _, r in results if not np.isfinite(r))
worst = max(finite, default=0.0)
print("checked=%d mismatches=%d nan_decodes=%d worst_finite_rel=%.4g"
      % (checked, len(bad), n_inf, worst))
print("VERDICT:", "NO-SAMPLES" if checked == 0
      else ("REPRODUCED" if bad else "no trigger"))
