"""Four threads writing disjoint slices of one shared param, 30 rounds.

    NT=4 python probe_shared_param_threads.py

At HEAD (2026-10-26) this shape passes 35/35 at four threads and 6/6 at one:
`VarHolder::sync()` no longer migrates, so the storage the batch reads is not
released underneath it. The same ownership gap is still live on the host-read
path (`.numpy()` while another thread rebinds the holder), where the unversioned
`probe_shared_param_readback.py` still segfaults 3/3; see KI-EXEC-007. The
phase-7 assert this probe used to reach first is KI-EXEC-005.

This is the KI-EXEC-005 shape: thread B's slice assignment rebinds the holder,
so the var thread A's batch requested becomes garbage while that batch runs and
its storage is released. Phase 7 then has to tell that apart from a var that
never got any.
"""
import sys, threading, traceback
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1
NT = int(__import__("os").environ.get("NT", 4))
N = 4096
ROUNDS = 30
param = jt.zeros((NT * N, 256), "float32")
errors = []

def worker(tid):
    global param
    try:
        for _ in range(ROUNDS):
            host = jt.array(np.random.rand(N, 256).astype("float32"))
            param[tid * N:(tid + 1) * N] = host
            param.sync()
    except BaseException as e:            # noqa: BLE001
        errors.append("thread %d: %s" % (tid, e)); traceback.print_exc()

ts = [threading.Thread(target=worker, args=(i,)) for i in range(NT)]
for t in ts: t.start()
for t in ts: t.join()
if errors:
    print("FAIL", errors[0].splitlines()[0][:200])
    sys.exit(1)
print("OK")
