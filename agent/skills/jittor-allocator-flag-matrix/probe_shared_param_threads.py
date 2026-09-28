"""Four threads writing disjoint slices of one shared param, 30 rounds.

    NT=4 python probe_shared_param_threads.py

Exits 1 and prints the first thread's error. KI-EXEC-007: this aborts in
`SFRLAllocator::free` with "allocation not found" in 4 runs out of 5 at four
threads, and never at one or two. KI-EXEC-005 is the phase-7 assert that the
same shape used to hit, and that this one now reaches first.

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
