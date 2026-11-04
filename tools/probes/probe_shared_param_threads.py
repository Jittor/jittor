"""Diagnostic CUDA crash probe: threads write disjoint shared-param slices.

Run with NT=<thread-count> in an isolated, externally time-bounded process.
This probe only collects exceptions; it has no final-value oracle. OK means
no exception was observed, not numerical or concurrency acceptance.
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
