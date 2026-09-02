"""Report which allocator configurations lose a getitem/setitem copy.

Run it the way SKILL.md says (PYTHONPATH is not optional).  Output is one line
per (device, flag combination, slice index); ``bad=0/N`` everywhere is the
expected state.
"""
import gc

import numpy as np

import jittor as jt

COMBOS = [
    ("default", {}),
    ("no_sfrl", {"use_sfrl_allocator": 0}),
    ("nfef", {"use_nfef_allocator": 1}),
    ("no_temp", {"use_temp_allocator": 0}),
    ("stat", {"use_stat_allocator": 1}),
]
N = 10


def _source(np_value):
    # `* 1.0` is required: it makes the var an op output, allocated by
    # exe.allocator instead of by the always-SFRL cpu_allocator.
    a = jt.array(np_value) * 1.0
    a.sync()
    return a


def getitem_bad(idx, cuda, flags):
    bad = 0
    with jt.flag_scope(use_cuda=cuda, **flags):
        for t in range(N):
            npv = (np.arange(32, dtype="float32") + 1 + t * 100).reshape(4, 8)
            a = _source(npv)
            b = a[idx]
            if not (b.numpy() == npv[idx]).all():
                bad += 1
            del a, b
            gc.collect()
    return bad


def setitem_bad(idx, cuda, flags):
    bad = 0
    with jt.flag_scope(use_cuda=cuda, **flags):
        for t in range(N):
            npv = np.zeros((4, 8), "float32")
            a = _source(npv)
            row = jt.array(np.arange(8, dtype="float32") + 1 + t * 100)
            a[idx] = row
            want = npv.copy()
            want[idx] = np.arange(8, dtype="float32") + 1 + t * 100
            if not (a.numpy() == want).all():
                bad += 1
            del a, row
            gc.collect()
    return bad


def main():
    devices = [0, 1] if jt.has_cuda else [0]
    for cuda in devices:
        for name, flags in COMBOS:
            for idx in (0, 1):
                g = getitem_bad(idx, cuda, flags)
                s = setitem_bad(idx, cuda, flags)
                print(f"cuda={cuda} {name:8s} idx={idx} "
                      f"getitem_bad={g}/{N} setitem_bad={s}/{N}")


if __name__ == "__main__":
    main()
