#!/usr/bin/env python3
"""Report how much default-stream compute ran *inside* each NCCL kernel.

Input is an nsys report exported to sqlite:

    nsys export --type sqlite --force-overwrite true -o tl.sqlite tl.nsys-rep
    python nccl_overlap_report.py tl.sqlite

For every NCCL kernel it prints how many default-stream kernels were running
concurrently and how much of the collective's window they covered. That is the
evidence "communication overlapped with compute" needs, and unlike a wall-clock
comparison it cannot be produced by the workload simply getting faster.

Judgement: run the same workload twice, once with the join taken immediately
and once deferred. The immediate run must report **0 concurrent kernels** on
every collective. A non-zero count there means the join is being recorded
somewhere it orders nothing -- with `ncclGroupStart` open, for instance, the
collectives are not on the stream yet, so a done event recorded before
`ncclGroupEnd()` is vacuous and the "overlap" you are looking at is a race.
"""

import sqlite3
import sys


def kernels(path):
    con = sqlite3.connect(path)
    return con.execute("""
        SELECT k.start, k.end, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start""").fetchall()


def main(path):
    rows = kernels(path)
    nccl = [(a, b) for a, b, _sid, n in rows if "nccl" in n.lower()]
    other = [(a, b) for a, b, _sid, n in rows if "nccl" not in n.lower()]
    if not nccl:
        raise SystemExit("no NCCL kernels in this report: nothing to say about "
                         "overlap (was the collective actually executed?)")
    print("nccl kernels: {}   other kernels: {}".format(len(nccl), len(other)))
    for i, (start, end) in enumerate(nccl):
        window = end - start
        inside = [(a, b) for a, b in other if b > start and a < end]
        covered = sum(min(b, end) - max(a, start) for a, b in inside)
        print("nccl#{:<3d} dur={:8.3f}ms  concurrent default-stream kernels={:<4d}"
              "  compute inside window={:8.3f}ms ({:.0f}%)".format(
                  i, window / 1e6, len(inside), covered / 1e6,
                  100.0 * covered / window))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
