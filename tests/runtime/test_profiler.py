# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import os

#: Both cases here used to count report *rows* for one matmul, from a time when
#: a CPU matmul reached its library kernel through the tuner relay and the
#: report carried a row for each side of it: two rows whose averages were the
#: same execution measured twice (hence the old `abs(x-y)/x < 1e-3`), and five
#: rows for five matmuls. Registering the CPU row of `matmul` (dc00e1d8) removed
#: the relay, so one kernel is one row, and identical unmarked work aggregates
#: into it. Row counts were never the subject; these now assert what the report
#: means -- its arithmetic, and one group per distinct mark set.
def _columns(report):
    header = [str(name) for name in report[0]]
    return header, [dict(zip(header, [str(cell) for cell in row]))
                    for row in report[1:]]


class TestProfiler(unittest.TestCase):
    def test_profiler(self):
        warmup, rerun = 10, 100
        a = jt.rand(1000,1000)
        b = jt.rand(1000,1000)
        jt.sync_all()
        with jt.profile_scope(warmup, rerun, profiler_record_peek=1) as rep:
            jt.matmul(a, b).sync()
        header, rows = _columns(rep)
        self.assertIn("Peek", header, "profiler_record_peek=1 must add the column")
        self.assertEqual(len(rows), 1, [row["Name"] for row in rows])
        entry = rows[0]
        # `rerun` is a budget, not the repetition count: `Profiler::record`
        # turns it into a bit width and repeats `1 << n` times, so the reported
        # Count is the schedule it chose, bounded by what was asked for.
        count = int(entry["Count"])
        self.assertTrue(1 <= count <= rerun, count)
        total, avg = float(entry["TotalTime"]), float(entry["AvgTime"])
        self.assertLess(abs(total - avg * int(entry["Count"])) / total, 1e-3)
        # Min/Max come from timing a loop of runs and dividing, while Avg is
        # TotalTime/Count, so on a kernel whose samples agree to the nanosecond
        # the two roundings can cross by one. The ordering is the property; the
        # last nanosecond of it is not.
        self.assertLessEqual(float(entry["MinTime"]), avg * 1.001)
        self.assertLessEqual(avg, float(entry["MaxTime"]) * 1.001)

    def test_marks(self):
        a = jt.rand(1000,1000)
        b = jt.rand(1000,1000)
        jt.sync_all()
        results = []
        with jt.profile_scope() as rep:
            results.append(jt.matmul(a, b))
            with jt.profile_mark("mark1"):
                results.append(jt.matmul(a, b))
                with jt.profile_mark("mark2"):
                    results.append(jt.matmul(a, b))
            with jt.profile_mark("mark3"):
                results.append(jt.matmul(a, b))
            results.append(jt.matmul(a, b))
            jt.sync_all()
        _, rows = _columns(rep)
        # One group per distinct set of enclosing marks; the two unmarked
        # matmuls are the same kernel and land in one row with Count 2.
        marks = {}
        for row in rows:
            name = row["Name"]
            prefix = name[len("[marks:"):name.index(",]") + 1] if name.startswith("[marks:") else ""
            marks[prefix] = int(row["Count"])
        self.assertEqual(marks, {"": 2, "mark1,": 1, "mark1,mark2,": 1, "mark3,": 1})


if __name__ == "__main__":
    unittest.main()