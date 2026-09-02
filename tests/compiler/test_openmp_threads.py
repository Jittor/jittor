"""Jittor must not start one OpenMP thread per hardware thread.

OpenMP's own default is one thread per *logical* CPU. On an SMT machine that
puts two threads on every core, and the cost is not a gentle slowdown: on a
dual 32-core host one batched oneDNN call took 437us with 64 threads and 5955us
with 128, because the barrier across twice as many threads dominates the work.
A 2048x768 by 768x768 matmul measured 1841 GFLOPS against 406 for the same
reason. PyTorch defaults to the physical count.

The default is only a default: an explicit ``OMP_NUM_THREADS`` always wins.
"""

import os
import tempfile
import unittest

from _helpers.child_process import run_python_child

from jittor_utils import (
    _physical_core_count_from_sysfs,
    limit_openmp_to_physical_cores,
    physical_core_count,
)


class TestPhysicalCoreCount(unittest.TestCase):
    @unittest.skipUnless(os.path.exists("/proc/cpuinfo"), "needs /proc/cpuinfo")
    def test_counts_cores_not_hardware_threads(self):
        physical = physical_core_count()
        self.assertIsNotNone(physical)
        self.assertGreaterEqual(physical, 1)
        self.assertLessEqual(physical, os.cpu_count())

    def test_sysfs_counts_smt_sibling_groups(self):
        with tempfile.TemporaryDirectory() as root:
            for cpu_id, siblings in enumerate(("0-1", "0-1", "2-3", "2-3")):
                topology = os.path.join(root, "cpu%d" % cpu_id, "topology")
                os.makedirs(topology)
                with open(os.path.join(topology, "thread_siblings_list"), "w") as handle:
                    handle.write(siblings)
            self.assertEqual(
                _physical_core_count_from_sysfs(range(4), root), 2)
            # An affinity mask containing one thread from each core still has
            # two independently runnable physical cores.
            self.assertEqual(
                _physical_core_count_from_sysfs((1, 2), root), 2)


class TestOpenmpDefault(unittest.TestCase):
    def test_an_explicit_setting_is_never_overridden(self):
        environ = {"OMP_NUM_THREADS": "3"}
        self.assertIsNone(limit_openmp_to_physical_cores(environ))
        self.assertEqual(environ["OMP_NUM_THREADS"], "3")

    def test_a_blank_setting_is_treated_as_unset(self):
        environ = {"OMP_NUM_THREADS": "  "}
        chosen = limit_openmp_to_physical_cores(environ)
        if chosen is None:
            self.skipTest("this host has no SMT to limit")
        self.assertEqual(environ["OMP_NUM_THREADS"], str(chosen))

    def test_the_default_never_exceeds_the_physical_core_count(self):
        environ = {}
        chosen = limit_openmp_to_physical_cores(environ)
        if chosen is None:
            self.skipTest("this host has no SMT to limit")
        self.assertEqual(chosen, physical_core_count())
        self.assertLess(chosen, os.cpu_count())

    def test_importing_jittor_applies_it(self):
        """The value has to be in place before anything links OpenMP."""
        environment = dict(os.environ)
        environment.pop("OMP_NUM_THREADS", None)
        completed = run_python_child(
            ["-c",
             "import os, jittor; print('THREADS', os.environ.get('OMP_NUM_THREADS'))"],
            env=environment,
            merge_stderr=True,
        )
        marker = [line for line in completed.stdout.splitlines()
                  if line.startswith("THREADS ")]
        self.assertTrue(marker, completed.stdout[-2000:])
        value = marker[0].split(None, 1)[1]
        expected = physical_core_count()
        if expected is None or expected >= (os.cpu_count() or expected):
            self.skipTest("this host has no SMT to limit")
        self.assertEqual(value, str(expected))


if __name__ == "__main__":
    unittest.main()
