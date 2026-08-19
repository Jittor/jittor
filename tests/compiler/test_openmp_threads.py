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
import subprocess
import sys
import unittest

from jittor_utils import limit_openmp_to_physical_cores, physical_core_count


class TestPhysicalCoreCount(unittest.TestCase):
    @unittest.skipUnless(os.path.exists("/proc/cpuinfo"), "needs /proc/cpuinfo")
    def test_counts_cores_not_hardware_threads(self):
        physical = physical_core_count()
        self.assertIsNotNone(physical)
        self.assertGreaterEqual(physical, 1)
        self.assertLessEqual(physical, os.cpu_count())


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
        completed = subprocess.run(
            (sys.executable, "-c",
             "import os, jittor; print('THREADS', os.environ.get('OMP_NUM_THREADS'))"),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
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
