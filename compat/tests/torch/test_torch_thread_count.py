"""`get_num_threads` has to answer what was asked for.

It returned `os.cpu_count()` unconditionally, which is wrong twice: it ignored
`OMP_NUM_THREADS`, which real torch honours (measured on this machine:
`OMP_NUM_THREADS=1` gives 1), and it ignored `set_num_threads`, whose value
never came back. Code that sizes a pool from this got the machine's logical
core count whatever it or its operator asked for.

Found because the ecosystem speed harness refuses to compare runtimes measured
under different thread counts, and the two sides disagreed: 384 on the shim
against real torch's 192.
"""
import os
import unittest
from unittest import mock

import torch


class TestThreadCountReporting(unittest.TestCase):
    def setUp(self):
        from jittor.compat.torch.installers import utilities
        self._utilities = utilities
        self.addCleanup(setattr, utilities, "_INTRA_OP_THREADS",
                        utilities._INTRA_OP_THREADS)

    def test_set_then_get_round_trips(self):
        torch.set_num_threads(3)
        self.assertEqual(torch.get_num_threads(), 3)

    def test_omp_num_threads_is_honoured(self):
        self._utilities._INTRA_OP_THREADS = None
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            self.assertEqual(torch.get_num_threads(), 2)

    def test_an_explicit_request_outranks_the_environment(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            torch.set_num_threads(5)
            self.assertEqual(torch.get_num_threads(), 5)

    def test_the_fallback_is_affinity_not_cpu_count(self):
        # Inside a cpuset the process cannot use the cores cpu_count counts.
        self._utilities._INTRA_OP_THREADS = None
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": ""}):
            reported = torch.get_num_threads()
        if hasattr(os, "sched_getaffinity"):
            self.assertEqual(reported, len(os.sched_getaffinity(0)))
        self.assertGreater(reported, 0)

    def test_a_junk_environment_value_does_not_crash(self):
        self._utilities._INTRA_OP_THREADS = None
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "not-a-number"}):
            self.assertGreater(torch.get_num_threads(), 0)
