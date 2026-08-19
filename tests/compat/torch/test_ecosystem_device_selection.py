"""The parity harness must put Jittor on the device it claims to measure.

Jittor has no per-tensor device: one global flag moves the whole graph, and on
a machine with a GPU that flag starts out *on*. A CPU comparison that merely
refrains from requesting CUDA therefore measures Jittor on the accelerator
against PyTorch on the CPU. That does not fail -- the numbers still agree,
because both runtimes are correct -- it silently reports the CPU half of the
2.0 goals as covered when it never ran, and it turned a 1.8x slowdown into a
reported 20x speedup.

These tests exercise the selection helper directly so the contract is checked
without an oracle interpreter or a downstream library.
"""

import sys
from pathlib import Path
import unittest

import jittor as jt


sys.path.insert(0, str(Path(__file__).resolve().parent))

import _ecosystem_runner  # noqa: E402


class _StubTorch(object):
    """Stands in for the ``torch`` argument on the Jittor paths."""


class TestEcosystemDeviceSelection(unittest.TestCase):
    def setUp(self):
        self._restore = jt.flags.use_cuda
        self.addCleanup(self._put_back)

    def _put_back(self):
        jt.flags.use_cuda = self._restore

    def test_cpu_request_turns_cuda_off(self):
        jt.flags.use_cuda = 1 if jt.has_cuda else 0
        _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu")
        self.assertEqual(jt.flags.use_cuda, 0)

    def test_cpu_request_is_reported_as_cpu(self):
        jt.flags.use_cuda = 1 if jt.has_cuda else 0
        _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu")
        self.assertEqual(
            _ecosystem_runner._device_in_use(_StubTorch(), "jittor", "cpu"), "cpu"
        )

    @unittest.skipUnless(jt.has_cuda, "CUDA is unavailable")
    def test_cuda_request_turns_cuda_on_and_is_reported(self):
        jt.flags.use_cuda = 0
        _ecosystem_runner._select_device(_StubTorch(), "jittor", "cuda")
        self.assertEqual(jt.flags.use_cuda, 1)
        self.assertEqual(
            _ecosystem_runner._device_in_use(_StubTorch(), "jittor", "cuda"), "cuda"
        )

    def test_jittor_tensors_are_never_moved_by_hand(self):
        """The returned callable is identity: Jittor moves the graph, not tensors."""
        move = _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu")
        sentinel = object()
        self.assertIs(move(sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()
