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

import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jittor as jt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import _ecosystem_runner  # noqa: E402
import _ecosystem_harness  # noqa: E402


class _StubTorch(object):
    """Stands in for the ``torch`` argument on the Jittor paths."""


class _StubFlags(object):
    use_cuda = 0
    use_acl = 0


class _SharedNumpyTensor(object):
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


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

    def test_npu_request_requires_acl_and_is_reported_separately(self):
        flags = _StubFlags()
        with mock.patch.object(jt.compiler, "has_acl", 1):
            with mock.patch.object(jt, "flags", flags):
                _ecosystem_runner._select_device(_StubTorch(), "jittor", "npu")
                self.assertEqual(flags.use_cuda, 1)
                self.assertEqual(flags.use_acl, 1)
                self.assertEqual(
                    _ecosystem_runner._device_in_use(
                        _StubTorch(), "jittor", "npu"
                    ),
                    "npu",
                )

    def test_npu_request_fails_without_acl(self):
        with mock.patch.object(jt.compiler, "has_acl", 0):
            with self.assertRaisesRegex(SystemExit, "ACL is unavailable"):
                _ecosystem_runner._select_device(_StubTorch(), "jittor", "npu")

    def test_jittor_tensors_are_never_moved_by_hand(self):
        """The returned callable is identity: Jittor moves the graph, not tensors."""
        move = _ecosystem_runner._select_device(_StubTorch(), "jittor", "cpu")
        sentinel = object()
        self.assertIs(move(sentinel), sentinel)

    def test_shared_package_site_is_inserted_without_duplicates(self):
        original = list(sys.path)

        def restore_path():
            sys.path[:] = original

        self.addCleanup(restore_path)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"JITTOR_ECOSYSTEM_PACKAGE_SITE": directory},
            ):
                sys.path.extend([directory, directory])
                actual = _ecosystem_runner._activate_package_site()
                self.assertEqual(actual, str(Path(directory).resolve()))
                self.assertEqual(sys.path[0], actual)
                self.assertEqual(sys.path.count(actual), 1)

    def test_harness_selects_independent_reference_package_site(self):
        with mock.patch.object(
            _ecosystem_harness, "PACKAGE_SITE", "/packages/python39"
        ):
            with mock.patch.object(
                _ecosystem_harness,
                "REFERENCE_PACKAGE_SITE",
                "/packages/python310",
            ):
                with mock.patch.object(
                    _ecosystem_harness, "REFERENCE_SHARES_PACKAGE_SITE", False
                ):
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(sys.executable),
                        "/packages/python39",
                    )
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(
                            "/oracle/bin/python"
                        ),
                        "/packages/python310",
                    )

    def test_harness_shares_package_site_for_compatible_abis(self):
        with mock.patch.object(
            _ecosystem_harness, "PACKAGE_SITE", "/packages/shared"
        ):
            with mock.patch.object(
                _ecosystem_harness, "REFERENCE_PACKAGE_SITE", ""
            ):
                with mock.patch.object(
                    _ecosystem_harness, "REFERENCE_SHARES_PACKAGE_SITE", True
                ):
                    self.assertEqual(
                        _ecosystem_harness._runner_package_site(
                            "/oracle/bin/python"
                        ),
                        "/packages/shared",
                    )

    def test_correctness_snapshot_does_not_alias_runtime_storage(self):
        storage = np.array([1.0, 2.0], dtype="float32")
        snapshot = _ecosystem_runner._numpy_snapshot(_SharedNumpyTensor(storage))
        storage[:] = -1.0
        np.testing.assert_array_equal(snapshot, np.array([1.0, 2.0], dtype="float32"))


if __name__ == "__main__":
    unittest.main()
