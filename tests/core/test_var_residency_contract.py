# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Where a ``Var``'s data is, and what the public surface says about it.

The device-method tests under ``tests/backends/cuda`` need *two* CUDA devices,
so on a CPU-only or single-GPU machine every one of them skips -- and a gate
entry that only ever skips looks exactly like one that passes. Residency is not
a two-device question: laziness, host reads and the ``cpu()``/``cuda()`` round
trip all have observable contracts on one device, and some on none.

Everything here therefore runs on CPU. The CUDA cases assert the *additional*
device behaviour and skip individually, so removing the accelerator narrows this
file rather than emptying it.

The contracts pinned here are the ones that surprised a reader at the REPL:

* a Var that has not been computed yet has no residency at all (``"none"``), and
  ``device`` answers with where it *will* land;
* reading a device Var's data moves its storage to the host -- ``location()``
  then reports ``"cpu"`` and is telling the truth, not going stale;
* ``device_id`` keeps the source device across ``cpu()`` while ``device`` and
  ``location()`` follow the data.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


requires_cuda = unittest.skipUnless(_has_cuda(), "a CUDA device is required")


class TestLazyResidency(unittest.TestCase):
    """A Var that has not been computed yet has nothing to be resident in."""

    def test_an_unmaterialized_var_has_no_location(self):
        with jt.flag_scope(use_cuda=0):
            self.assertEqual(jt.ones((4, 4)).location(), "none")

    def test_device_predicts_the_placement_the_flag_will_give_it(self):
        with jt.flag_scope(use_cuda=0):
            self.assertEqual(jt.ones((4, 4)).device, "cpu")

    def test_a_synced_host_var_is_cpu_by_both_spellings(self):
        with jt.flag_scope(use_cuda=0):
            a = jt.ones((4, 4))
            a.sync()
            self.assertEqual(a.location(), "cpu")
            self.assertEqual(a.device, "cpu")

    def test_the_prediction_is_kept_when_the_var_materializes(self):
        # `device` on an unmaterialized Var is a promise about where the data
        # will land. A promise the allocation then breaks would be worse than
        # refusing to answer, so it is asserted rather than assumed.
        with jt.flag_scope(use_cuda=0):
            a = jt.ones((4, 4))
            promised = a.device
            a.sync()
            self.assertEqual(a.device, promised)


class TestHostReadMovesStorage(unittest.TestCase):
    """Reading the data of a device Var moves it, and says so."""

    @requires_cuda
    def test_numpy_migrates_a_device_var_to_the_host(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((64, 64)).cuda()
            a.sync()
            self.assertEqual(a.location(), "device")
            a.numpy()
            # Not a stale reading: location() reports var->allocator, so the
            # storage really is on the host now.
            self.assertEqual(a.location(), "cpu")
            self.assertEqual(a.device, "cpu")

    @requires_cuda
    def test_a_device_op_brings_a_read_var_back(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((64, 64)).cuda()
            a.sync()
            a.numpy()
            self.assertEqual(a.location(), "cpu")
            total = (a * 2).sum().item()
            self.assertEqual(total, 2 * 64 * 64)
            self.assertEqual(a.location(), "device")

    @requires_cuda
    def test_repr_moves_the_storage_the_same_way_numpy_does(self):
        # repr() reaches the data through the same path, so a debugging print
        # costs what an explicit read costs. Pinned so the two spellings cannot
        # drift apart silently.
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((64, 64)).cuda()
            a.sync()
            repr(a)
            self.assertEqual(a.location(), "cpu")


class TestReadingDoesNotRelocate(unittest.TestCase):
    """A read is a query. These say what it currently does instead.

    ``numpy()``, ``repr()`` and an element read all move a device Var's storage
    to the host; the next device operation then migrates it back, measured at
    215x the cost of the same operation without the read, with both copies live
    in device memory meanwhile. Filed as KI-MEM-002 and pinned here as strict
    xfail so the fix produces XPASS and forces the ledger entry closed.

    The reduction case is *not* xfail: ``u.sum().item()`` leaves ``u`` alone,
    because the scalar is a new Var rather than a view. It is asserted
    positively so a future fix cannot regress the one spelling that already
    behaves, and so the defect is never restated as "reading anything moves it".
    """

    @requires_cuda
    @unittest.expectedFailure
    def test_numpy_leaves_the_source_on_the_device(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((256, 256)).cuda()
            a.sync()
            a.numpy()
            self.assertEqual(a.location(), "device")

    @requires_cuda
    @unittest.expectedFailure
    def test_repr_leaves_the_source_on_the_device(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((256, 256)).cuda()
            a.sync()
            repr(a)
            self.assertEqual(a.location(), "device")

    @requires_cuda
    @unittest.expectedFailure
    def test_reading_one_element_does_not_move_the_whole_tensor(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((256, 256)).cuda()
            a.sync()
            a[0][0].item()
            self.assertEqual(a.location(), "device")

    @requires_cuda
    def test_a_reduction_read_leaves_the_source_alone(self):
        # The spelling every training loop uses. Not broken, and asserted so it
        # cannot become broken while the three above are being fixed.
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((256, 256)).cuda()
            a.sync()
            total = a.sum().item()
            self.assertEqual(total, 256 * 256)
            self.assertEqual(a.location(), "device")


class TestCopyToHost(unittest.TestCase):
    """``cpu()`` copies; it does not move the Var it was called on."""

    @requires_cuda
    def test_cpu_returns_a_new_var_and_leaves_the_source_on_the_device(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((32, 32)).cuda()
            a.sync()
            host = a.cpu()
            host.sync()
            self.assertEqual(host.location(), "cpu")
            self.assertEqual(host.device, "cpu")
            self.assertEqual(a.location(), "device")
            self.assertIsNot(host, a)

    @requires_cuda
    def test_device_id_keeps_the_source_device_across_cpu(self):
        # device_id is the index alone and deliberately survives the copy, so
        # `x.cpu().cuda()` returns to the device it came from. `device` and
        # `location()` follow the data instead. Both halves are asserted
        # together because reading device_id alone is what misleads people.
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((32, 32)).cuda()
            a.sync()
            host = a.cpu()
            host.sync()
            self.assertEqual(host.device_id, a.device_id)
            self.assertEqual(host.device, "cpu")

    @requires_cuda
    def test_the_round_trip_returns_the_same_values_to_the_device(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((32, 32)).cuda() * 3
            a.sync()
            back = a.cpu().cuda()
            back.sync()
            self.assertEqual(back.location(), "device")
            np.testing.assert_array_equal(
                back.numpy(), np.full((32, 32), 3.0, "float32"))


class TestDeviceAgreesWithLocation(unittest.TestCase):
    """``device`` is a view of the residency, never a second opinion."""

    def _assert_consistent(self, var):
        where = var.location()
        if where == "cpu":
            self.assertEqual(var.device, "cpu")
        elif where in ("device", "disk"):
            self.assertEqual(var.device, "cuda:%d" % var.device_id)

    def test_host_var_is_consistent(self):
        with jt.flag_scope(use_cuda=0):
            a = jt.ones((8, 8))
            a.sync()
            self._assert_consistent(a)

    @requires_cuda
    def test_device_var_is_consistent_through_a_host_read(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((8, 8)).cuda()
            a.sync()
            self._assert_consistent(a)
            a.numpy()
            self._assert_consistent(a)

    def test_device_is_a_property_not_a_method(self):
        # `a.device()` was tried before `a.device`, so the shape of the answer
        # is part of the contract: a torch caller reads it, never calls it.
        with jt.flag_scope(use_cuda=0):
            a = jt.ones((2, 2))
            self.assertIsInstance(a.device, str)
            with self.assertRaises(TypeError):
                a.device()


if __name__ == "__main__":
    unittest.main()
