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
* reading a device Var's data -- ``numpy()``, ``repr()``, ``tolist()`` -- copies
  it to the host and leaves the Var where it lies; ``location()`` keeps
  reporting ``"device"`` because the storage never moved (KI-MEM-002);
* ``device_id`` keeps the source device across ``cpu()`` while ``device`` and
  ``location()`` follow the data.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


#: Probed when the case runs, not when the file is imported.
#:
#: ``unittest.skipUnless(_has_cuda(), ...)`` evaluated the probe in a decorator
#: argument, which runs at *collection* -- where this suite forbids backend
#: work -- and froze one answer for the whole process.
requires_cuda = _test_capability.accelerator_required('cuda', backend=jt)


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


class TestReadingDoesNotRelocate(unittest.TestCase):
    """A read is a query: it must not rebind the source's storage.

    ``numpy()``, ``repr()`` and ``tolist()`` all reach the data through the
    native fetch primitive, which used to call ``migrate_to_cpu`` and replace
    the Var's device storage with a host block (KI-MEM-002). The next device
    operation then copied it back -- measured at 20x the cost of the same
    operation on a tensor that was never read, with both copies live in device
    memory meanwhile. The primitive now materialises a host copy and leaves the
    source where it is.

    Placement alone is a weak assertion: a fix that migrated and then migrated
    back would pass it. So each spelling also pins the storage address, which a
    rebind cannot preserve, and the live device bytes, which a migration
    releases. Element and reduction reads are asserted alongside because they
    were never broken, and the defect must not be restated as "reading
    anything moves it".
    """

    def setUp(self):
        _test_capability.require_accelerator("cuda")

    def _resident_tensor(self):
        a = jt.ones((256, 256)).cuda() * 2
        a.sync()
        self.assertEqual(a.location(), "device")
        return a

    def _assert_still_resident(self, a, address, used_before):
        self.assertEqual(a.location(), "device")
        self.assertEqual(a.device, "cuda:%d" % a.device_id)
        self.assertEqual(a._storage_address, address)
        self.assertGreaterEqual(
            jt.core.device_memory_used(a.device_id), used_before)

    def test_numpy_leaves_the_source_on_the_device(self):
        with jt.flag_scope(use_cuda=1):
            a = self._resident_tensor()
            address = a._storage_address
            used = jt.core.device_memory_used(a.device_id)
            host = a.numpy()
            self._assert_still_resident(a, address, used)
            np.testing.assert_array_equal(
                host, np.full((256, 256), 2.0, "float32"))

    def test_repr_leaves_the_source_on_the_device(self):
        # Printing a tensor at a REPL must not relocate it.
        with jt.flag_scope(use_cuda=1):
            a = self._resident_tensor()
            address = a._storage_address
            used = jt.core.device_memory_used(a.device_id)
            text = repr(a)
            self._assert_still_resident(a, address, used)
            self.assertIn("2.", text)

    def test_tolist_leaves_the_source_on_the_device(self):
        with jt.flag_scope(use_cuda=1):
            a = self._resident_tensor()
            address = a._storage_address
            used = jt.core.device_memory_used(a.device_id)
            values = a.tolist()
            self._assert_still_resident(a, address, used)
            self.assertEqual(values[0][0], 2.0)
            self.assertEqual(len(values), 256)
            self.assertEqual(len(values[0]), 256)

    def test_a_device_op_after_a_read_uses_the_resident_source(self):
        # The cost of the read was never the transfer itself: it was that the
        # next device operation had to bring the storage back. The source must
        # still hold the same device allocation after the read and the op.
        with jt.flag_scope(use_cuda=1):
            for read in (lambda a: a.numpy(), repr, lambda a: a.tolist()):
                a = self._resident_tensor()
                address = a._storage_address
                used = jt.core.device_memory_used(a.device_id)
                read(a)
                self.assertEqual((a * 2).sum().item(), 4 * 256 * 256)
                self._assert_still_resident(a, address, used)

    def test_reading_one_element_does_not_move_the_whole_tensor(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((256, 256)).cuda()
            a.sync()
            self.assertEqual(a[0][0].item(), 1.0)
            self.assertEqual(a.location(), "device")
            # A slice view's element too, and the base keeps working on the
            # device afterwards.
            self.assertEqual(a[3, 4:5].item(), 1.0)
            self.assertEqual(a.location(), "device")
            self.assertEqual((a * 2).sum().item(), 2 * 256 * 256)

    def test_a_reduction_read_leaves_the_source_alone(self):
        # The spelling every training loop uses. It was never broken, and is
        # asserted so it cannot become broken while the reads above were fixed.
        with jt.flag_scope(use_cuda=1):
            a = jt.ones((256, 256)).cuda()
            a.sync()
            total = a.sum().item()
            self.assertEqual(total, 256 * 256)
            self.assertEqual(a.location(), "device")


class TestCopyToHost(unittest.TestCase):
    """``cpu()`` copies; it does not move the Var it was called on."""

    def setUp(self):
        _test_capability.require_accelerator("cuda")

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

    def test_device_var_is_consistent_through_a_host_read(self):
        _test_capability.require_accelerator("cuda")
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
