# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``_write_inplace`` overwrites a tensor's buffer without building an op.

It exists so a caller can feed a new value into a graph that is being
executed repeatedly: the buffer identity survives, so everything already
built on top of it keeps pointing at the same storage. The tests pin down
that the bytes land, that the storage really is the same one, and that the
sizes, dtypes and layouts it cannot serve are refused rather than
half-written.
"""

from _helpers import capability as _test_capability
import unittest

import numpy as np

import jittor as jt


class TestWriteInplace(unittest.TestCase):

    def test_value_lands_and_storage_is_reused(self):
        x = jt.array(np.zeros((2, 3), dtype="float32"))
        x.sync()
        before = x.var_ptr
        want = np.arange(6, dtype="float32").reshape(2, 3)
        x._write_inplace(want)
        np.testing.assert_array_equal(x.numpy(), want)
        # Same Var, same storage: that is the whole point of the entry point.
        self.assertEqual(x.var_ptr, before)

    def test_a_consumer_built_earlier_sees_the_new_value(self):
        x = jt.array(np.ones((4,), dtype="float32"))
        y = x * 2
        y.sync()
        np.testing.assert_array_equal(y.numpy(), np.full(4, 2, dtype="float32"))
        x._write_inplace(np.full((4,), 3, dtype="float32"))
        # y was built before the write; rebuilding it must read the new bytes.
        np.testing.assert_array_equal((x * 2).numpy(), np.full(4, 6, dtype="float32"))

    def test_size_mismatch_is_refused(self):
        x = jt.array(np.zeros((2, 3), dtype="float32"))
        x.sync()
        with self.assertRaises(Exception) as caught:
            x._write_inplace(np.zeros((2, 4), dtype="float32"))
        self.assertIn("size mismatch", str(caught.exception))

    def test_dtype_mismatch_is_refused(self):
        x = jt.array(np.zeros((4,), dtype="float32"))
        x.sync()
        with self.assertRaises(Exception) as caught:
            x._write_inplace(np.zeros((4,), dtype="int32"))
        self.assertIn("dtype mismatch", str(caught.exception))

    def test_a_non_dense_tensor_is_refused(self):
        base = jt.array(np.zeros((4, 4), dtype="float32"))
        base.sync()
        view = base[:, :2]
        view.sync()
        if view._storage_is_contiguous():
            self.skipTest("this slice is dense; nothing to refuse here")
        with self.assertRaises(Exception) as caught:
            view._write_inplace(np.zeros((4, 2), dtype="float32"))
        self.assertIn("dense", str(caught.exception))


@unittest.skipIf(not _test_capability.machine_has_accelerator("cuda"),
                 "no CUDA device")
class TestWriteInplaceCuda(TestWriteInplace):
    """The same contract on the accelerator.

    The accelerator path also has to be *fast*: it is ordered against the
    compute stream rather than blocking the host on it, because a blocking
    small host-to-device copy costs milliseconds on some drivers.
    """

    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = self._use_cuda

    def test_the_write_is_ordered_before_a_later_consumer(self):
        x = jt.array(np.zeros((512,), dtype="float32"))
        x.sync()
        # No synchronize between the write and the consumer: stream order is
        # what must make the new bytes visible to the kernel.
        for step in range(8):
            x._write_inplace(np.full((512,), step, dtype="float32"))
            got = (x + 1).numpy()
            np.testing.assert_array_equal(got, np.full(512, step + 1, dtype="float32"))


if __name__ == "__main__":
    unittest.main()
