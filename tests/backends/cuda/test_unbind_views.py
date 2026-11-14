"""No-grad unbind must retain slices of a strided source instead of copying twice."""
import unittest

import numpy as np

from _helpers import capability as _test_capability
import jittor as jt


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled,
                 'no usable CUDA in this build')
class TestUnbindViews(unittest.TestCase):
    def test_permuted_slices_preserve_values_and_storage_strides(self):
        for use_cuda in (0, 1):
            for count in (2, 3):
                for dtype in ('float32', 'int32'):
                    with self.subTest(cuda=use_cuda, count=count, dtype=dtype):
                        # Both sizes enter the old multi-output copy route.
                        shape = (4, 257, count, 16, 80)
                        array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                        with jt.flag_scope(use_cuda=use_cuda, transpose_storage_view=1):
                            with jt.no_grad():
                                source = jt.array(array).transpose(2, 0, 3, 1, 4)
                                outputs = jt.unbind(source, 0)
                                self.assertEqual(len(outputs), count)
                                expected_strides = tuple(source._storage_strides())[1:]
                                for index, output in enumerate(outputs):
                                    self.assertEqual(tuple(output._storage_strides()), expected_strides)
                                    np.testing.assert_array_equal(output.numpy(), array[:, :, index].transpose(0, 2, 1, 3))
                                    self.assertEqual(output.location(), 'device' if use_cuda else 'cpu')
