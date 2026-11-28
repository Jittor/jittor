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


    def test_dense_no_grad_unbind_copies_aligned_and_unaligned_rows(self):
        for count, rows, width, offset, dtype in (
            (3, 4096, 32, 0, "float32"),
            (3, 4096, 10, 0, "float32"),
            (3, 4096, 13, 0, "int32"),
            (3, 4096, 32, 1, "float32"),
            (3, 4096, 32, 0, "bfloat16"),
            (2, 16384, 64, 0, "float16"),
        ):
            with self.subTest(count=count, width=width, offset=offset, dtype=dtype):
                length = rows * count * width
                values = (np.arange(length + offset) % 111).astype(np.float32)
                with jt.flag_scope(use_cuda=1, backend_fallback="error"), jt.no_grad():
                    source = jt.array(values).cast(dtype)[offset:].reshape(rows, count, width)
                    outputs = jt.unbind(source, -2)
                    expected = values[offset:].reshape(rows, count, width)
                    self.assertEqual(len(outputs), count)
                    for index, output in enumerate(outputs):
                        self.assertFalse(output.requires_grad)
                        self.assertEqual(output.dtype, source.dtype)
                        output.sync()
                        self.assertEqual(output.location(), "device")
                        np.testing.assert_array_equal(output.float32().numpy(), expected[:, index])
