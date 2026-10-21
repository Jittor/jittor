# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# DLPack ownership/device/stream semantics (bindings/pyjt/dlpack.cc): this
# repo previously had zero DLPack support at all (torch.utils.dlpack.
# from_dlpack/to_dlpack were literal NotImplementedError stubs in
# compat/torch, and there was no native __dlpack__ anywhere). This covers
# the native, jittor-core entry points (Var.dlpack/__dlpack__/
# __dlpack_device__, jt.to_dlpack, jt.from_dlpack).
import unittest

import numpy as np

import jittor as jt


class TestDLPack(unittest.TestCase):
    def test_export_then_import_capsule(self):
        a_np = np.arange(12, dtype="float32").reshape(3, 4)
        x = jt.array(a_np)
        capsule = jt.to_dlpack(x)
        y = jt.from_dlpack(capsule)
        np.testing.assert_array_equal(y.numpy(), a_np)

    def test_dunder_protocol_object(self):
        # jt.from_dlpack(obj) where obj implements __dlpack__/__dlpack_device__,
        # not just the legacy raw-capsule calling convention.
        a_np = np.arange(6, dtype="float32")
        x = jt.array(a_np)
        y = jt.from_dlpack(x)
        np.testing.assert_array_equal(y.numpy(), a_np)
        self.assertEqual(x.__dlpack_device__(), (1, 0))  # (kDLCPU, 0)

    def test_capsule_consumed_exactly_once(self):
        x = jt.array(np.arange(4, dtype="float32"))
        capsule = jt.to_dlpack(x)
        jt.from_dlpack(capsule)
        with self.assertRaises(Exception):
            jt.from_dlpack(capsule)

    def test_numpy_producer_contiguous(self):
        a_np = np.arange(24, dtype="float32").reshape(4, 6)
        y = jt.from_dlpack(a_np)
        np.testing.assert_array_equal(y.numpy(), a_np)

    def test_numpy_producer_transposed(self):
        # Non-contiguous: Jittor's Var has no stride concept, so this can't
        # alias zero-copy, but it must still import the correct VALUES
        # (materializing a contiguous copy) rather than reject the tensor
        # or silently misread it.
        a_np = np.arange(24, dtype="float32").reshape(4, 6)
        a_t = a_np.T
        y = jt.from_dlpack(a_t)
        self.assertEqual(tuple(y.shape), a_t.shape)
        np.testing.assert_array_equal(y.numpy(), a_t)

    def test_numpy_producer_strided_slice(self):
        a_np = np.arange(24, dtype="float32").reshape(4, 6)
        a_s = a_np[:, ::2]
        y = jt.from_dlpack(a_s)
        self.assertEqual(tuple(y.shape), a_s.shape)
        np.testing.assert_array_equal(y.numpy(), a_s)

    def test_numpy_producer_negative_stride(self):
        a_np = np.arange(12, dtype="float32").reshape(3, 4)
        a_r = a_np[::-1, :]
        y = jt.from_dlpack(a_r)
        np.testing.assert_array_equal(y.numpy(), a_r)

    def test_empty_tensor(self):
        e = np.zeros((0, 3), dtype="float32")
        y = jt.from_dlpack(e)
        self.assertEqual(tuple(y.shape), (0, 3))

    def test_from_dlpack_capsule_rejects_noncontiguous_directly(self):
        # jt.from_dlpack() implements the strided fallback; the low-level
        # core.from_dlpack_capsule() entry point it's built on must still
        # fail loud on a raw non-contiguous capsule rather than silently
        # misreading strides Jittor's Var has no way to represent.
        a_np = np.arange(24, dtype="float32").reshape(4, 6)
        capsule = a_np.T.__dlpack__()
        with self.assertRaises(Exception):
            jt.core.from_dlpack_capsule(capsule)

    def test_export_copy_true_materializes_independent_buffer(self):
        a_np = np.arange(8, dtype="float32")
        x = jt.array(a_np)
        capsule = x.dlpack(copy=True)
        y = jt.from_dlpack(capsule)
        np.testing.assert_array_equal(y.numpy(), a_np)

    def test_export_versioned_capsule_on_request(self):
        a_np = np.arange(8, dtype="float32")
        x = jt.array(a_np)
        capsule = x.dlpack(max_version=(1, 0))
        y = jt.from_dlpack(capsule)
        np.testing.assert_array_equal(y.numpy(), a_np)

    def test_all_supported_dtypes_round_trip(self):
        for dtype in ("float32", "float64", "int8", "int16", "int32", "int64",
                      "uint8", "bool"):
            a_np = np.array([0, 1, 2, 3], dtype=dtype)
            with self.subTest(dtype=dtype):
                y = jt.from_dlpack(a_np)
                np.testing.assert_array_equal(y.numpy(), a_np)


if __name__ == "__main__":
    unittest.main()
