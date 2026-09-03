# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Native Var device-method contracts that do not need accelerator hardware."""

import unittest

import numpy as np

import jittor as jt


class TestVarToSignature(unittest.TestCase):

    def test_device_and_dtype_keywords_are_order_independent(self):
        with jt.flag_scope(use_cuda=0):
            x = jt.array(np.array([1.25, 2.5], dtype="float32"))
            device_first = x.to(device="cpu", dtype=jt.float16)
            dtype_first = x.to(dtype=jt.float16, device="cpu")
            jt.sync([device_first, dtype_first])

        self.assertEqual(str(device_first.dtype), "float16")
        self.assertEqual(str(dtype_first.dtype), "float16")
        self.assertEqual(device_first.location(), "cpu")
        self.assertEqual(dtype_first.location(), "cpu")

    def test_unavailable_npu_is_not_silently_cuda(self):
        if getattr(jt.compiler, "has_acl", False):
            self.skipTest("this check is for a build without the ACL backend")
        with jt.flag_scope(use_cuda=0):
            x = jt.ones((2,))
            with self.assertRaisesRegex(RuntimeError, "NPU backend is unavailable"):
                x.npu(2)
            with self.assertRaisesRegex(RuntimeError, "NPU backend is unavailable"):
                x.to(device="npu:2", dtype=jt.float16)
            with self.assertRaisesRegex(RuntimeError, "NPU backend is unavailable"):
                jt.nn.Linear(2, 2).npu(2)


if __name__ == "__main__":
    unittest.main()
