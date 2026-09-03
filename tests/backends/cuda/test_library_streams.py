# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""CUDA library handles bind their stream on every real execution."""

import unittest

import numpy as np

import jittor as jt
from jittor.nn.legacy_complex import _fft2


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


class TestCudaLibraryStreams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not jt.has_cuda or _device_count() < 2:
            raise unittest.SkipTest("two CUDA devices are required")
        cls.libs = {
            name: getattr(jt.compile_extern, name, None)
            for name in ("cublas", "cudnn", "cusparse", "curand", "cufft")
        }
        missing = [name for name, module in cls.libs.items() if module is None]
        if missing:
            raise unittest.SkipTest(
                "CUDA library support unavailable: " + ", ".join(missing))

    def _run_libraries(self, device):
        with jt.flag_scope(use_cuda=1, device_id=device):
            matrix = jt.array(np.arange(16, dtype="float32").reshape(4, 4))
            jt.compile_extern.cublas_ops.cublas_matmul(
                matrix, matrix, False, False).sync()

            image = jt.ones((1, 1, 5, 5), "float32")
            kernel = jt.ones((1, 1, 3, 3), "float32")
            jt.cudnn.ops.cudnn_conv(
                image, kernel, 1, 1, 0, 0, 1, 1, 1,
                "abcd", "oihw", "").sync()

            dense = jt.ones((2, 2), "float32")
            values = jt.ones((2,), "float32")
            output = jt.zeros((2, 2), "float32")
            jt.compile_extern.cusparse_ops.cusparse_spmmcsr(
                output, dense, jt.array(np.array([0, 1], "int32")), values,
                jt.array(np.array([0, 1, 2], "int32")),
                2, 2, False, False).sync()

            jt.rand((32,)).sync()
            complex_input = jt.zeros((1, 4, 4, 2), "float32")
            _fft2(complex_input).sync()

    def test_every_library_binds_on_both_devices(self):
        before = {
            (name, device): getattr(
                module, name + "_stream_bind_count")(device)
            for name, module in self.libs.items() for device in (0, 1)
        }
        for device in (0, 1):
            for _ in range(2):
                self._run_libraries(device)

        for name, module in self.libs.items():
            counter = getattr(module, name + "_stream_bind_count")
            with self.subTest(library=name):
                self.assertGreaterEqual(counter(0) - before[name, 0], 2, name)
                self.assertGreaterEqual(counter(1) - before[name, 1], 2, name)


if __name__ == "__main__":
    unittest.main()
