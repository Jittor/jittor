# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Native ``Var``/``Module`` device methods on two real CUDA devices."""

import ctypes
import unittest

import numpy as np

import jittor as jt


_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
_libcuda = None


def _device_count():
    try:
        return int(jt.get_device_count())
    except Exception:
        return 0


def _pointer_device(var):
    global _libcuda
    if _libcuda is None:
        try:
            _libcuda = ctypes.CDLL("libcuda.so.1")
        except OSError:
            _libcuda = False
    if _libcuda is False:
        return None
    device = ctypes.c_int(-1)
    result = _libcuda.cuPointerGetAttribute(
        ctypes.byref(device), _CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        ctypes.c_void_p(var.device_raw_ptr))
    return device.value if result == 0 else None


class _TwoDeviceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not jt.has_cuda or _device_count() < 2:
            raise unittest.SkipTest("two CUDA devices are required")

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=1, device_id=0)
        self._scope.__enter__()

    def tearDown(self):
        jt.sync_all(True)
        self._scope.__exit__(None, None, None)


class TestVarDeviceMethods(_TwoDeviceCase):

    def test_to_applies_device_and_dtype_in_either_keyword_order(self):
        x = jt.array(np.arange(8, dtype="float32"))
        x.sync()
        device_first = x.to(device="cuda:1", dtype=jt.float16)
        dtype_first = x.to(dtype=jt.float16, device="cuda:1")
        device_first.sync()
        dtype_first.sync()

        for value in (device_first, dtype_first):
            self.assertEqual(str(value.dtype), "float16")
            self.assertEqual(value.device_id, 1)
            self.assertEqual(_pointer_device(value), 1)
            np.testing.assert_array_equal(
                value.numpy(), np.arange(8, dtype="float16"))

    def test_cuda_index_moves_to_the_requested_device(self):
        x = jt.arange(8).float32()
        x.sync()
        moved = x.cuda(1)
        moved.sync()
        self.assertEqual(_pointer_device(x), 0)
        self.assertEqual(moved.device_id, 1)
        self.assertEqual(_pointer_device(moved), 1)

    def test_cpu_returns_host_resident_copy(self):
        x = jt.arange(8).float32()
        x.sync()
        host = x.cpu()
        host_via_to = x.to("cpu")
        host.sync()
        host_via_to.sync()
        self.assertIsNot(host, x)
        self.assertEqual(x.location(), "device")
        self.assertEqual(host.location(), "cpu")
        self.assertEqual(host_via_to.location(), "cpu")
        np.testing.assert_array_equal(host.numpy(), np.arange(8, dtype="float32"))

        # Host residency and accelerator ownership are separate. The host copy
        # remembers device 0; cuda(0) must still upload it instead of treating
        # the matching device_id as proof that it is already in device memory.
        round_trip = host.cuda(0)
        round_trip.sync()
        self.assertEqual(host.location(), "cpu")
        self.assertEqual(_pointer_device(round_trip), 0)


class _WithBuffer(jt.Module):
    def __init__(self):
        self.weight = jt.ones((4,))
        self.register_buffer("scale", jt.ones((4,)) * 2)

    def execute(self, x):
        return x * self.weight * self.scale


class TestModuleDeviceMethods(_TwoDeviceCase):

    def test_cuda_index_moves_parameters_and_buffers_in_place(self):
        module = _WithBuffer()
        weight = module.weight
        scale = module.scale
        jt.sync([weight, scale], device_sync=True)

        returned = module.cuda(1)
        jt.sync([module.weight, module.scale], device_sync=True)

        self.assertIs(returned, module)
        self.assertIs(module.weight, weight)
        self.assertIs(module.scale, scale)
        self.assertEqual(_pointer_device(module.weight), 1)
        self.assertEqual(_pointer_device(module.scale), 1)
        with jt.flag_scope(device_id=1):
            output = module(jt.ones((4,)))
            output.sync()
        self.assertEqual(_pointer_device(output), 1)
        np.testing.assert_array_equal(output.numpy(), np.full(4, 2.0, "float32"))


if __name__ == "__main__":
    unittest.main()
