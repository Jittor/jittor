"""Real CUDA dtype eligibility for native Var and initialized owned carriers."""
from contextlib import contextmanager
import os
import struct
import sys
import unittest

from _helpers import capability
import jittor as jt
from jittor._runtime import dispatch


@contextmanager
def _frontend(tensor_type):
    token = jt.core._set_tensor_frontend_type(tensor_type)
    try:
        yield
    finally:
        jt.core._reset_tensor_frontend_type(token)


@unittest.skipUnless(capability.check_accelerator('cuda', backend=jt).enabled,
                     'real CUDA required; skip is not acceptance')
@unittest.skipUnless(sys.version_info[:3] == (3, 11, 13),
                     'owned carrier currently requires reviewed CPython3.11.13')
class TestOwnedCarrierKernelCUDA(unittest.TestCase):
    def _state(self, value):
        value.sync(True, False)
        self.assertEqual((value.location(), value.device_id, value.placement_backend),
                         ('device', 0, 1))
        return (value.var_ptr, value._storage_address, tuple(value.shape),
                str(value.dtype), value.is_stop_grad(), value.is_backward_leaf,
                value.grad_fn_node_id)

    def _check(self, value, expected):
        before = self._state(value)
        with _frontend(jt.core.Var), jt.no_grad():
            host = jt.device_copy(value, -1)
            self.assertIs(type(host), jt.core.Var)
            self.assertIsNot(host, value)
            host.sync(True, False)
            self.assertEqual(host.location(), 'cpu')
            self.assertEqual(jt.core.dispatch_context([host]), ('cpu', -1))
            self.assertNotEqual(host._storage_address, before[1])
            array = host.numpy()
            self.assertEqual(str(array.dtype), 'float32')
            body = array.tobytes(order='C')
        self.assertEqual(self._state(value), before)
        self.assertEqual(len(body), 4 * len(expected))
        self.assertTrue(struct.unpack('=' + 'f' * len(expected), body) == expected,
                        'complete CUDA kernel-selection result differs; values omitted')

    def _contract(self, tensor_type, family):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error',
                              cuda_kernel_math='strict'), forbid_backend_fallbacks():
            token = jt.core._set_construction_placement(1, 0)
            try:
                with _frontend(tensor_type):
                    value = jt.core.ops.array_([0.25, -0.5, 1.5, 2.0]).start_grad()
                    dy = jt.core.ops.array_([1.0, 2.0, -1.0, 0.5]).stop_grad()
                calls = {'fp32': 0, 'fp64': 0}

                def fp32(argument):
                    calls['fp32'] += 1
                    return argument * 2.0

                def fp64(argument):
                    calls['fp64'] += 1
                    return argument * 2.0

                op = 'test.owned_kernel_dtype.' + str(os.getpid()) + '.' + family
                self.assertIsNone(dispatch.registered_kernel(op, 'cuda'))
                self.addCleanup(self._unregistered, op)
                dispatch.register_kernel(op, 'cuda', fp64, dtypes=('float64',), priority=10)
                self.addCleanup(dispatch.unregister_kernel, op, 'cuda', fp64)
                dispatch.register_kernel(op, 'cuda', fp32, dtypes=('float32',), priority=0)
                self.addCleanup(dispatch.unregister_kernel, op, 'cuda', fp32)
                with _frontend(jt.core.Var):
                    anchor = jt.core.ops.array_([1.0])
                self.assertIs(dispatch.select_kernel(op, anchor), fp32)
                self.assertIs(dispatch._NATIVE_SELECT, jt.core._kernel_select)
                selected = jt.core._kernel_select(op, (value,), {})
                self.assertIs(selected, fp32, 'FP32 input must exclude FP64 candidate')
                output = selected(value)
                self.assertIs(type(output), tensor_type)
                self._check(output, (0.5, -1.0, 3.0, 4.0))
                loss = jt.core.ops.sum(output * dy, 0)
                self.assertIs(type(loss), tensor_type)
                self._check(loss, (-2.5,))
                gradient, = jt.core.grad(loss, [value], False)
                self.assertIs(type(gradient), tensor_type)
                self._check(gradient, (2.0, 4.0, -2.0, 1.0))
                self.assertEqual(calls, {'fp32': 1, 'fp64': 0})
                self._check(value, (0.25, -0.5, 1.5, 2.0))
                self._check(dy, (1.0, 2.0, -1.0, 0.5))
                self.assertEqual(jt.core.backend_fallback_count(), 0)
            finally:
                jt.core._restore_construction_placement(token)

    def _unregistered(self, op):
        self.assertIsNone(dispatch.registered_kernel(op, 'cuda'))
        self.assertIsNone(dispatch.registered_kernel(op, '*'))

    def test_native_Var_FP32_eligibility(self):
        self._contract(jt.core.Var, 'native')

    def test_owned_TensorBase_FP32_eligibility(self):
        root = jt.core._create_owned_tensor_base_type()
        self.assertEqual(root.__bases__, (object,))
        self.assertFalse(issubclass(root, jt.core.Var))
        self._contract(root, 'owned')
