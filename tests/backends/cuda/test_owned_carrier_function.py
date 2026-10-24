"""Native CUDA custom-VJP contract for Var and the owned TensorBase carrier.

The supplied VJP deliberately differs from natural differentiation of square.
Host data is initialization or an independent D2H check, never model execution.
"""
from contextlib import contextmanager
import struct
import sys
import unittest

from _helpers import capability
import jittor as jt


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
class TestOwnedCarrierFunctionCUDA(unittest.TestCase):
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
            self.assertEqual(jt.core.dispatch_context([host]), ('cpu', -1))
            self.assertEqual(host.location(), 'cpu')
            self.assertNotEqual(host._storage_address, before[1])
            array = host.numpy()
            self.assertEqual(str(array.dtype), 'float32')
            body = array.tobytes(order='C')
        self.assertEqual(self._state(value), before)
        self.assertEqual(len(body), 4 * len(expected))
        self.assertTrue(struct.unpack('=' + 'f' * len(expected), body) == expected,
                        'full FP32 custom-VJP contract differs; values omitted')

    def _contract(self, tensor_type):
        native = jt.core.Var
        native_before = (native.__bases__, native.__mro__, dict(native.__dict__))
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error',
                              cuda_kernel_math='strict'), forbid_backend_fallbacks():
            token = jt.core._set_construction_placement(1, 0)
            try:
                with _frontend(tensor_type):
                    value = jt.core.ops.array_([0.25, -0.5, 1.5, 2.0]).start_grad()
                    dy = jt.core.ops.array_([1.0, 2.0, -1.0, 0.5]).stop_grad()
                self.assertIs(type(value), tensor_type)
                self.assertTrue(jt.core._is_tensor_carrier(value))
                self.assertFalse(jt.core._is_tensor_carrier(object()))
                calls = {'execute': 0, 'grad': 0}

                class CustomVJP(jt.Function):
                    def execute(self, argument):
                        calls['execute'] += 1
                        return argument * argument

                    def grad(self, actual_dy):
                        calls['grad'] += 1
                        return actual_dy * 7.0

                output = CustomVJP.apply(value)
                self.assertIs(type(output), tensor_type)
                self._check(output, (0.0625, 0.25, 2.25, 4.0))
                loss = jt.core.ops.sum(output * dy, 0)
                self._check(loss, (0.3125,))
                gradient, = jt.core.grad(loss, [value], False)
                self.assertIs(type(gradient), tensor_type)
                self.assertEqual(calls, {'execute': 1, 'grad': 1})
                self._check(gradient, (7.0, 14.0, -7.0, 3.5))
                self._check(value, (0.25, -0.5, 1.5, 2.0))
                self._check(dy, (1.0, 2.0, -1.0, 0.5))
                self.assertEqual(jt.core.backend_fallback_count(), 0)
            finally:
                jt.core._restore_construction_placement(token)
        self.assertEqual((native.__bases__, native.__mro__), native_before[:2])
        self.assertEqual(set(native.__dict__), set(native_before[2]))
        self.assertTrue(all(native.__dict__[key] is value
                            for key, value in native_before[2].items()))

    def test_native_Var_custom_VJP(self):
        self._contract(jt.core.Var)

    def test_owned_TensorBase_custom_VJP(self):
        root = jt.core._create_owned_tensor_base_type()
        self.assertEqual(root.__bases__, (object,))
        self.assertFalse(issubclass(root, jt.core.Var))
        self._contract(root)
