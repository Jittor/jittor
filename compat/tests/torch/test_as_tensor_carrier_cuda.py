"""as_tensor retains owned CUDA identity/aliases and the complete AD graph."""
from contextlib import nullcontext
import ctypes
import unittest
from unittest import mock
import numpy as np
import torch


class TestAsTensorCarrierCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('real CUDA required')
        scope=nullcontext()
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def check(self,value,expected):
        self.assertIs(type(value),torch.Tensor)
        self.assertEqual(str(value.device),'cuda:0')
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync();self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        host=value.detach().clone().cpu().numpy().copy()
        self.assertEqual(host.shape,expected.shape);self.assertEqual(host.dtype,expected.dtype)
        if expected.dtype.kind in 'biu':np.testing.assert_array_equal(host,expected)
        else:np.testing.assert_allclose(host,expected,atol=1e-5,rtol=1e-5)

    def test_integral_tensor_identity_and_alias(self):
        values=np.array([2,0,2],dtype=np.int64)
        x=torch.tensor(values,device='cuda:0');y=torch.as_tensor(x)
        self.assertIs(y,x);self.check(y,values)
        y[1]=7;self.check(x,np.array([2,7,2],dtype=np.int64))

    def test_nonleaf_identity_and_dtype_cast_complete_vjp(self):
        values=np.arange(6,dtype=np.float32).reshape(2,3)/13
        x=torch.tensor(values,device='cuda:0',requires_grad=True);z=x*2
        y=torch.as_tensor(z);self.assertIs(y,z);self.assertFalse(y.is_leaf)
        cast=torch.as_tensor(y,dtype=torch.float64,device='cuda:0')
        self.assertIsNot(cast,y);self.check(cast,(values*2).astype(np.float64))
        grad,=torch.autograd.grad(cast.sum(),(x,));self.check(grad,np.full_like(values,2))
        self.assertIsNone(x.grad);self.assertTrue(x.requires_grad)


class TestSubmitPendingCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "_torch_compat_install_context"):
            raise unittest.SkipTest("Jittor-specific native submission API")
        import jittor as jt
        from jittor.compat.torch import namespace_owner

        if not getattr(jt.compiler, "is_cuda", False) or not torch.cuda.is_available():
            raise unittest.SkipTest("real native CUDA build required")
        if namespace_owner(torch) is not jt or torch.Tensor is jt.Var:
            raise AssertionError("independent Jittor Torch namespace required")
        cls.jt = jt

    def setUp(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        fallback = forbid_backend_fallbacks()
        fallback.__enter__()
        self.addCleanup(fallback.__exit__, None, None, None)
        scope = self.jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback="error")
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        torch.cuda.set_device(0)
        self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def tearDown(self):
        self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def assert_cuda_pointer(self, value):
        value.sync()
        self.assertEqual(value.location(), "device")
        self.assertEqual(int(value.device_id), 0)
        self.assertEqual(tuple(self.jt.core.dispatch_context([value])), ("cuda", 0))
        driver = ctypes.CDLL("libcuda.so.1")
        self.assertEqual(driver.cuInit(0), 0)
        pointer = ctypes.c_void_p(int(value.device_raw_ptr))
        ordinal, memory_type = ctypes.c_int(-1), ctypes.c_int(-1)
        self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, pointer), 0)
        self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2, pointer), 0)
        self.assertEqual((ordinal.value, memory_type.value), (0, 2))

    def native_value(self):
        value = self.jt.Var([2.0, 4.0]).cuda(0)
        self.assertIsInstance(value, self.jt.Var)
        return value

    def owned_value(self, *, requires_grad=False):
        value = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda:0",
                             requires_grad=requires_grad)
        self.assertIs(type(value), torch.Tensor)
        self.assertNotIsInstance(value, self.jt.Var)
        self.assertTrue(self.jt.core._is_tensor_carrier(value))
        return value

    def test_native_var_original_object_and_values(self):
        value = self.native_value()
        self.assertIs(self.jt.submit_pending(value, device_sync=True), value)
        self.assert_cuda_pointer(value)
        np.testing.assert_array_equal(value.numpy(), np.array([2.0, 4.0], dtype=np.float32))

    def test_owned_carrier_original_object_and_complete_vjp(self):
        source = self.owned_value(requires_grad=True)
        value = source * 3.0
        self.assertNotIsInstance(value, self.jt.Var)
        self.assertFalse(value.is_leaf)
        self.assertIs(self.jt.submit_pending(value, device_sync=True), value)
        self.assert_cuda_pointer(value)
        self.assertTrue(value.requires_grad)
        value.sum().backward()
        self.assertIsNotNone(source.grad)
        self.assert_cuda_pointer(source.grad)
        np.testing.assert_array_equal(source.grad.cpu().numpy(), np.array([3.0, 3.0], dtype=np.float32))

    def test_multiple_roots_return_original_objects_and_sync_each_once(self):
        native, owned = self.native_value(), self.owned_value() * 2.0
        # Per-instance spies delegate to the genuine bound native methods;
        # no Tensor/Var class, payload, submission method or math is replaced.
        with mock.patch.object(native, "sync", wraps=native.sync) as native_sync, \
                mock.patch.object(owned, "sync", wraps=owned.sync) as owned_sync:
            returned = self.jt.submit_pending(native, owned, device_sync=True)
            self.assertIsInstance(returned, tuple)
            self.assertEqual(len(returned), 2)
            self.assertIs(returned[0], native)
            self.assertIs(returned[1], owned)
            native_sync.assert_called_once_with()
            owned_sync.assert_called_once_with()
        self.assert_cuda_pointer(native)
        self.assert_cuda_pointer(owned)
        np.testing.assert_array_equal(owned.cpu().numpy(), np.array([2.0, 4.0], dtype=np.float32))

    def test_default_submission_does_not_request_python_device_sync(self):
        value = self.owned_value() * 2.0
        with mock.patch.object(value, "sync", wraps=value.sync) as sync:
            self.assertIs(self.jt.submit_pending(value), value)
            sync.assert_not_called()
        self.assert_cuda_pointer(value)
        np.testing.assert_array_equal(value.cpu().numpy(), np.array([2.0, 4.0], dtype=np.float32))

    def test_foreign_objects_with_forged_methods_are_rejected(self):
        calls = []

        class Foreign:
            def submit_pending(self):
                calls.append("submit")

            def sync(self):
                calls.append("sync")

        for foreign in (object(), Foreign(), None):
            with self.subTest(kind=type(foreign).__name__):
                self.assertFalse(self.jt.core._is_tensor_carrier(foreign))
                with self.assertRaises(TypeError):
                    self.jt.submit_pending(foreign, device_sync=True)
        self.assertEqual(calls, [])

    def test_empty_root_list_keeps_existing_error(self):
        with self.assertRaises(ValueError):
            self.jt.submit_pending(device_sync=True)
