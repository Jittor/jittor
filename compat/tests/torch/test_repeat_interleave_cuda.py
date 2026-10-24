"""Real CUDA repeat_interleave regression; values only cross to RAM for checks."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestRepeatInterleaveCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():self.skipTest('real CUDA required')
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        else:scope=nullcontext()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def array(self,value):
        self.assertIs(type(value),torch.Tensor)
        self.assertEqual(str(value.device),'cuda:0')
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual((value.location(),value.device_id,value.placement_backend),('device',0,1))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        return value.detach().clone().cpu().numpy().copy()

    def check(self,value,expected):
        actual=self.array(value)
        self.assertEqual(actual.dtype,expected.dtype);self.assertEqual(actual.shape,expected.shape)
        if expected.dtype.kind in 'biu':np.testing.assert_array_equal(actual,expected)
        else:np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)


    def test_scalar_repeat_values_fp32_int64(self):
        for npdtype in (np.float32,np.int64):
            values=np.arange(6,dtype=npdtype).reshape(2,3)
            source=torch.tensor(values,device='cuda:0')
            for dim in (None,0,1,-1):
                for repeats in (1,2):
                    expected=np.repeat(values,repeats,axis=dim)
                    result=source.repeat_interleave(repeats,dim=dim,output_size=expected.size if dim is None else expected.shape[dim])
                    self.check(result,expected);self.check(source,values)

    def test_scalar_repeat_full_input_vjp(self):
        values=np.arange(6,dtype=np.float32).reshape(2,3)/17
        for dim in (None,0,1):
            for repeats in (1,2):
                source=torch.tensor(values,device='cuda:0',requires_grad=True)
                result=(source*2).repeat_interleave(repeats,dim=dim)
                expected=np.repeat(values*2,repeats,axis=dim)
                dy=(np.arange(expected.size,dtype=np.float32).reshape(expected.shape)-2)/19
                gradient,=torch.autograd.grad(result,(source,),grad_outputs=torch.tensor(dy,device='cuda:0'))
                if dim is None:want=dy.reshape(6,repeats).sum(axis=1).reshape(2,3)*2
                else:
                    shape=list(values.shape);shape.insert(dim+1,repeats)
                    want=dy.reshape(shape).sum(axis=dim+1)*2
                self.check(result,expected);self.check(gradient,want);self.check(source,values)
                self.assertTrue(source.requires_grad);self.assertIsNone(source.grad)

    def test_repeat_one_has_independent_storage_and_no_grad(self):
        values=np.arange(6,dtype=np.float32).reshape(2,3)
        source=torch.tensor(values,device='cuda:0',requires_grad=True)
        for dim in (None,0,1):
            with torch.no_grad():result=source.repeat_interleave(1,dim=dim)
            self.assertIsNot(result,source)
            self.assertNotEqual(result.data_ptr(),source.data_ptr())
            self.assertFalse(result.requires_grad)
            result.zero_()
            self.check(source,values);self.assertTrue(source.requires_grad)
            self.assertIsNone(source.grad)

    def test_invalid_scalar_options(self):
        source=torch.ones((2,3),device='cuda:0')
        with self.assertRaises(RuntimeError):source.repeat_interleave(-1,dim=0)
        with self.assertRaises(IndexError):source.repeat_interleave(1,dim=3)
        with self.assertRaises(RuntimeError):source.repeat_interleave(2,dim=0,output_size=5)
        self.check(source,np.ones((2,3),dtype=np.float32))
