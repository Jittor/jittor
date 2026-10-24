"""Real CUDA split views regression; values only cross to RAM for checks."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestSplitViewsCUDA(unittest.TestCase):
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


    def vjp(self,method):
        values=np.arange(6,dtype=np.float32).reshape(2,3)/17
        dy=(np.arange(6,dtype=np.float32).reshape(2,3)-2)/19
        source=torch.tensor(values,device='cuda:0',requires_grad=True)
        scaled=source*2
        parts=scaled.split([1,1],dim=0)if method else torch.split(scaled,[1,1],dim=0)
        loss=sum((part*torch.tensor(dy[i:i+1],device='cuda:0')).sum()for i,part in enumerate(parts))
        gradient,=torch.autograd.grad(loss,(source,))
        self.check(gradient,dy*2);self.check(source,values)
        for i,part in enumerate(parts):self.check(part,(values*2)[i:i+1])
        self.assertTrue(source.requires_grad);self.assertIsNone(source.grad)

    def test_existing_functional_values_full_vjp(self):self.vjp(False)

    def test_method_values_zero_sections_and_dimensions(self):
        for npdtype in(np.float32,np.int64,np.uint8):
            values=np.arange(12,dtype=npdtype).reshape(2,6)
            source=torch.tensor(values,device='cuda:0')
            for dim,sections in((0,[1,1]),(1,[2,0,4]),(-1,[2,0,4])):
                parts=source.split(sections,dim=dim)
                refs=np.split(values,np.cumsum(sections)[:-1],axis=dim)
                self.assertIs(type(parts),tuple);self.assertEqual(len(parts),len(refs))
                for part,ref in zip(parts,refs):
                    self.check(part,ref);self.assertTrue(part._is_view())
                self.check(source,values)
            parts=source.split(4,dim=1)
            for part,ref in zip(parts,(values[:,:4],values[:,4:])):
                self.check(part,ref);self.assertTrue(part._is_view())
            empty=torch.tensor(np.empty((0,6),dtype=npdtype),device='cuda:0')
            for size in(0,1):
                parts=empty.split(size,dim=0)
                self.assertEqual(len(parts),1);self.assertTrue(parts[0]._is_view())
                self.check(parts[0],np.empty((0,6),dtype=npdtype))

    def test_method_full_input_vjp(self):self.vjp(True)

    def test_tensordict_uint8_storage_views_writeback(self):
        values=np.arange(12,dtype=np.uint8)
        source=torch.tensor(values,device='cuda:0')
        first,empty,last=source.split([3,0,9])
        for part,ref in zip((first,empty,last),(values[:3],values[3:3],values[3:])):
            self.check(part,ref);self.assertTrue(part._is_view())
        self.assertEqual(first.data_ptr(),source.data_ptr())
        self.assertEqual(last.data_ptr(),source.data_ptr()+3)
        last.fill_(21)
        expected=values.copy();expected[3:]=21
        self.check(source,expected);self.check(first,expected[:3]);self.check(last,expected[3:])

    def test_method_strided_views_writeback(self):
        values=np.arange(12,dtype=np.float32).reshape(2,6)
        source=torch.tensor(values,device='cuda:0')
        first,last=source.split([2,4],dim=1)
        self.assertTrue(first._is_view());self.assertTrue(last._is_view())
        self.assertEqual(last.data_ptr(),source.data_ptr()+2*source.element_size())
        first.fill_(19)
        expected=values.copy();expected[:,:2]=19
        self.check(source,expected);self.check(last,expected[:,2:])

    def test_invalid_method_options(self):
        source=torch.ones((2,3),device='cuda:0')
        with self.assertRaises(RuntimeError):source.split(0)
        with self.assertRaises(RuntimeError):source.split(-1)
        with self.assertRaises(RuntimeError):source.split([1],dim=0)
        with self.assertRaises(RuntimeError):source.split([3,-1],dim=0)
        with self.assertRaises(IndexError):source.split(1,dim=2)
        self.check(source,np.ones((2,3),dtype=np.float32))
