"""Owned CUDA unbind, real QKV chain and view write-through controls."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch

class TestUnbindCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("real CUDA required")
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        else:scope=nullcontext()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def check(self,value,expected):
        self.assertIs(type(value),torch.Tensor);self.assertEqual(str(value.device),'cuda:0')
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync();self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual((value.location(),value.device_id,value.placement_backend),('device',0,1))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        got=value.detach().clone().cpu().numpy().copy();self.assertEqual(got.shape,expected.shape)
        self.assertEqual(got.dtype,expected.dtype)
        if expected.dtype.kind in 'biu':np.testing.assert_array_equal(got,expected)
        else:np.testing.assert_allclose(got,expected,atol=1e-5,rtol=1e-5)

    def test_method_values_and_full_input_vjp(self):
        x=np.arange(6,dtype=np.float32).reshape(2,3)/13;dy=(np.arange(6,dtype=np.float32).reshape(2,3)-2)/17
        for dim in (0,1,-1):
            with self.subTest(dim=dim):
                tx=torch.tensor(x,device='cuda:0',requires_grad=True)
                parts=(tx*2).unbind(dim=dim);self.assertIs(type(parts),tuple)
                refs=np.moveaxis(x*2,dim,0);seeds=np.moveaxis(dy,dim,0)
                self.assertEqual(len(parts),len(refs))
                for part,ref in zip(parts,refs):self.check(part,ref)
                grads=torch.autograd.grad(parts,(tx,),grad_outputs=tuple(torch.tensor(v,device='cuda:0')for v in seeds))
                self.check(grads[0],dy*2);self.check(tx,x);self.assertIsNone(tx.grad)

    def test_qwen_qkv_chain_full_input_weight_bias_vjp(self):
        x=np.arange(8,dtype=np.float32).reshape(2,4)/13
        w=(np.arange(48,dtype=np.float32).reshape(12,4)-12)/43
        b=(np.arange(12,dtype=np.float32)-4)/29
        dy=(np.arange(24,dtype=np.float32).reshape(2,12)-9)/31
        tx,tw,tb=[torch.tensor(v,device='cuda:0',requires_grad=True)for v in(x,w,b)]
        qkv=torch.nn.functional.linear(tx,tw,tb).reshape(2,3,2,2).permute(1,0,2,3).unbind(0)
        self.assertIs(type(qkv),tuple);self.assertEqual(len(qkv),3)
        refs=(x@w.T+b).reshape(2,3,2,2).transpose(1,0,2,3)
        seeds=dy.reshape(2,3,2,2).transpose(1,0,2,3)
        for part,ref in zip(qkv,refs):self.check(part,ref)
        grads=torch.autograd.grad(qkv,(tx,tw,tb),grad_outputs=tuple(torch.tensor(v,device='cuda:0')for v in seeds))
        for got,ref in zip(grads,(dy@w,dy.T@x,dy.sum(axis=0))):self.check(got,ref)
        for tensor,ref in zip((tx,tw,tb),(x,w,b)):
            self.check(tensor,ref);self.assertIsNone(tensor.grad);self.assertTrue(tensor.requires_grad)

    def test_strided_views_writeback_no_grad_empty_and_errors(self):
        for dt in(np.float32,np.int64):
            values=np.arange(12,dtype=dt).reshape(3,4)
            source=torch.tensor(values,device='cuda:0')
            parts=source.transpose(0,1).unbind(0)
            self.assertIs(type(parts),tuple)
            for i,part in enumerate(parts):
                self.check(part,values[:,i]);self.assertTrue(part._is_view())
                self.assertEqual(part.data_ptr(),source.data_ptr()+i*source.element_size())
            parts[1].fill_(21);expected=values.copy();expected[:,1]=21;self.check(source,expected)
            empty=torch.tensor(np.empty((0,3),dtype=dt),device='cuda:0')
            self.assertEqual(empty.unbind(0),())
        source=torch.ones((2,3),device='cuda:0',requires_grad=True)
        with torch.no_grad():parts=source.unbind(0)
        # Torch views created in no_grad still reflect the leaf's grad eligibility.
        for part in parts:self.assertTrue(part.requires_grad)
        self.assertTrue(source.requires_grad);self.assertIsNone(source.grad)
        with self.assertRaises(IndexError):source.unbind(2)
        with self.assertRaises(TypeError):source.unbind(1.5)
