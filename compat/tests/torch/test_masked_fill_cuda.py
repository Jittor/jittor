"""Finite real-CUDA public masked_fill contract, for independent runtimes.

NumPy is an analytic public-fixture reference,
never a same-interpreter Torch oracle or a CPU fallback for the operation.
"""
from contextlib import nullcontext
import unittest

import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestMaskedFillCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():self.skipTest('actual CUDA required')
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            from jittor._core.var import masked_fill as native_masked_fill
            from jittor.compat.torch.installers import numerical as numerical_owner
            import jittor as jt
            self.assertIs(jt.core.Var.masked_fill,native_masked_fill)
            if hasattr(numerical_owner,'_native_masked_fill'):
                self.assertIs(numerical_owner._native_masked_fill,native_masked_fill)
            scope=forbid_backend_fallbacks()
        else:
            scope=nullcontext()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def resident(self,value):
        self.assertIs(type(value),torch.Tensor)
        self.assertEqual(str(value.device),'cuda:0')
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync()
            self.assertTrue(jt.core._is_tensor_carrier(value))
            self.assertFalse(isinstance(value,jt.core.Var))
            self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual((value.location(),value.device_id,value.placement_backend),('device',0,1))
            self.assertEqual(jt.core.backend_fallback_count(),0)

    def array(self,value):
        self.resident(value)
        before=(tuple(value.shape),tuple(value.stride()),value.dtype,value.device,value.requires_grad,value.is_leaf,value.grad_fn is None)
        transport=value.detach().clone()
        self.assertIsNot(transport,value);self.resident(transport)
        if value.numel():self.assertNotEqual(transport.data_ptr(),value.data_ptr())
        result=transport.cpu().numpy().copy()
        self.assertEqual(before,(tuple(value.shape),tuple(value.stride()),value.dtype,value.device,value.requires_grad,value.is_leaf,value.grad_fn is None))
        return result

    def check(self,value,reference):
        actual=self.array(value)
        self.assertEqual(actual.shape,reference.shape)
        self.assertEqual(actual.dtype,reference.dtype)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,reference,atol=1e-5,rtol=1e-5)

    def exercise(self,method):
        for shape,mask_values in (((2,3),[[True,False,True]]),
                                  ((2,2,3),[[[False],[True]]])):
            values=(np.arange(np.prod(shape),dtype=np.float32).reshape(shape)-3)/17
            dy=(np.arange(np.prod(shape),dtype=np.float32).reshape(shape)-2)/19
            condition=np.asarray(mask_values,dtype=np.bool_)
            full_mask=np.broadcast_to(condition,shape)
            x=torch.tensor(values,device='cuda:0',requires_grad=True)
            mask=torch.tensor(condition,device='cuda:0')
            fill=torch.tensor(.375,dtype=torch.float32,device='cuda:0',requires_grad=True)
            source=x*2+.125
            y=source.masked_fill(mask,fill)if method else torch.masked_fill(input=source,mask=mask,value=fill)
            self.assertTrue(y.requires_grad);self.assertFalse(y.is_leaf);self.assertIsNotNone(y.grad_fn)
            gradients=torch.autograd.grad(y,(x,fill),grad_outputs=torch.tensor(dy,device='cuda:0'),allow_unused=False)
            self.assertEqual(len(gradients),2)
            self.check(y,np.where(full_mask,np.float32(.375),values*2+np.float32(.125)))
            self.check(gradients[0],np.where(full_mask,np.float32(0),dy*2))
            self.check(gradients[1],np.asarray(dy[full_mask].sum(dtype=np.float32)))
            self.check(x,values);self.check(mask,condition);self.check(fill,np.asarray(.375,dtype=np.float32))
            self.assertTrue(x.requires_grad and fill.requires_grad)
            self.assertTrue(x.is_leaf and fill.is_leaf)
            self.assertFalse(mask.requires_grad)
            self.assertIsNone(x.grad);self.assertIsNone(fill.grad);self.assertIsNone(mask.grad)

    def test_function_broadcast_full_input_and_scalar_value_vjp(self):self.exercise(False)

    def test_method_broadcast_full_input_and_scalar_value_vjp(self):self.exercise(True)

    def test_no_grad_fp32_int64_and_sources_unchanged(self):
        for dtype,npdtype in ((torch.float32,np.float32),(torch.int64,np.int64)):
            for method in (False,True):
                values=np.arange(6,dtype=npdtype).reshape(2,3)
                mask_values=np.asarray([[True,False,True]],dtype=np.bool_)
                x=torch.tensor(values,dtype=dtype,device='cuda:0',requires_grad=dtype is torch.float32)
                mask=torch.tensor(mask_values,device='cuda:0')
                with torch.no_grad():
                    y=x.masked_fill(mask,-3)if method else torch.masked_fill(x,mask,-3)
                self.assertEqual(y.dtype,dtype);self.assertFalse(y.requires_grad)
                self.assertTrue(y.is_leaf);self.assertIsNone(y.grad_fn)
                self.check(y,np.where(np.broadcast_to(mask_values,values.shape),npdtype(-3),values))
                self.check(x,values);self.check(mask,mask_values)
                self.assertIsNone(x.grad);self.assertIsNone(mask.grad)

    def test_qwen_nonleaf_negative_mask_zero_chain_full_vjp(self):
        shape=(2,2,1,4,3)
        a=(np.arange(np.prod(shape),dtype=np.float32).reshape(shape)%11-5)/31
        b=(np.arange(np.prod(shape),dtype=np.float32).reshape(shape)%13-6)/29
        c=(np.arange(16,dtype=np.float32).reshape(1,1,1,4,4)%7)/41+np.float32(.5)
        dy=(np.arange(64,dtype=np.float32).reshape(2,2,1,4,4)%17-8)/37
        left=torch.tensor(a,device='cuda:0',requires_grad=True)
        key=torch.tensor(b,device='cuda:0',requires_grad=True)
        decay=torch.tensor(c,device='cuda:0',requires_grad=True)
        mask=torch.triu(torch.ones((4,4),dtype=torch.bool,device='cuda:0'),diagonal=0)
        output=-((left@key.transpose(-1,-2))*decay).masked_fill(mask,0)
        self.assertTrue(output.requires_grad);self.assertFalse(output.is_leaf)
        gradients=torch.autograd.grad(output,(left,key,decay),grad_outputs=torch.tensor(dy,device='cuda:0'),allow_unused=False)
        product=a@b.swapaxes(-1,-2);condition=np.triu(np.ones((4,4),dtype=np.bool_))
        expected=-np.where(condition,np.float32(0),product*c)
        dproduct=np.where(condition,np.float32(0),-dy)*c
        self.check(output,expected)
        for grad,reference in zip(gradients,(dproduct@b,dproduct.swapaxes(-1,-2)@a,
            (np.where(condition,np.float32(0),-dy)*product).sum(axis=(0,1,2),keepdims=True))):
            self.check(grad,reference)
        for tensor,reference in ((left,a),(key,b),(decay,c)):
            self.check(tensor,reference);self.assertIsNone(tensor.grad)
        self.check(mask,condition);self.assertFalse(mask.requires_grad);self.assertIsNone(mask.grad)

    def test_invalid_options_are_not_silently_swallowed(self):
        x=torch.ones((2,3),device='cuda:0');mask=torch.ones((1,3),dtype=torch.bool,device='cuda:0')
        with self.assertRaises(RuntimeError):torch.masked_fill(x,mask.to(torch.float32),0)
        with self.assertRaises(RuntimeError):x.masked_fill(mask,torch.ones((1,),device='cuda:0'))
        with self.assertRaises(TypeError):x.masked_fill(mask,0,device='cuda')
        if hasattr(torch,'_torch_compat_install_context'):
            out=torch.ones((2,3),device='cuda:0')
            with self.assertRaises(NotImplementedError):torch.masked_fill(x,mask,0,out=out)
            self.check(out,np.ones((2,3),dtype=np.float32))
        self.check(x,np.ones((2,3),dtype=np.float32))
        self.check(mask,np.ones((1,3),dtype=np.bool_))


if __name__=='__main__':unittest.main()
