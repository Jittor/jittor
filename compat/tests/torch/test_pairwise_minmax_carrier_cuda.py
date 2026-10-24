"""CUDA min/max overload and original VERL tensor-bounded clip VJPs."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch


class TestPairwiseMinMaxCarrierCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('real CUDA required')
        scope=nullcontext()
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def check(self,v,ref):
        self.assertIs(type(v),torch.Tensor);self.assertEqual(str(v.device),'cuda:0')
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            v.sync();self.assertEqual(jt.core.dispatch_context([v]),('cuda',0))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        host=v.detach().clone().cpu().numpy().copy()
        self.assertEqual(host.shape,ref.shape);self.assertEqual(host.dtype,ref.dtype)
        np.testing.assert_allclose(host,ref,atol=1e-5,rtol=1e-5)

    def pairwise(self,kind):
        a=np.array([[1,2,4],[-2,3,1]],dtype=np.float32)
        b=np.array([[2,2,0]],dtype=np.float32)
        dy=np.array([[1,-2,3],[4,2,-1]],dtype=np.float32)/7
        x,y=[torch.tensor(v,device='cuda:0',requires_grad=True)for v in(a,b)]
        if kind=='min':op=torch.min;expected=np.minimum(a,b);mask=(a<b).astype(np.float32)
        else:op=torch.max;expected=np.maximum(a,b);mask=(a>b).astype(np.float32)
        mask+=(a==b).astype(np.float32)*0.5
        out=op(x,y);self.check(out,expected)
        gx,gy=torch.autograd.grad(out,(x,y),grad_outputs=torch.tensor(dy,device='cuda:0'))
        self.check(gx,mask*dy);self.check(gy,((1-mask)*dy).sum(axis=0,keepdims=True))
        self.assertIsNone(x.grad);self.assertIsNone(y.grad)

    def test_positional_minimum_complete_broadcast_vjp(self):self.pairwise('min')
    def test_positional_maximum_complete_broadcast_vjp(self):self.pairwise('max')

    def test_single_trainable_operand_and_no_grad_ties(self):
        a=np.array([1,2],dtype=np.float32);b=np.array([1,0],dtype=np.float32)
        for kind in ('min','max'):
            for swapped in (False,True):
                with self.subTest(kind=kind,swapped=swapped):
                    x=torch.tensor(a,device='cuda:0',requires_grad=True)
                    constant=torch.tensor(b,device='cuda:0')
                    op=torch.min if kind=='min'else torch.max
                    args=(constant,x)if swapped else(x,constant)
                    output=op(*args)
                    ref=np.minimum(a,b)if kind=='min'else np.maximum(a,b)
                    self.check(output,ref)
                    grad,=torch.autograd.grad(output.sum(),(x,))
                    expected=np.array([0.5,0 if kind=='min'else 1],dtype=np.float32)
                    self.check(grad,expected)
                    self.assertIsNone(x.grad);self.assertFalse(constant.requires_grad)
                    with torch.no_grad():no_grad_output=op(*args)
                    self.check(no_grad_output,ref)
                    self.assertFalse(no_grad_output.requires_grad);self.assertTrue(x.requires_grad)
