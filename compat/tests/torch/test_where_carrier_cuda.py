"""Real CUDA public where graph preservation and masked reduction backward."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch


class TestWhereCarrierCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('real CUDA required')
        scope=nullcontext()
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)

    def check(self,value,reference):
        self.assertIsNotNone(value);self.assertIs(type(value),torch.Tensor)
        self.assertEqual(value.device,torch.device('cuda:0'))
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync();self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        host=value.detach().clone().cpu().numpy().copy()
        self.assertEqual(host.shape,reference.shape);self.assertEqual(host.dtype,reference.dtype)
        np.testing.assert_allclose(host,reference,atol=1e-5,rtol=1e-5)

    def test_broadcast_complete_backward_preserves_input_flags(self):
        a=np.array([[1,2],[3,4]],np.float32);b=np.array([[-2,5]],np.float32)
        m=np.array([[True,False],[False,True]]);seed=np.array([[1,-2],[.25,.5]],np.float32)
        x=torch.tensor(a,device='cuda:0',requires_grad=True);y=torch.tensor(b,device='cuda:0',requires_grad=True)
        out=torch.where(torch.tensor(m,device='cuda:0'),x,y)
        self.assertTrue(x.requires_grad);self.assertTrue(y.requires_grad);self.assertTrue(out.requires_grad)
        out.backward(torch.tensor(seed,device='cuda:0'))
        self.check(out,np.where(m,a,b));self.check(x.grad,seed*m);self.check(y.grad,(seed*~m).sum(0,keepdims=True))

    def test_scalar_branch_all_parameter_input_gradients_and_update(self):
        a=np.array([[.5,1],[-.5,2]],np.float32);b=np.array([1,-1],np.float32)
        data=np.array([[1,2],[3,-1]],np.float32);mask=np.array([[True,False],[False,True]])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(a,device='cuda:0'))
                self.bias=torch.nn.Parameter(torch.tensor(b,device='cuda:0'))
            def forward(self,x):return torch.nn.functional.linear(x,self.weight,self.bias)
        model=Model();optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        x=torch.tensor(data,device='cuda:0',requires_grad=True);values=model(x)
        out=torch.where(torch.tensor(mask,device='cuda:0'),values,0.)
        self.assertTrue(values.requires_grad);self.assertTrue(out.requires_grad);out.sum().backward()
        dy=mask.astype(np.float32);gw=dy.T@data;gb=dy.sum(0)
        self.check(out,np.where(mask,data@a.T+b,0.).astype(np.float32))
        self.check(model.weight.grad,gw);self.check(model.bias.grad,gb);self.check(x.grad,dy@a)
        optimizer.step();self.check(model.weight.detach(),a-np.float32(.01)*gw)
        self.check(model.bias.detach(),b-np.float32(.01)*gb)

    def test_masked_reduction_backward_and_nan_mask(self):
        a=np.array([[1,np.nan],[3,4]],np.float32);mask=np.array([[1,0],[0,1]],np.float32)
        x=torch.tensor(a,device='cuda:0',requires_grad=True)
        tensor_mask=torch.tensor(mask,device='cuda:0')
        total=(torch.where(tensor_mask.bool(),x,0.)*tensor_mask).sum()
        self.assertTrue(x.requires_grad);self.assertTrue(total.requires_grad)
        total.backward();self.check(total,np.asarray(5.,dtype=np.float32));self.check(x.grad,mask)
