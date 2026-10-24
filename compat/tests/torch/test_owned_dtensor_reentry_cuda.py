"""True-CUDA custom Parameter re-entry and complete two-micro accumulation."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

class TestOwnedReentryCUDA(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        scope=nullcontext()
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)
        if not torch.distributed.is_initialized():
            d=tempfile.TemporaryDirectory(prefix='owned-reentry-cuda-');self.addCleanup(d.cleanup)
            torch.distributed.init_process_group('nccl',init_method=Path(d.name,'rdzv').as_uri(),rank=0,world_size=1,timeout=timedelta(seconds=120))
            self.addCleanup(torch.distributed.destroy_process_group)
        from torch.distributed.device_mesh import init_device_mesh
        self.mesh=init_device_mesh('cuda',(1,))

    def check(self,value,reference):
        self.assertIsNotNone(value);self.assertEqual(value.device,torch.device('cuda:0'))
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            value.sync();self.assertEqual(jt.core.dispatch_context([value]),('cuda',0))
            self.assertEqual(jt.core.backend_fallback_count(),0)
        host=value.detach().clone().cpu().numpy().copy()
        self.assertEqual(host.shape,reference.shape);self.assertEqual(host.dtype,reference.dtype)
        np.testing.assert_allclose(host,reference,atol=1e-5,rtol=1e-5)

    def test_repeated_parameter_keeps_real_dtensor_and_original_leaf(self):
        from torch.distributed.tensor import DTensor,Shard
        x=torch.tensor([[1.,2.],[3.,-1.]],device='cuda:0',requires_grad=True)
        d=DTensor.from_local(x,self.mesh,(Shard(0),),run_check=False)
        p=torch.nn.Parameter(d);again=torch.nn.Parameter(p,requires_grad=False)
        self.assertIs(type(again),DTensor);self.assertIsNot(again,p)
        self.assertIsInstance(again,torch.nn.Parameter);self.assertTrue(again.is_leaf)
        self.assertFalse(again.requires_grad);self.assertTrue(p.requires_grad)
        self.assertEqual(again.to_local().data_ptr(),p.to_local().data_ptr())
        (p.to_local()*2).sum().backward()
        self.assertIs(type(p.grad),DTensor)
        self.check(p.grad.to_local(),np.full((2,2),2,dtype=np.float32));self.assertIsNone(again.grad)

    def test_two_reduced_micros_publish_complete_accumulated_local_gradient(self):
        from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.5,1.],[-.5,2.]],device='cuda:0'))
                self.bias=torch.nn.Parameter(torch.tensor([1.,-1.],device='cuda:0'))
            def forward(self,x):return torch.nn.functional.linear(x,self.weight,self.bias)
        model=Model()
        fully_shard(model,mesh=self.mesh,mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32,reduce_dtype=torch.float32),reshard_after_forward=True)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        model(torch.tensor([[1.,2.],[3.,-1.]],device='cuda:0',requires_grad=True)).sum().backward()
        model(torch.tensor([[2.,-1.],[1.,3.]],device='cuda:0',requires_grad=True)).sum().backward()
        expected={'weight':np.array([[7,3],[7,3]],dtype=np.float32),'bias':np.array([4,4],dtype=np.float32)}
        for name,p in model.named_parameters():
            with torch.no_grad():
                self.check(p.grad.to_local(),expected[name]);self.check(p.grad.full_tensor(),expected[name])
        optimizer.step()
        wanted={'weight':np.array([[.43,.97],[-.57,1.97]],dtype=np.float32),'bias':np.array([.96,-1.04],dtype=np.float32)}
        for name,p in model.named_parameters():
            with torch.no_grad():self.check(p.to_local(),wanted[name])
