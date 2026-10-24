"""Real public DTensor objects, native storage and complete VJPs on CUDA."""
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch


class TestOwnedDTensorCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():self.skipTest('real CUDA required')
        scope=nullcontext()
        if hasattr(torch,'_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope=forbid_backend_fallbacks()
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)
        if not torch.distributed.is_initialized():
            d=tempfile.TemporaryDirectory(prefix='owned-dtensor-cuda-');self.addCleanup(d.cleanup)
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

    def tensor(self):
        a=np.array([[1,2],[3,-1]],dtype=np.float32)
        return a,torch.tensor(a,device='cuda:0',requires_grad=True)

    def test_real_subclass_from_local_complete_vjp(self):
        from torch.distributed.tensor import DTensor,Shard
        a,x=self.tensor();d=DTensor.from_local(x,self.mesh,(Shard(0),),run_check=False)
        self.assertIs(type(d),DTensor);self.assertEqual(DTensor.__bases__,(torch.Tensor,))
        self.assertEqual(DTensor.__mro__,(DTensor,torch.Tensor,torch.Tensor.__bases__[0],object))
        self.assertIs(type(x),torch.Tensor);self.assertFalse(isinstance(x,DTensor));self.assertTrue(x.requires_grad)
        local=d.to_local();(local*2).sum().backward()
        self.check(local,a);self.check(x.grad,np.full_like(a,2))

    def test_custom_parameter_remains_real_dtensor_and_gets_gradient(self):
        from torch.distributed.tensor import DTensor,Shard
        a,x=self.tensor();d=DTensor.from_local(x,self.mesh,(Shard(0),),run_check=False)
        p=torch.nn.Parameter(d)
        self.assertIs(type(p),DTensor);self.assertIsInstance(p,torch.nn.Parameter)
        self.assertFalse(isinstance(d,torch.nn.Parameter));self.assertTrue(p.is_leaf)
        self.assertTrue(p.requires_grad);self.assertTrue(x.requires_grad)
        (p.to_local()*3).sum().backward()
        self.assertIs(type(p.grad),DTensor);self.check(p.grad.to_local(),np.full_like(a,3))
        self.assertIsNone(x.grad)

    def test_detach_clone_and_no_grad_local_have_real_storage(self):
        from torch.distributed.tensor import DTensor,Shard
        a,x=self.tensor();d=DTensor.from_local(x,self.mesh,(Shard(0),),run_check=False)
        stopped=d.detach();copied=stopped.clone()
        self.assertIs(type(stopped),DTensor);self.assertIs(type(copied),DTensor)
        self.assertFalse(stopped.requires_grad);self.assertFalse(copied.requires_grad)
        with torch.no_grad():
            original=d.to_local();alias=stopped.to_local();other=copied.to_local()
        self.assertTrue(original.requires_grad);self.assertTrue(d.requires_grad)
        self.assertEqual(original.data_ptr(),alias.data_ptr());self.assertNotEqual(alias.data_ptr(),other.data_ptr())
        self.check(original,a);self.check(alias,a);self.check(other,a)
