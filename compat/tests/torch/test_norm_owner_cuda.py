"""Real backward/clip CUDA ownership controls; no fabricated gradient values."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch



class TestNormOwnerCUDA(unittest.TestCase):
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

    def backward(self):
        weights=np.array([[.5,1],[-.5,2]],dtype=np.float32)
        bias=np.array([1,-1],dtype=np.float32)
        data=np.array([[1,2],[3,-1]],dtype=np.float32)
        seed=np.array([[1,-.5],[.25,2]],dtype=np.float32)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor(weights,device='cuda:0'))
                self.bias=torch.nn.Parameter(torch.tensor(bias,device='cuda:0'))
            def forward(self,x):return torch.nn.functional.linear(x,self.weight,self.bias)
        model=Tiny();opt=torch.optim.SGD(model.parameters(),lr=.01)
        x=torch.tensor(data,device='cuda:0',requires_grad=True)
        (model(x)*torch.tensor(seed,device='cuda:0')).sum().backward()
        expected=(seed.T@data,seed.sum(axis=0));params=list(model.parameters())
        self.assertEqual(len(params),2)
        for p,g in zip(params,expected):self.assertIsNotNone(p.grad);self.check(p.grad,g)
        self.check(x.grad,seed@weights)
        norm=np.sqrt(sum(np.sum(g*g)for g in expected)).astype(np.float32)
        return model,opt,x,params,expected,norm,(weights,bias,seed@weights)


    def test_public_total_norm_is_owned_and_to_integer(self):
        from torch.nn.utils.clip_grad import _get_total_norm
        _,_,_,params,_,norm,_=self.backward()
        total=_get_total_norm([p.grad for p in params],2.0,False,False)
        self.check(total,np.asarray(norm));self.check(total.to(0,non_blocking=True),np.asarray(norm))

    def test_public_clip_all_gradients_and_sgd_update(self):
        from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_
        _,opt,x,params,expected,norm,initial=self.backward()
        total=_get_total_norm([p.grad for p in params],2.0,True,False)
        _clip_grads_with_norm_(params,.7,total,False)
        self.check(total,np.asarray(norm));coef=np.float32(.7)/(norm+np.float32(1e-6))
        for p,g in zip(params,expected):self.check(p.grad,g*coef)
        self.check(x.grad,initial[2]);opt.step()
        for p,value,g in zip(params,initial[:2],expected):self.check(p.detach(),value-np.float32(.01)*g*coef)



    def test_empty_gradient_list_returns_owned_cuda_scalar(self):
        from torch.nn.utils.clip_grad import _get_total_norm
        old=torch.get_default_device()
        torch.set_default_device('cuda:0')
        try:
            total=_get_total_norm([],2.0,False,False)
            self.check(total,np.asarray(0,dtype=np.float32))
            self.check(total.to(0,non_blocking=True),np.asarray(0,dtype=np.float32))
        finally:
            torch.set_default_device(old)

    def test_single_tensor_norm_and_parameter_clipping(self):
        from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_
        _,_,_,params,expected,_,_=self.backward()
        target=params[0];ref=expected[0]
        norm=np.linalg.norm(ref).astype(np.float32)
        total=_get_total_norm(target.grad,2.0,True,False)
        self.check(total,np.asarray(norm))
        _clip_grads_with_norm_(target,.7,total,False)
        coef=np.float32(.7)/(norm+np.float32(1e-6))
        self.check(target.grad,ref*coef)
        self.check(params[1].grad,expected[1])


    def test_multiple_optimizer_clipping_public_gradient_ownership(self):
        """CUDA backward, public grad=None, clipping aliases and actual SGD."""
        import ctypes
        candidate = hasattr(torch, '_torch_compat_install_context')
        if candidate:
            import jittor as jt
            from jittor.compat.diagnostics import counts
            runtime = jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error')
        else:
            counts = lambda: {}
            runtime = nullcontext()
        from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(driver.cuInit(0), 0)

        def resident(value):
            if candidate:
                value.sync()
                self.assertEqual(tuple(jt.core.dispatch_context([value])), ('cuda', 0))
                self.assertEqual(value.location(), 'device')
            else:
                torch.cuda.synchronize()
            self.assertEqual(str(value.device), 'cuda:0')
            pointer = int(value.device_raw_ptr if candidate else value.data_ptr())
            ordinal, memory = ctypes.c_int(-1), ctypes.c_int(-1)
            self.assertGreater(pointer, 0)
            self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
            self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(memory), 2, ctypes.c_void_p(pointer)), 0)
            self.assertEqual((ordinal.value, memory.value), (0, 2))
            return pointer

        initial_counts = counts()
        self.runtime_evidence = dict(scope='actual_CUDA_multi_optimizer_clip_NOT_PPO', cases=[],
            startup_all_swallowed_zero=not sum(initial_counts.values()))
        with runtime, torch.enable_grad(), torch.device('cuda:0'):
            specs = [(dtype, operation, False, False)
                     for dtype in (torch.float32, torch.bfloat16)
                     for operation in ('split_norm', 'public_norm')]
            specs += [(torch.float32, 'public_norm', True, False)]
            if candidate:
                specs += [(torch.float32, 'public_norm', False, True)]
            for dtype, operation, reverse, native_backward in specs:
                with self.subTest(dtype=str(dtype), operation=operation, reverse=reverse, native_backward=native_backward):
                    p = torch.nn.Parameter(torch.tensor([.25, -.5], dtype=dtype, device='cuda:0'))
                    q = torch.nn.Parameter(torch.tensor([.5, -.25], dtype=dtype, device='cuda:0'))
                    unused = torch.nn.Parameter(torch.ones((2,), dtype=dtype, device='cuda:0'))
                    if reverse:
                        foreign = torch.optim.SGD([q], lr=.125)
                        owner = torch.optim.SGD([p, unused], lr=.125)
                    else:
                        owner = torch.optim.SGD([p] if native_backward else [p, unused], lr=.125)
                        foreign = torch.optim.SGD([q], lr=.125)
                    seed = torch.tensor([3., 4.], dtype=dtype, device='cuda:0')
                    foreign_seed = torch.tensor([5., 12.], dtype=dtype, device='cuda:0')
                    if native_backward:
                        owner.backward((p * seed).sum())
                    else:
                        ((p * seed).sum() + (q * foreign_seed).sum()).backward()
                    gradient = p.grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(bool(torch.equal(gradient, seed)))
                    unused.grad = torch.ones_like(unused)
                    unused.grad = None
                    alias = gradient.detach()
                    pointer = resident(gradient)
                    self.assertEqual(resident(alias), pointer)
                    foreign_grad = q.grad
                    foreign_copy = foreign_grad.clone() if foreign_grad is not None else None
                    old_p, old_q, old_unused = p.detach().clone(), q.detach().clone(), unused.detach().clone()
                    expected = (seed.clamp(-1., 1.) if operation == 'public_value'
                                else seed * (1. / (torch.tensor(5., device='cuda:0') + 1e-6)).to(dtype))
                    expected_step = (old_p.float() - expected.float() * .125).to(dtype)
                    before_clip = counts()
                    if operation == 'split_norm':
                        total = _get_total_norm([gradient], 2., False, None)
                        _clip_grads_with_norm_([p, unused], 1., total, None)
                    elif operation == 'public_norm':
                        total = torch.nn.utils.clip_grad_norm_([p, unused], 1.)
                    else:
                        total = None
                        torch.nn.utils.clip_grad_value_([p, unused], 1.)
                    self.assertEqual(counts(), before_clip)
                    resident(expected)
                    self.assertIs(p.grad, gradient)
                    self.assertEqual(resident(gradient), pointer)
                    self.assertEqual(resident(alias), pointer)
                    self.assertTrue(bool(torch.equal(gradient, expected)))
                    self.assertTrue(bool(torch.equal(alias, expected)))
                    if total is not None:
                        resident(total)
                        self.assertTrue(bool(torch.equal(total, torch.tensor(5., device='cuda:0'))))
                    self.assertIsNone(unused.grad)
                    self.assertIs(q.grad, foreign_grad)
                    if foreign_grad is not None:
                        resident(foreign_grad)
                        self.assertTrue(bool(torch.equal(foreign_grad, foreign_copy)))
                    owner.step()
                    resident(p); resident(expected_step)
                    self.assertTrue(bool(torch.equal(p, expected_step)))
                    self.assertTrue(bool(torch.equal(q, old_q)))
                    self.assertTrue(bool(torch.equal(unused, old_unused)))
                    self.assertIsNone(unused.grad)
                    self.assertEqual(counts(), before_clip)
                    self.runtime_evidence['cases'].append(dict(dtype=str(dtype), operation=operation,
                        reverse_registration=reverse, native_optimizer_backward=native_backward, passed=True))
        self.assertEqual(counts(), initial_counts)
        self.runtime_evidence['new_swallowed_count'] = 0
        if candidate:
            self.assertEqual(jt.core.backend_fallback_count(), 0)
            self.runtime_evidence['fallback_count'] = 0
