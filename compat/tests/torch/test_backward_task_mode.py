"""Public CUDA backward task-mode regression; no seed/history or PPO claim.

Run the same test independently under native Torch and the compatibility namespace.
CUDA absence is an explicit skip, never CUDA support evidence.
"""
from contextlib import contextmanager, nullcontext
import os
import sys
import unittest

X = [-2., -1., .5, 2.]
# L=sum((x*x)^2 + (x*x)/8): dy=2*x*x+1/8, dx=4*x^3+x/4.
# These fixed literals must pass independently in real native Torch first.
Y = [4., 1., .25, 4.]
DY = [8.125, 2.125, .625, 8.125]
DX = [-32.5, -4.25, .625, 32.5]
D2X = [48.25, 12.25, 3.25, 48.25]


class HookFailure(RuntimeError): pass


class TestBackwardTaskMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = None
        if os.environ.get('JITTOR_TORCH_SHIM') == '1':
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        else:
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)
        import numpy as np
        cls.np = np

    @contextmanager
    def cuda(self):
        if not self.torch.cuda.is_available(): self.skipTest("CUDA unavailable; no CUDA support claim")
        self.assertEqual(self.torch.cuda.current_device(), 0)
        jt = self.jt
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            if jt is not None: self.assertEqual(jt.core.backend_fallback_count(), 0)
            try: yield
            finally:
                if jt is not None: self.assertEqual(jt.core.backend_fallback_count(), 0)

    def resident(self, value):
        self.assertIsInstance(value, self.torch.Tensor)
        self.assertEqual((value.device.type, value.device.index), ('cuda', 0))
        if self.jt is not None:
            native = getattr(value, '_var', value); native.sync()
            self.assertEqual(self.jt.core.dispatch_context([native]), ('cuda',0))
            self.assertEqual(int(native.placement_backend), 1)
            self.assertEqual(native.location(), 'device')

    def snapshot(self, value):
        self.resident(value)
        with self.torch.no_grad():
            copy = value.clone(); self.assertIsNot(copy, value); self.assertFalse(copy.requires_grad)
            self.resident(copy)
            result = self.np.array(copy.cpu().numpy(), copy=True)
        self.assertEqual(result.dtype, self.np.dtype('float32'))
        self.assertTrue(bool(self.np.isfinite(result).all()), 'nonfinite complete public tensor')
        return result

    def equal(self, value, expected):
        actual = self.snapshot(value)
        reference = self.np.asarray(expected,dtype=self.np.float32)
        self.assertEqual(actual.shape, reference.shape)
        delta = self.np.abs(actual.astype(self.np.float64)-reference)
        self.assertEqual(int(self.np.count_nonzero(delta>1e-5+1e-5*self.np.abs(reference))),0,'complete gradient differs')

    def graph(self):
        # Never build the target graph under the case's outer no_grad context.
        with self.torch.enable_grad():
            x = self.torch.tensor(X,dtype=self.torch.float32,device='cuda:0',requires_grad=True)
            y = x*x
            loss = (y*y).sum()+(y*.125).sum()
        return x,y,loss

    def test_scalar_backward_hook_task_mode_cuda(self):
        t=self.torch
        with self.cuda():
            for outer in (False, True):
                for cg in (False, True):
                    with self.subTest(outer=outer,create_graph=cg):
                        entry=t.is_grad_enabled(); x,y,loss=self.graph(); before=self.snapshot(x); events={}; handles=[]
                        def callback(name):
                            def observe(gradient):
                                self.assertNotIn(name,events)
                                flags=(gradient.requires_grad,gradient.is_leaf,gradient.grad_fn is None)
                                events[name]=(t.is_grad_enabled(),flags,self.snapshot(gradient))
                                self.assertEqual(flags,(gradient.requires_grad,gradient.is_leaf,gradient.grad_fn is None))
                                return None
                            return observe
                        try:
                            handles=[y.register_hook(callback('y')),x.register_hook(callback('x'))]
                            with t.set_grad_enabled(outer):
                                loss.backward(create_graph=cg)
                                self.assertIs(t.is_grad_enabled(),outer)
                            self.assertIs(t.is_grad_enabled(),entry)
                            self.assertEqual(set(events),{'x','y'})
                            for name in ('x','y'):
                                self.assertIs(events[name][0],cg)
                                self.assertEqual(events[name][1],(cg,not cg,not cg))
                                expected=DX if name=='x' else DY
                                ref=self.np.asarray(expected,dtype=self.np.float32)
                                self.assertEqual(events[name][2].shape,ref.shape)
                                self.assertTrue(bool((self.np.abs(events[name][2]-ref)<=1e-5+1e-5*self.np.abs(ref)).all()),'hook full values differ')
                            self.equal(y,Y); self.assertIsNotNone(x.grad); self.equal(x.grad,DX)
                            self.assertEqual(self.snapshot(x).tobytes(),before.tobytes())
                            self.assertTrue(x.requires_grad and x.is_leaf and x.grad_fn is None)
                        finally:
                            for handle in reversed(handles): handle.remove()
                            self.assertIs(t.is_grad_enabled(),entry)

    def test_hook_exception_restores_outer_mode_cuda(self):
        t=self.torch
        with self.cuda():
            for outer in (False,True):
                for cg in (False,True):
                    with self.subTest(outer=outer,create_graph=cg):
                        entry=t.is_grad_enabled(); x,y,loss=self.graph(); observed=[]
                        def failure(gradient):
                            observed.append(t.is_grad_enabled())
                            raise HookFailure('intentional public hook failure')
                        handle=y.register_hook(failure)
                        try:
                            with t.set_grad_enabled(outer):
                                with self.assertRaises(HookFailure): loss.backward(create_graph=cg)
                                self.assertIs(t.is_grad_enabled(),outer)
                            self.assertIs(t.is_grad_enabled(),entry)
                            self.assertEqual(observed,[cg])
                        finally: handle.remove()

    def test_create_graph_second_derivative_cuda(self):
        t=self.torch
        with self.cuda():
            for outer in (False,True):
                with self.subTest(outer=outer):
                    entry=t.is_grad_enabled(); x,y,loss=self.graph()
                    with t.set_grad_enabled(outer):
                        loss.backward(create_graph=True)
                        self.assertIs(t.is_grad_enabled(),outer)
                    self.assertIs(t.is_grad_enabled(),entry)
                    self.assertIsNotNone(x.grad); self.assertTrue(x.grad.requires_grad)
                    self.equal(x.grad,DX)
                    with t.enable_grad(): second_target=x.grad.sum()
                    second,=t.autograd.grad(second_target,(x,),create_graph=False)
                    self.assertFalse(second.requires_grad); self.equal(second,D2X)
                    x.grad=None  # Break the public create_graph backward leaf/gradient cycle.


if __name__=='__main__':
    unittest.main(verbosity=2)
