"""Public parameter-owned gradients survive optimizer construction.

Run the same CPU/CUDA node independently in Torch and Jittor. Full arrays
remain in RAM and are checked at fixed atol=rtol=1e-5; this is not full PPO.
"""
from contextlib import ExitStack, contextmanager
import os
import sys
import unittest

import numpy as np


class TestOptimizerLateInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = sys.modules.get("jittor")
        if cls.jt is None:
            assert hasattr(torch, "_C")
            assert not any(n.startswith("jittor.") for n in sys.modules)
        else:
            from jittor.compat.torch.publication import namespace_owner
            assert namespace_owner(torch) is cls.jt and torch.Tensor is not cls.jt.Var

    @contextmanager
    def _scope(self, device):
        selected = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        if selected not in ("cpu", "cuda"):
            raise ValueError("one CPU or CUDA node per process required")
        if selected != device:
            self.skipTest("node not selected by JITTOR_TEST_DEVICES")
        with ExitStack() as stack:
            if device == "cuda":
                self.assertTrue(self.torch.cuda.is_available())
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                if device == "cuda":
                    self.assertTrue(self.jt.compiler.has_cuda and self.jt.compiler.is_cuda)
                    self.assertFalse(any(getattr(self.jt.compiler, n, False)
                                         for n in ("has_acl", "has_rocm", "has_corex")))
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(
                    use_cuda=int(device == "cuda"), device_id=0 if device == "cuda" else -1,
                    backend_fallback="error"))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _resident(self, value, device):
        self.assertEqual(value.device.type, device)
        if device == "cuda":
            self.assertEqual(value.device.index, 0)
        if self.jt is not None:
            native = getattr(value, "_var", value)
            native.sync()
            self.assertEqual(self.jt.core.dispatch_context([native]),
                             (device, 0 if device == "cuda" else -1))
            self.assertEqual(native.placement_backend, 1 if device == "cuda" else 0)
            self.assertEqual(native.location(), "device" if device == "cuda" else "cpu")
        elif device == "cuda":
            self.torch.cuda.synchronize(value.device)

    def _equal(self, value, expected, device):
        before = (value.requires_grad, value.is_leaf, value.grad_fn is None,
                  tuple(value.shape), value.dtype, str(value.device))
        self._resident(value, device)
        with self.torch.no_grad():
            clone = value.clone()
            self.assertTrue(clone is not value)
            self.assertFalse(clone.requires_grad)
            self._resident(clone, device)
            host = clone.cpu()
            self._resident(host, "cpu")
            actual = np.array(host.numpy(), copy=True)
        self.assertEqual(before, (value.requires_grad, value.is_leaf, value.grad_fn is None,
                                  tuple(value.shape), value.dtype, str(value.device)))
        self._resident(value, device)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(bool(np.isfinite(actual).all()))
        error = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        self.assertTrue(bool(np.all(error <= 1e-5+1e-5*np.abs(expected))), "full_array_fixed_1e5")

    def _case(self, device, kind, late, assigned, multiple_groups):
        torch = self.torch
        initial = np.arange(6, dtype=np.float32).reshape(2, 3)/8-.5
        gradient = (np.arange(6, dtype=np.float32).reshape(2, 3)-2)/4
        param = torch.nn.Parameter(torch.tensor(initial, device=device))
        unused = torch.nn.Parameter(torch.tensor(initial, device=device))
        params = ([{"params": [unused]}, {"params": [param]}] if multiple_groups
                  else [unused, param])
        def construct():
            return getattr(torch.optim, kind)(params, lr=.125, weight_decay=.01)
        opt = None if late else construct()
        if assigned:
            param.grad = torch.tensor(gradient, device=device)
        else:
            (param*torch.tensor(gradient, device=device)).sum().backward()
        published = param.grad
        self.assertTrue(published is not None)
        self._equal(published, gradient, device)
        if late:
            opt = construct()
        self.assertTrue(param.grad is published)
        self.assertTrue(unused.grad is None)
        expected = initial.astype(np.float64)
        moment = np.zeros_like(expected)
        variance = np.zeros_like(expected)
        step_count = 0
        def update(g):
            nonlocal expected, moment, variance, step_count
            step_count += 1
            g = g.astype(np.float64)
            if kind == "SGD":
                expected -= .125*(g+.01*expected)
            else:
                moment = .9*moment+.1*g
                variance = .999*variance+.001*g*g
                expected = expected*(1-.125*.01)-.125*(moment/(1-.9**step_count))/(np.sqrt(
                    variance/(1-.999**step_count))+1e-8)
            opt.step()
            self._equal(param, expected, device)
            self._equal(unused, initial, device)
            self.assertTrue(unused.grad is None)
            if kind == "AdamW":
                state = opt.state_dict()["state"]
                self.assertEqual(len(state), 1)
                entry = next(iter(state.values()))
                self._equal(entry["exp_avg"], moment, device)
                self._equal(entry["exp_avg_sq"], variance, device)
                self.assertEqual(float(entry["step"].item()), float(step_count))
        update(gradient)
        self.assertTrue(param.grad is published)
        self._equal(param.grad, gradient, device)
        update(gradient)
        self.assertTrue(param.grad is published)
        opt.zero_grad(set_to_none=True)
        self.assertTrue(param.grad is None)
        opt.step()
        self._equal(param, expected, device)
        param.grad = torch.tensor(gradient, device=device)
        update(gradient)
        opt.zero_grad(set_to_none=False)
        self.assertTrue(param.grad is not None)
        zeros = np.zeros_like(gradient)
        self._equal(param.grad, zeros, device)
        update(zeros)
        self.assertTrue(unused.grad is None)

    def _run(self, device):
        with self._scope(device):
            for kind in ("SGD", "AdamW"):
                for late in (False, True):
                    for assigned in (False, True):
                        for multiple_groups in (False, True):
                            with self.subTest(kind=kind, late=late, assigned=assigned,
                                              multiple_groups=multiple_groups):
                                self._case(device, kind, late, assigned, multiple_groups)

    def test_cpu(self):
        self._run("cpu")

    def test_cuda(self):
        self._run("cuda")


if __name__ == "__main__":
    unittest.main()
