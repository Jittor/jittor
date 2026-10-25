"""Public log1p and Softplus precision, gradient and promotion regression.

Select TestLog1pSoftplusNative.test_cpu or test_cuda in distinct native/candidate
processes and distinct functional caches. JITTOR_SOFTPLUS_RUNTIME=oracle or
candidate; no other backend claim. FP32/FP64 only. Log1p small inputs and
softplus negative tails use zero absolute tolerance; all original arrays stay
RAM-only and assertion messages never disclose raw arrays or bytes.
"""
from contextlib import ExitStack, contextmanager
import math
import os
import sys
import unittest

import numpy as np


def softplus_reference(x, beta, threshold):
    z = beta * x.astype(np.float64)
    nonlinear = (np.maximum(z, 0.) + np.log1p(np.exp(-np.abs(z)))) / beta
    y = np.where(z > threshold, x, nonlinear)
    derivative = np.empty_like(z)
    positive = z >= 0.
    derivative[positive] = 1. / (1. + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    derivative[~positive] = ez / (1. + ez)
    return y, np.where(z > threshold, 1., derivative)


class TestLog1pSoftplusNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get("JITTOR_SOFTPLUS_RUNTIME",
                                 "candidate" if os.environ.get("JITTOR_TORCH_SHIM") == "1" else "oracle")
        if runtime not in ("oracle", "candidate"):
            raise ValueError("JITTOR_SOFTPLUS_RUNTIME=oracle|candidate required")
        import torch
        cls.torch, cls.jt = torch, None
        if runtime == "oracle":
            assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
            assert not any(n == "jittor" or n.startswith("jittor.") for n in sys.modules)
        else:
            import jittor as jt
            from jittor.compat.torch.publication import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            assert torch.Tensor is not jt.Var
            cls.jt = jt

    @contextmanager
    def _scope(self, device):
        with ExitStack() as stack:
            if device == "cuda":
                if not self.torch.cuda.is_available():
                    self.skipTest("CUDA unavailable; no CUDA acceptance")
                self.torch.cuda.set_device(0)
                self.assertEqual(self.torch.cuda.current_device(), 0)
                if self.jt is not None:
                    compiler = self.jt.compiler
                    self.assertTrue(compiler.has_cuda and compiler.is_cuda)
                    self.assertFalse(any(getattr(compiler, k, False) for k in ("has_acl", "has_rocm", "has_corex")))
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == "cuda"), backend_fallback="error"))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _resident(self, value, device):
        self.assertEqual(value.device.type, device)
        self.assertEqual(value.device.index, 0 if device == "cuda" else None)
        if self.jt is None:
            if device == "cuda":
                self.torch.cuda.synchronize(value.device)
            return
        native = getattr(value, "_var", value)
        native.sync()
        ordinal = 0 if device == "cuda" else -1
        self.assertEqual(self.jt.core.dispatch_context([native]), (device, ordinal))
        self.assertEqual(int(native.device_id), ordinal)
        self.assertEqual(int(native.placement_backend), int(device == "cuda"))
        self.assertEqual(native.location(), "device" if device == "cuda" else "cpu")

    def _snapshot(self, value, device):
        self._resident(value, device)
        metadata = (tuple(value.shape), value.dtype, value.requires_grad, value.is_leaf)
        with self.torch.no_grad():
            clone = value.detach().clone()
            self.assertIsNot(clone, value)
            self._resident(clone, device)
            host = clone.cpu()
            self._resident(host, "cpu")
            array = np.array(host.numpy(), copy=True)
        self._resident(value, device)
        self.assertEqual(metadata, (tuple(value.shape), value.dtype, value.requires_grad, value.is_leaf))
        return array

    def _numeric(self, actual, reference, rtol=1e-5, atol=1e-5):
        reference = np.asarray(reference, dtype=np.float64)
        self.assertEqual(actual.shape, reference.shape)
        finite = np.isfinite(actual) & np.isfinite(reference)
        delta = np.abs(actual.astype(np.float64) - reference)
        failed = ~finite | (delta > atol + rtol * np.abs(reference))
        self.assertEqual(int(np.count_nonzero(failed)), 0,
            "full-element numerical mismatch; arrays remain RAM-only")

    def _vjp(self, device, dtype, values, dy, operation, ref_y, ref_grad, forward_rtol=1e-5, forward_atol=1e-5):
        torch = self.torch
        x = torch.tensor(values.copy(), dtype=getattr(torch, dtype), device=device, requires_grad=True)
        cot = torch.tensor(dy.copy(), dtype=getattr(torch, dtype), device=device, requires_grad=False)
        original = self._snapshot(x, device)
        self.assertTrue(original.tobytes() == values.tobytes(), "input factory bytes changed")
        self.assertTrue(x.is_leaf and x.requires_grad and x.grad is None)
        y = operation(x)
        self._resident(y, device)
        grad, = torch.autograd.grad(y, (x,), grad_outputs=cot, create_graph=False)
        self._resident(grad, device)
        self.assertEqual(y.dtype, x.dtype)
        self.assertEqual(grad.dtype, x.dtype)
        self.assertFalse(grad.requires_grad)
        self._numeric(self._snapshot(y, device), ref_y, forward_rtol, forward_atol)
        self._numeric(self._snapshot(grad, device), ref_grad)
        self.assertTrue(self._snapshot(x, device).tobytes() == original.tobytes(), "math changed original input bytes")
        self.assertTrue(x.requires_grad and x.is_leaf and x.grad is None)
        self.assertFalse(cot.requires_grad)
        self.assertTrue(self._snapshot(cot, device).tobytes() == dy.tobytes(), "math changed cotangent bytes")

    def _check(self, device):
        torch = self.torch
        import torch.nn.functional as functional
        with self._scope(device):
            for dtype in ("float32", "float64"):
                npdtype = getattr(np, dtype)
                tight_rtol = 2e-6 if dtype == "float32" else 2e-14
                small = 2. ** (-30 if dtype == "float32" else -60)
                values = np.array([-0.75, -0.25, -small, -0., 0., small, .25, 3.], dtype=dtype)
                cot = np.array([1., -.625, .375, -1.5, 2.25, .125, -3., 1.875], dtype=dtype)
                ref_y = np.log1p(values.astype(np.float64))
                ref_grad = cot.astype(np.float64) / (1. + values.astype(np.float64))
                for entry in (torch.log1p, lambda x: x.log1p(), torch.special.log1p):
                    with self.subTest(dtype=dtype, primitive="log1p", entry=getattr(entry, "__name__", "method")):
                        self._vjp(device, dtype, values, cot, entry, ref_y, ref_grad, tight_rtol, 0.)
                for beta, threshold in ((1., 20.), (2., 3.), (1., 0.)):
                    boundary = npdtype(threshold / beta)
                    x = np.array([-100., -30., -20., -1., 0., 1.,
                        np.nextafter(boundary, npdtype(-np.inf)), boundary,
                        np.nextafter(boundary, npdtype(np.inf)), threshold / beta + 1., 1000.], dtype=dtype)
                    dy = np.array([1., -.625, .375, -1.5, 2.25, .125, -3., 1.875, .25, -.75, .5625], dtype=dtype)
                    ref_y, derivative = softplus_reference(x, beta, threshold)
                    with self.subTest(dtype=dtype, primitive="softplus", beta=beta, threshold=threshold):
                        self._vjp(device, dtype, x, dy, lambda v: functional.softplus(v, beta=beta, threshold=threshold),
                                  ref_y, dy.astype(np.float64) * derivative)
                tails = np.array([-30., -20.], dtype=dtype)
                dy = np.array([.375, -.625], dtype=dtype)
                ref_y, derivative = softplus_reference(tails, 1., 20.)
                with self.subTest(dtype=dtype, primitive="softplus_nonzero_negative_tail"):
                    self._vjp(device, dtype, tails, dy, functional.softplus, ref_y,
                              dy.astype(np.float64) * derivative, tight_rtol, 0.)
                for state in ("stopped", "no_grad"):
                    with self.subTest(dtype=dtype, state=state):
                        x = torch.tensor([0., 20., 1000.], dtype=getattr(torch, dtype), device=device,
                                         requires_grad=state == "no_grad")
                        if state == "no_grad":
                            with torch.no_grad():
                                y = functional.softplus(x)
                        else:
                            y = functional.softplus(x)
                        self._resident(y, device)
                        self.assertFalse(y.requires_grad)
                        self.assertEqual(x.requires_grad, state == "no_grad")
            original_default = torch.get_default_dtype()
            try:
                for default in (torch.float32, torch.float64):
                    torch.set_default_dtype(default)
                    for dtype in (torch.bool, torch.int8, torch.int32, torch.int64):
                        for entry in (torch.log1p, lambda x: x.log1p(), torch.special.log1p):
                            with self.subTest(default=str(default), input_dtype=str(dtype), entry=getattr(entry, "__name__", "method")):
                                x = torch.tensor([0, 1], dtype=dtype, device=device, requires_grad=False)
                                original = self._snapshot(x, device)
                                y = entry(x)
                                self._resident(y, device)
                                self.assertEqual(y.dtype, default)
                                self.assertFalse(y.requires_grad)
                                self._numeric(self._snapshot(y, device), [0., math.log(2.)])
                                self.assertTrue(self._snapshot(x, device).tobytes() == original.tobytes(),
                                                "integer promotion mutated original")
            finally:
                torch.set_default_dtype(original_default)

    def test_cpu(self):
        self._check("cpu")

    def test_cuda(self):
        self._check("cuda")


if __name__ == "__main__":
    unittest.main()
