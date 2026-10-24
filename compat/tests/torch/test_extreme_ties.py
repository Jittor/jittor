"""Shared equality gradients through independent public Torch CUDA/CPU APIs."""
from contextlib import ExitStack
import os
import sys
import unittest
import numpy as np


class TestTorchExtremeTies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch, cls.jt = torch, None
        if os.environ.get('JITTOR_TORCH_SHIM') == '1':
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            assert namespace_owner(torch) is jt
            cls.jt = jt
        else:
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)

    def _check(self, cuda):
        torch = self.torch
        if cuda and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; no CUDA claim')
        device = 'cuda:0' if cuda else 'cpu'
        base_a = np.array([[-2., 0., 3.], [4., -5., 0.]], np.float32)
        base_b = np.array([[-2., 1., 3.], [-4., -5., 0.]], np.float32)
        cases = ((base_a, base_b), (base_a, base_b[:1]),
                 (base_a[:, :1], base_b[:1]), (np.array(0., np.float32), base_b),
                 (np.array([np.nan, np.inf, -np.inf, -0., 0.], np.float32),
                  np.array([1., np.inf, -np.inf, 0., -0.], np.float32)))
        with ExitStack() as stack:
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(cuda),
                    device_id=0 if cuda else -1, backend_fallback='error', auto_convert_64_to_32=0))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
            for name in ('maximum', 'minimum'):
                for api in ('function', 'method', 'minmax'):
                    for dtype in (torch.float32, torch.float64):
                        for a, b in cases:
                            with self.subTest(name=name, api=api, dtype=dtype, a=a.shape, b=b.shape):
                                x = torch.tensor(a, dtype=dtype, device=device, requires_grad=True)
                                y = torch.tensor(b, dtype=dtype, device=device, requires_grad=True)
                                if api == 'function':
                                    value = getattr(torch, name)(x, y)
                                elif api == 'method':
                                    value = getattr(x, name)(y)
                                else:
                                    value = getattr(torch, 'max' if name == 'maximum' else 'min')(x, y)
                                aa, bb = np.broadcast_arrays(a, b)
                                cot = (np.arange(aa.size).reshape(aa.shape) + 1).astype(np.float64) * .125
                                incoming = torch.tensor(cot, dtype=dtype, device=device)
                                gx, gy = torch.autograd.grad(value, (x, y), grad_outputs=incoming)
                                want_value = np.maximum(aa, bb) if name == 'maximum' else np.minimum(aa, bb)
                                shared = np.where(aa == bb, cot * .5, cot)
                                want_x = np.where(aa < bb if name == 'maximum' else aa > bb, 0, shared)
                                want_y = np.where(aa > bb if name == 'maximum' else aa < bb, 0, shared)
                                for expected, original in ((want_x, a), (want_y, b)):
                                    while expected.ndim > original.ndim:
                                        expected = expected.sum(0)
                                    for axis, size in enumerate(original.shape):
                                        if size == 1 and expected.shape[axis] != 1:
                                            expected = expected.sum(axis, keepdims=True)
                                    if original is a:
                                        want_x = expected
                                    else:
                                        want_y = expected
                                for result, expected in ((value, want_value), (gx, want_x), (gy, want_y)):
                                    self.assertEqual(result.device.type, 'cuda' if cuda else 'cpu')
                                    self.assertEqual(result.dtype, dtype)
                                    if self.jt is not None:
                                        native = getattr(result, '_var', result)
                                        native.sync()
                                        self.assertEqual(self.jt.core.dispatch_context([native]), ('cuda', 0) if cuda else ('cpu', -1))
                                    actual = np.array(result.detach().cpu().numpy(), copy=True)
                                    np.testing.assert_allclose(actual, expected, atol=0, rtol=0, equal_nan=True)
                                self.assertIsNone(x.grad)
                                self.assertIsNone(y.grad)
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), before)

    def test_cpu(self):
        self._check(False)

    def test_cuda(self):
        self._check(True)


if __name__ == '__main__':
    unittest.main()
