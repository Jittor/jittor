"""Public held-grad_fn identity, independently runnable on one device/process."""
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np

_ORACLE = os.environ.get('JITTOR_GRAD_FN_RUNTIME') == 'oracle'
if not _ORACLE:
    import jittor as jt
import torch


class TestGradFnIdentity(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertTrue(self.device in ('cpu', 'cuda'))
        if _ORACLE:
            self.assertFalse(hasattr(torch, '_torch_compat_install_context'))
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.skipTest('accelerator prerequisite: independent PyTorch CUDA unavailable')
        else:
            if self.device == 'cuda' and not jt.compiler.has_cuda:
                self.skipTest('accelerator prerequisite: candidate CUDA compiler unavailable')
            scope = jt.runtime.scope(use_cuda=int(self.device == 'cuda'),
                device_id=0 if self.device == 'cuda' else -1, backend_fallback='error')
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)
            self.assertTrue(jt.core.backend_fallback_count() == 0)
            self.addCleanup(lambda: self.assertTrue(jt.core.backend_fallback_count() == 0))
        scope = torch.enable_grad()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def tensor(self, dtype):
        return torch.tensor([-.5, .25, 1.5], device=self.device,
                            dtype=dtype, requires_grad=True)

    def full(self, value, expected, dtype):
        self.assertTrue(value.device.type == self.device)
        if self.device == 'cuda':
            self.assertTrue(value.device.index == 0)
        if not _ORACLE:
            native = getattr(value, '_var', value)
            native.sync()
            self.assertTrue(jt.core.dispatch_context([native]) ==
                (self.device, 0 if self.device == 'cuda' else -1))
            self.assertTrue(native.location() == ('device' if self.device == 'cuda' else 'cpu'))
        with torch.no_grad():
            actual = np.array(value.detach().cpu().numpy(), copy=True)
        expected = np.asarray(expected, dtype='float32' if dtype == torch.float32 else 'float64')
        self.assertTrue(actual.shape == expected.shape)
        self.assertTrue(actual.dtype == expected.dtype)
        self.assertTrue(bool(np.isfinite(actual).all() and np.array_equal(actual, expected)))

    def test_same_live_producer_reuses_held_wrapper(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype)
                y = x * 2
                held = y.grad_fn
                self.assertTrue(held is not None)
                self.assertTrue(y.grad_fn is held)
                other = x * 3
                self.assertTrue(other.grad_fn is not held)
                self.assertTrue(y.grad_fn is held)
                y.sum().backward(retain_graph=True)
                self.full(x.grad, [2., 2., 2.], dtype)
                self.full(y, [-1., .5, 3.], dtype)
                self.assertTrue(y.grad_fn is held)

    def test_detach_never_revives_cached_producer(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype)
                y = x * 2
                held = y.grad_fn
                detached = y.detach()
                self.assertTrue(detached.grad_fn is None)
                detached.requires_grad_(True)
                self.assertTrue(detached.is_leaf and detached.grad_fn is None)
                self.assertTrue(y.grad_fn is held)
                self.full(detached, [-1., .5, 3.], dtype)

    def test_copy_uses_current_producer(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype)
                y = x * 2
                old = y.grad_fn
                rhs = self.tensor(dtype)
                y.copy_(rhs)
                copied = y.grad_fn
                self.assertTrue(copied is not None and copied is not old)
                self.assertTrue(y.grad_fn is copied)
                y.sum().backward(retain_graph=True)
                self.full(x.grad, [0., 0., 0.], dtype)
                self.full(rhs.grad, [1., 1., 1.], dtype)
                self.full(y, [-.5, .25, 1.5], dtype)

    def test_concurrent_queries_intern_one_cold_wrapper(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                x = self.tensor(dtype)
                y = x * 2
                barrier = Barrier(4)
                def query(_):
                    barrier.wait(timeout=20)
                    return y.grad_fn
                with ThreadPoolExecutor(max_workers=4) as pool:
                    wrappers = list(pool.map(query, range(4)))
                held = wrappers[0]
                self.assertTrue(held is not None and all(item is held for item in wrappers))
                self.assertTrue(y.grad_fn is held)
                self.full(y, [-1., .5, 3.], dtype)


if __name__ == '__main__':
    unittest.main()
