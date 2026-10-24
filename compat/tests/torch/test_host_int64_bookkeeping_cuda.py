"""Host-only loader seed bookkeeping, anchored by real CUDA batch tensors.

Run unchanged in independent Native and Candidate interpreters. RNG byte/sample
equality is checked only within one implementation, never across runtimes.
This is not CPU model training or complete TorchData/PPO resume acceptance.
"""
from contextlib import ExitStack
import os
import sys
import unittest

import pytest

pytestmark = pytest.mark.cuda


class TestHostInt64BookkeepingCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jt = None
        if os.environ.get('JITTOR_TORCH_SHIM') == '1':
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        else:
            import torch
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)
        cls.t = torch

    def setUp(self):
        if not self.t.cuda.is_available():
            self.skipTest('CUDA unavailable; not CUDA acceptance')
        stack = ExitStack()
        self.addCleanup(stack.close)
        if self.jt is not None:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(self.jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
        self.anchor = self.t.tensor([17, 29], dtype=self.t.int64, device='cuda:0')
        self._cuda_anchor()
        self.addCleanup(self._cuda_anchor)
        saved = self.t.default_generator.get_state().clone()
        self.addCleanup(self.t.default_generator.set_state, saved)

    def _cuda_anchor(self):
        self.assertEqual((self.anchor.device.type, self.anchor.device.index), ('cuda', 0))
        if self.jt is not None:
            native = getattr(self.anchor, '_var', self.anchor)
            native.sync()
            self.assertEqual(native.location(), 'device')
            self.assertEqual((int(native.placement_backend), int(native.device_id)), (1, 0))
            self.assertEqual(self.jt.core.dispatch_context([native]), ('cuda', 0))
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
        self.t.cuda.synchronize(0)

    def _state(self, generator):
        saved = generator.get_state().clone()
        self.assertEqual(saved.device.type, 'cpu')
        self.assertEqual(saved.dtype, self.t.uint8)
        self.assertEqual(len(saved.shape), 1)
        self.assertGreater(saved.numel(), 0)
        self.assertLessEqual(saved.numel(), 65536)
        return saved, bytes(saved.tolist())

    def _draw(self, generator=None, bounds=(), shape=()):
        value = self.t.empty(shape, dtype=self.t.int64, device='cpu')
        self.assertIs(value.random_(*bounds, generator=generator), value)
        # Keep exactly the scalar expression used by genuine DataLoader.
        actual = value.item() if shape == () else value.tolist()
        self.assertEqual(value.device.type, 'cpu')
        self.assertIsNone(value.device.index)
        if self.jt is not None and value.numel():
            native = getattr(value, '_var', value)
            native.sync()
            self.assertEqual(native.location(), 'cpu')
            self.assertEqual((int(native.placement_backend), int(native.device_id)), (0, -1))
            self.assertEqual(self.jt.core.dispatch_context([native]), ('cpu', -1))
        self._cuda_anchor()
        return actual

    def test_host_seed_global_default_and_private_restore(self):
        t = self.t
        for kind in ('implicit', 'default', 'private'):
            with self.subTest(kind=kind):
                g = t.Generator(device='cpu') if kind == 'private' else t.default_generator
                g.manual_seed(94193)
                selected = None if kind == 'implicit' else g
                before, before_bytes = self._state(g)
                first = self._draw(selected)
                after, after_bytes = self._state(g)
                second = self._draw(selected)
                end_bytes = self._state(g)[1]
                self.assertIs(type(first), int)
                self.assertTrue(0 <= first < (1 << 63))
                self.assertNotEqual(before_bytes, after_bytes)
                self.assertNotEqual(after_bytes, end_bytes)
                g.set_state(before)
                self.assertEqual(self._draw(selected), first)
                g.set_state(after)
                self.assertEqual(self._draw(selected), second)
                self.assertEqual(self._state(g)[1], end_bytes)
        # A private stream must not consume the default owner's state.
        before = self._state(t.default_generator)[1]
        self._draw(t.Generator(device='cpu').manual_seed(191))
        self.assertEqual(self._state(t.default_generator)[1], before)

    def test_host_integer_ranges_and_empty_restore(self):
        minimum, maximum = -(1 << 63), (1 << 63) - 1
        bounds = [(), (maximum - 1, maximum), (minimum, minimum + 1),
                  (-17, 19), (minimum, 0), (maximum, None), (minimum, None)]
        g = self.t.Generator(device='cpu').manual_seed(71913)
        for limits in bounds:
            with self.subTest(bounds=limits):
                saved, _ = self._state(g)
                first = self._draw(g, limits, (65,))
                end = self._state(g)[1]
                low = limits[0] if limits else 0
                high = maximum + 1 if not limits or limits[1] is None else limits[1]
                self.assertTrue(all(type(v) is int and low <= v < high for v in first))
                if high - low == 1:
                    self.assertEqual(first, [low] * 65)
                g.set_state(saved)
                self.assertEqual(self._draw(g, limits, (65,)), first)
                self.assertEqual(self._state(g)[1], end)
        for shape in ((0,), (2, 0, 3)):
            with self.subTest(shape=shape):
                saved = self._state(g)[1]
                self._draw(g, (-11, 11), shape)
                self.assertEqual(self._state(g)[1], saved)

    def test_invalid_host_bounds_do_not_advance_state(self):
        g = self.t.Generator(device='cpu').manual_seed(7191)
        value = self.t.full((7,), 23, dtype=self.t.int64, device='cpu')
        for bounds in ((4, 4), (5, -3), (0,), (-(1 << 63) - 1, 0), (0, 1 << 63)):
            with self.subTest(bounds=bounds):
                before = self._state(g)[1]
                with self.assertRaises((RuntimeError, OverflowError, TypeError)):
                    value.random_(*bounds, generator=g)
                self.assertEqual(self._state(g)[1], before)
                self.assertEqual(value.tolist(), [23] * 7)

    def test_candidate_unsupported_requests_fail_before_rng(self):
        if self.jt is None:
            self.skipTest('Candidate capability boundary, not a Native limitation')
        t = self.t
        cpu = t.Generator(device='cpu').manual_seed(941)
        cuda = t.Generator(device='cuda:0').manual_seed(812)
        bad = [(t.empty((3,), dtype=t.float32, device='cpu'), cpu),
               (t.empty((2, 3), dtype=t.int64, device='cpu').transpose(0, 1), cpu),
               (t.empty((3,), dtype=t.int64, device='cpu'), cuda),
               (t.empty((3,), dtype=t.int64, device='cuda:0'), None),
               (t.empty((3,), dtype=t.int64, device='cuda:0'), t.default_generator)]
        for value, selected in bad:
            before = (self._state(cpu)[1], self._state(cuda)[1], self._state(t.default_generator)[1])
            with self.assertRaises((NotImplementedError, RuntimeError)):
                value.random_(generator=selected)
            self.assertEqual((self._state(cpu)[1], self._state(cuda)[1],
                              self._state(t.default_generator)[1]), before)
            self._cuda_anchor()


if __name__ == '__main__':
    unittest.main()
