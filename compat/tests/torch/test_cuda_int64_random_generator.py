"""CUDA int64 explicit-Generator integer sampling and exact stream replay.

Independent Native/Jittor runs use
the same public methods, never cross-runtime RNG-state or sample equality.
The proposed implementation supports contiguous CUDA int64 with an explicit
Generator only. CPU/default/strided support and TorchData resume remain open.
CPU ByteTensor state and independent host snapshots are transport, not CPU RNG
or CPU-framework numerical tests. No checkpoint/model data is read or written.
"""
from contextlib import ExitStack
import os
import sys
import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.cuda


class TestCudaInt64RandomGenerator(unittest.TestCase):
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
        cls.torch = torch

    def setUp(self):
        if not self.torch.cuda.is_available():
            self.skipTest('CUDA unavailable; not CUDA evidence')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        if self.jt is not None:
            self.assertTrue(self.jt.has_cuda)
            self.assertTrue(self.jt.compiler.is_cuda)
            self.assertFalse(self.jt.compiler.has_acl)
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.stack.enter_context(self.jt.runtime.scope(
                use_cuda=1, device_id=0, backend_fallback='error'))
            self.stack.enter_context(forbid_backend_fallbacks())
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)
        # Empty tensors cannot prove an allocation's physical residence.
        self.anchor = self.torch.ones((1,), dtype=self.torch.int64, device='cuda:0')
        self._materialize(self.anchor)
        self.addCleanup(self._finish_cuda)

    def _finish_cuda(self):
        self._materialize(self.anchor)
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _materialize(self, value, target='cuda'):
        self.assertEqual(value.device.type, target)
        if target == 'cuda':
            self.assertEqual(value.device.index, 0)
        if self.jt is None:
            if target == 'cuda':
                self.torch.cuda.synchronize(value.device)
        else:
            native = getattr(value, '_var', value)
            native.sync()
            index = 0 if target == 'cuda' else -1
            self.assertEqual(native.location(), 'device' if target == 'cuda' else 'cpu')
            self.assertEqual(int(native.device_id), index)
            self.assertEqual(int(native.placement_backend), int(target == 'cuda'))
            self.assertEqual(self.jt.core.dispatch_context([native]), (target, index))

    def _descriptor(self, value):
        # Crucially no sync: transport must not migrate then silently restore it.
        row = (value.device.type, value.device.index, str(value.dtype), tuple(value.shape),
               tuple(value.stride()), value.storage_offset(), bool(value.requires_grad),
               bool(value.is_leaf), value.grad_fn is None)
        if self.jt is not None:
            native = getattr(value, '_var', value)
            row += (native.location(), int(native.device_id), int(native.placement_backend),
                    self.jt.core.dispatch_context([native]))
        return row

    def test_cuda_native_write_version_and_alias_contract(self):
        t = self.torch
        value = t.empty((4,), dtype=t.int64, device='cuda:0')
        # No getter/alias first: real writes must already be counted, not start
        # counting only because somebody has read the new property.
        self.assertIs(value.fill_(3), value)
        self.assertIs(value.add_(2), value)
        self.assertIs(type(value._version), int)
        self.assertEqual(value._version, 2)
        self.assertEqual(value._version, 2)  # Reading must not advance the counter.
        window, detached, data, independent = value[1:3], value.detach(), value.data, value.clone()
        self.assertEqual(tuple(v._version for v in (window, detached, data, independent)), (2, 2, 0, 0))
        for tensor in (value, window, detached, data, independent):
            self._materialize(tensor)
        self.assertIs(window.add_(7), window)
        self.assertEqual(tuple(v._version for v in (value, window, detached, data, independent)), (3, 3, 3, 0, 0))
        for tensor, expected in ((value, [5, 12, 12, 5]), (window, [12, 12]),
                                 (detached, [5, 12, 12, 5]), (data, [5, 12, 12, 5]),
                                 (independent, [5, 5, 5, 5])):
            self.assertTrue(bool(np.array_equal(self._take(tensor), np.array(expected, dtype=np.int64))),
                            'complete CUDA alias values')
        self.assertIs(data.fill_(11), data)
        self.assertEqual(tuple(v._version for v in (value, window, detached, data, independent)), (3, 3, 3, 1, 0))
        for tensor, expected in ((value, [11, 11, 11, 11]), (window, [11, 11]),
                                 (detached, [11, 11, 11, 11]), (data, [11, 11, 11, 11]),
                                 (independent, [5, 5, 5, 5])):
            self.assertTrue(bool(np.array_equal(self._take(tensor), np.array(expected, dtype=np.int64))),
                            'complete CUDA data-alias values')
        # An explicit logical self-copy counts even though no device bytes differ.
        self.assertIs(value.copy_(value), value)
        self.assertEqual(tuple(v._version for v in (value, window, detached, data, independent)), (4, 4, 4, 1, 0))
        with self.assertRaises(AttributeError):
            value._version = 99
        self.assertEqual(value._version, 4)
        self._materialize(value)
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def test_cuda_int64_random_generator_contract(self):
        import numpy as np
        torch = self.torch
        seed, maximum = 123456789, (1 << 63) - 1
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

        def take(value, shape):
            self.assertEqual(value.dtype, torch.int64)
            self.assertEqual(tuple(value.shape), shape)
            self.assertFalse(value.requires_grad)
            self._materialize(value, 'cuda')
            before = self._descriptor(value)
            transport = value.detach().clone()
            self.assertIsNot(transport, value)
            self._materialize(transport, 'cuda')
            self.assertNotEqual(transport.data_ptr(), value.data_ptr())
            actual = np.array(transport.cpu().numpy(), copy=True)
            # No restorative sync of the original after the host read.
            self.assertEqual(self._descriptor(value), before)
            self.assertEqual(actual.shape, shape)
            self.assertEqual(actual.dtype, np.dtype('int64'))
            return actual

        def draw(generator, shape=(64,), bounds=()):
            value = torch.empty(shape, dtype=torch.int64, device='cuda')
            self.assertIs(value.random_(*bounds, generator=generator), value)
            return take(value, shape)

        def state(generator):
            # CPU ByteTensor is the public CUDA-generator state format, not a
            # CPU RNG or tensor-compute test. Never compare bytes across runtimes.
            saved = generator.get_state().clone()
            self.assertEqual(saved.device.type, 'cpu')
            self.assertEqual(saved.dtype, torch.uint8)
            self.assertEqual(len(saved.shape), 1)
            self.assertGreater(saved.numel(), 0)
            self.assertLessEqual(saved.numel(), 65536)
            payload = np.array(saved.numpy(), copy=True)
            self.assertEqual(payload.dtype, np.dtype('uint8'))
            return saved, payload.tobytes()

        g = torch.Generator(device='cuda').manual_seed(seed)
        initial, initial_bytes = state(g)
        # Preserve the actual downstream spelling, including scalar .item().
        scalar = torch.empty((), dtype=torch.int64, device='cuda')
        self.assertIs(scalar.random_(generator=g), scalar)
        scalar_array = take(scalar, ())
        scalar_before = self._descriptor(scalar)
        scalar_item = scalar.item()
        self.assertIs(type(scalar_item), int)
        self.assertEqual(scalar_item, int(scalar_array.item()))
        self.assertEqual(self._descriptor(scalar), scalar_before)
        after_scalar, after_scalar_bytes = state(g)
        self.assertNotEqual(initial_bytes, after_scalar_bytes)
        first = draw(g)
        after_first, after_first_bytes = state(g)
        second = draw(g)
        unused, after_second_bytes = state(g)
        self.assertNotEqual(after_scalar_bytes, after_first_bytes)
        self.assertNotEqual(after_first_bytes, after_second_bytes)

        # Same state -> same COMPLETE draws and subsequent state, not just seed.
        g.set_state(after_scalar)
        replay_first, replay_second = draw(g), draw(g)
        self.assertTrue(np.array_equal(replay_first, first))
        self.assertTrue(np.array_equal(replay_second, second))
        self.assertEqual(state(g)[1], after_second_bytes)
        g.set_state(after_first)
        self.assertTrue(np.array_equal(draw(g), second))
        self.assertEqual(state(g)[1], after_second_bytes)

        fresh = torch.Generator(device='cuda').manual_seed(seed)
        self.assertEqual(state(fresh)[1], initial_bytes)
        fresh_scalar = draw(fresh, ())
        self.assertTrue(np.array_equal(fresh_scalar, scalar_array))
        self.assertEqual(state(fresh)[1], after_scalar_bytes)
        self.assertTrue(np.array_equal(draw(fresh), first))
        self.assertTrue(np.array_equal(draw(fresh), second))
        self.assertEqual(state(fresh)[1], after_second_bytes)

        # Integer-only host checks: do not round to float before range/low-bit tests.
        default_values = [scalar_item] + [int(v) for v in first] + [int(v) for v in second]
        self.assertTrue(all(0 <= v <= maximum for v in default_values))
        self.assertTrue(any(v >= (1 << 62) for v in default_values))
        self.assertTrue(any(v > (1 << 53) and (v & 1) for v in default_values))
        # Near-maximum adjacent singleton intervals must retain every integer bit.
        # No explicit to=2**63: that value is outside signed-int64 API arguments.
        for low in (maximum - 1, maximum - 2):
            exact = draw(g, (1,), (low, low + 1))
            self.assertEqual(int(exact[0]), low)
        low, high = maximum - 256, maximum
        bounded = [int(v) for v in draw(g, (64,), (low, high))]
        self.assertTrue(all(low <= v < high for v in bounded))
        self.assertTrue(any(v & 1 for v in bounded))
        self.assertTrue(any(not (v & 1) for v in bounded))
        # Finite precision/stream regression, NOT a proof of uniform full support.
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _take(self, value):
        self._materialize(value)
        before = self._descriptor(value)
        with self.torch.no_grad():
            transport = value.detach().clone()
            self.assertIsNot(transport, value)
            self._materialize(transport)
            if value.numel():
                self.assertNotEqual(transport.data_ptr(), value.data_ptr())
            actual = np.array(transport.cpu().numpy(), copy=True)
        self.assertEqual(self._descriptor(value), before)
        self.assertEqual(actual.shape, tuple(value.shape))
        self.assertEqual(actual.dtype, np.dtype(str(value.dtype).removeprefix('torch.')))
        self.assertTrue(bool(np.isfinite(actual).all()))
        return actual

    def _state(self, generator):
        saved = generator.get_state().clone()
        self.assertEqual(saved.device.type, 'cpu')
        self.assertEqual(saved.dtype, self.torch.uint8)
        self.assertEqual(len(saved.shape), 1)
        self.assertGreater(saved.numel(), 0)
        self.assertLessEqual(saved.numel(), 65536)
        payload = np.array(saved.numpy(), copy=True)
        self.assertEqual(payload.dtype, np.dtype('uint8'))
        return saved, payload.tobytes()

    def _draw(self, generator, bounds, shape=(65,)):
        value = self.torch.empty(shape, dtype=self.torch.int64, device='cuda:0')
        self.assertIs(value.random_(*bounds, generator=generator), value)
        actual = self._take(value)
        self.assertFalse(value.requires_grad)
        self.assertTrue(value.is_leaf)
        self.assertIsNone(value.grad_fn)
        return actual

    def test_cuda_signed_full64_and_width_boundaries(self):
        minimum, maximum = -(1 << 63), (1 << 63) - 1
        bounds = [(minimum, minimum + 1), (-9, -1), (-17, 19),
                  (minimum, 0), (maximum, None), (minimum, None)]
        bounds += [(-width, 0) for width in ((1 << 28) - 1, 1 << 28, (1 << 28) + 1)]
        generator = self.torch.Generator(device='cuda').manual_seed(918273645)
        for low, high in bounds:
            with self.subTest(low=low, high=high):
                before, before_bytes = self._state(generator)
                first = self._draw(generator, (low, high))
                after_bytes = self._state(generator)[1]
                self.assertNotEqual(before_bytes, after_bytes)
                # Python integers preserve all64 bits; no float range surrogate.
                end = maximum + 1 if high is None else high
                self.assertTrue(all(low <= int(x) < end for x in first))
                if end - low == 1:
                    self.assertTrue(all(int(x) == low for x in first))
                generator.set_state(before)
                repeated = self._draw(generator, (low, high))
                self.assertTrue(bool(np.array_equal(first, repeated)))
                self.assertEqual(self._state(generator)[1], after_bytes)

    def test_cuda_state_unchanged_empty_and_invalid_bounds(self):
        t = self.torch
        generator = t.Generator(device='cuda').manual_seed(71031)
        for shape in ((0,), (2, 0, 3)):
            with self.subTest(empty_shape=shape):
                before = self._state(generator)[1]
                value = t.empty(shape, dtype=t.int64, device='cuda:0')
                self.assertIs(value.random_(-11, 11, generator=generator), value)
                self.assertEqual(tuple(value.shape), shape)
                self.assertEqual(value.numel(), 0)
                self.assertEqual(value.dtype, t.int64)
                self.assertEqual((value.device.type, value.device.index), ('cuda', 0))
                self.assertFalse(value.requires_grad)
                self.assertEqual(self._state(generator)[1], before)
                # Explicit nonempty anchor; no assertion that empty has bytes.
                self._materialize(self.anchor)
        value = t.full((7,), 23, dtype=t.int64, device='cuda:0')
        original = self._take(value)
        descriptor, version = self._descriptor(value), value._version
        for bounds in ((4, 4), (5, -3), (0,), (-(1 << 63) - 1, 0), (0, 1 << 63)):
            with self.subTest(invalid_bounds=bounds):
                before = self._state(generator)[1]
                # Both APIs reject; their public exception class may differ at
                # Python binding overflow. Never catch an error from valid draws.
                with self.assertRaises((RuntimeError, OverflowError, TypeError)):
                    value.random_(*bounds, generator=generator)
                self.assertEqual(self._state(generator)[1], before)
                self.assertEqual(self._descriptor(value), descriptor)
                self.assertEqual(value._version, version)
                self.assertTrue(bool(np.array_equal(self._take(value), original)))

    def test_cuda_window_alias_in_place(self):
        t = self.torch
        generator = t.Generator(device='cuda').manual_seed(62117)
        base = t.full((10,), -123, dtype=t.int64, device='cuda:0')
        window, held = base[2:8], base[2:8]
        self.assertIsNot(window, held)
        for value in (base, window, held):
            self._materialize(value)
        self.assertTrue(window.is_contiguous())
        self.assertEqual(window.storage_offset(), 2)
        before = tuple(self._descriptor(v) for v in (base, window, held))
        pointers = tuple(v.data_ptr() for v in (base, window, held))
        versions = tuple(v._version for v in (base, window, held))
        self.assertIs(window.random_(-7, 8, generator=generator), window)
        full, selected, alias = (self._take(v) for v in (base, window, held))
        self.assertTrue(bool(np.array_equal(selected, alias)))
        self.assertTrue(bool(np.array_equal(full[2:8], selected)))
        self.assertTrue(bool(np.array_equal(full[:2], np.full(2, -123, dtype=np.int64))))
        self.assertTrue(bool(np.array_equal(full[8:], np.full(2, -123, dtype=np.int64))))
        self.assertTrue(all(-7 <= int(x) < 8 for x in selected))
        self.assertEqual(tuple(self._descriptor(v) for v in (base, window, held)), before)
        self.assertEqual(tuple(v.data_ptr() for v in (base, window, held)), pointers)
        self.assertEqual(tuple(v._version for v in (base, window, held)), tuple(v + 1 for v in versions))

    def test_cuda_mixed_integer_and_old_float_replay(self):
        t = self.torch
        generator = t.Generator(device='cuda').manual_seed(5271893)
        other = t.Generator(device='cuda').manual_seed(6113)
        other_state = self._state(other)[1]
        def sequence():
            first = t.rand((7,), dtype=t.float32, device='cuda:0', generator=generator)
            integers = t.empty((9,), dtype=t.int64, device='cuda:0')
            self.assertIs(integers.random_(-31, 23, generator=generator), integers)
            normal = t.randn((5,), dtype=t.float64, device='cuda:0', generator=generator)
            last = t.empty((), dtype=t.int64, device='cuda:0')
            self.assertIs(last.random_(generator=generator), last)
            return tuple(self._take(v) for v in (first, integers, normal, last))
        saved, saved_bytes = self._state(generator)
        first = sequence()
        after = self._state(generator)[1]
        self.assertNotEqual(after, saved_bytes)
        self.assertEqual(self._state(other)[1], other_state)
        generator.set_state(saved)
        second = sequence()
        self.assertEqual(self._state(generator)[1], after)
        for left, right in zip(first, second):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertTrue(bool(np.array_equal(left, right)))


if __name__ == '__main__':
    unittest.main()
