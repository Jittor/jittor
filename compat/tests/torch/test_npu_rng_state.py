"""Real NPU default-stream continuation, separately runnable under either Torch.

Run TestNpuRngState for one selected NPU, and TestNpuRngTwoDevices separately
with at least two visible devices. No cross-framework RNG-byte/value parity is
asserted: each runtime must exactly resume its own stream after advancement.
"""
import contextlib
import random
import unittest

import numpy as np
import torch

CANDIDATE = hasattr(torch, "_torch_compat_install_context")
if CANDIDATE:
    import jittor as jt
else:
    import torch_npu  # noqa: F401: official backend registration


class NpuRngCase(unittest.TestCase):
    def setUp(self):
        if CANDIDATE and not jt.compiler.has_acl:
            self.skipTest("ACL backend not built")
        self.assertTrue(torch.npu.is_available(), "built NPU backend must be usable")
        self.stack = contextlib.ExitStack()
        if CANDIDATE:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.stack.enter_context(jt.runtime.scope(use_cuda=0))
            self.stack.enter_context(forbid_backend_fallbacks())
        self.original_device = torch.npu.current_device()
        self.original_global_seed = jt.get_seed() if CANDIDATE else None
        self.py_state, self.np_state = random.getstate(), np.random.get_state()
        self.cpu_state = torch.get_rng_state()
        self.npu_states = torch.npu.get_rng_state_all()

    def tearDown(self):
        if CANDIDATE:
            jt.set_seed(self.original_global_seed)
        torch.set_rng_state(self.cpu_state)
        torch.npu.set_rng_state_all(self.npu_states)
        torch.npu.set_device(self.original_device)
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        self.stack.close()

    def draw(self, device=None, dtype=None):
        device = self.original_device if device is None else device
        dtype = torch.float32 if dtype is None else dtype
        arrays = []
        for normal, count in ((False, 7), (True, 9), (False, 3), (True, 5)):
            factory = torch.randn if normal else torch.rand
            tensor = factory(count, dtype=dtype, device="npu:%d" % device)
            self.assertEqual(tensor.device.type, "npu")
            self.assertEqual(tensor.device.index, device)
            if CANDIDATE:
                native = getattr(tensor, "_var", tensor)
                native.sync()
                self.assertEqual(native.location(), "device")
                self.assertEqual(native.device_id, device)
            arrays.append(tensor.to(dtype=torch.float32).cpu().numpy().copy())
        return arrays

    def assertSuffix(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)

    def state_bytes(self, state):
        self.assertEqual(state.dtype, torch.uint8)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.device.type, "cpu")
        result = state.numpy().tobytes()
        if CANDIDATE:
            self.assertEqual(getattr(state, "_var", state).location(), "cpu")
        return result


class TestNpuRngState(NpuRngCase):
    def test_advanced_suffix_all_supported_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                torch.npu.manual_seed(1701)
                fresh = self.state_bytes(torch.npu.get_rng_state())
                self.draw(dtype=dtype)
                advanced = torch.npu.get_rng_state()
                self.assertNotEqual(self.state_bytes(advanced), fresh)
                expected = self.draw(dtype=dtype)
                self.draw(dtype=dtype)
                torch.npu.set_rng_state(advanced)
                self.assertSuffix(self.draw(dtype=dtype), expected)
                self.assertEqual(torch.npu.current_device(), self.original_device)
                if CANDIDATE:
                    self.assertEqual(jt.runtime.use_cuda, 0)

    def test_cpu_restore_does_not_change_actual_npu_suffix(self):
        self.draw()
        advanced_npu = torch.npu.get_rng_state()
        expected = self.draw()
        torch.npu.set_rng_state(advanced_npu)
        cpu = torch.get_rng_state()
        torch.randn(19, device="cpu").numpy()
        torch.set_rng_state(cpu)
        self.assertSuffix(self.draw(), expected)

    def test_npu_restore_does_not_change_actual_cpu_suffix_or_python_numpy(self):
        torch.randn(17, device="cpu").numpy()
        advanced_cpu = torch.get_rng_state()
        expected = torch.randn(13, device="cpu").numpy().copy()
        torch.set_rng_state(advanced_cpu)
        self.draw()
        advanced_npu = torch.npu.get_rng_state()
        self.draw()
        py_before, np_before = random.getstate(), np.random.get_state()
        torch.npu.set_rng_state(advanced_npu)
        self.assertEqual(random.getstate(), py_before)
        np_after = np.random.get_state()
        self.assertEqual(np_after[0], np_before[0])
        np.testing.assert_array_equal(np_after[1], np_before[1])
        self.assertEqual(np_after[2:], np_before[2:])
        np.testing.assert_array_equal(torch.randn(13, device="cpu").numpy(), expected)

    def test_invalid_all_state_rejected(self):
        self.draw()
        before = torch.npu.get_rng_state_all()
        broken = list(before)
        broken[-1] = torch.tensor(list(b"foreign RNG state"), dtype=torch.uint8, device="cpu")
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            torch.npu.set_rng_state_all(broken)
        self.assertEqual([self.state_bytes(s) for s in torch.npu.get_rng_state_all()],
                         [self.state_bytes(s) for s in before])

    def test_large_seed_no_truncation_and_roundtrip(self):
        for seed in ((1 << 40) + 123, (1 << 63) - 1):
            with self.subTest(seed=seed):
                torch.npu.manual_seed(seed)
                self.assertEqual(torch.npu.initial_seed(), seed)
                self.draw()
                state = torch.npu.get_rng_state()
                expected = self.draw()
                torch.npu.set_rng_state(state)
                self.assertSuffix(self.draw(), expected)

    @unittest.skipUnless(CANDIDATE, "Jittor ACL bounded seed/state format")
    def test_negative_legacy_native_seed_and_public_rejection(self):
        jt.set_seed(-3)
        self.draw()
        state = torch.npu.get_rng_state()
        self.assertEqual(torch.npu.initial_seed(), -3)
        expected = self.draw()
        torch.npu.set_rng_state(state)
        self.assertSuffix(self.draw(), expected)
        before = self.state_bytes(torch.npu.get_rng_state())
        for seed in (-1, 1 << 63, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises((TypeError, ValueError, OverflowError)):
                torch.npu.manual_seed(seed)
            self.assertEqual(self.state_bytes(torch.npu.get_rng_state()), before)


class TestNpuRngTwoDevices(NpuRngCase):
    @unittest.skipUnless(CANDIDATE, "candidate all-state restore atomicity contract")
    def test_later_bad_blob_cannot_partially_restore_earlier_different_state(self):
        self.assertGreaterEqual(torch.npu.device_count(), 2, "requires two visible NPUs")
        older0 = torch.npu.get_rng_state(0)
        self.draw(0)
        self.draw(1)
        before = torch.npu.get_rng_state_all()
        self.assertNotEqual(self.state_bytes(older0), self.state_bytes(before[0]))
        broken = list(before)
        broken[0] = older0  # valid but observably different: exposes partial commit
        broken[1] = torch.tensor(list(b"foreign RNG state"), dtype=torch.uint8, device="cpu")
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            torch.npu.set_rng_state_all(broken)
        self.assertEqual([self.state_bytes(s) for s in torch.npu.get_rng_state_all()],
                         [self.state_bytes(s) for s in before])

    def test_interleaved_device_streams_and_all_restore(self):
        self.assertGreaterEqual(torch.npu.device_count(), 2, "requires two visible NPUs")
        torch.npu.manual_seed_all(917)
        self.draw(0)
        self.draw(1)
        state0 = torch.npu.get_rng_state(0)
        state1 = torch.npu.get_rng_state(1)
        expected0 = self.draw(0)
        self.draw(1)
        advanced1 = self.state_bytes(torch.npu.get_rng_state(1))
        torch.npu.set_rng_state(state0, 0)
        self.assertEqual(self.state_bytes(torch.npu.get_rng_state(1)), advanced1)
        self.assertSuffix(self.draw(0), expected0)
        torch.npu.set_rng_state(state1, 1)
        all_states = torch.npu.get_rng_state_all()
        expected = [self.draw(1), self.draw(0), self.draw(1)]
        torch.npu.set_rng_state_all(all_states)
        self.assertSuffix(self.draw(1), expected[0])
        self.assertSuffix(self.draw(0), expected[1])
        self.assertSuffix(self.draw(1), expected[2])
        self.assertEqual(torch.npu.current_device(), self.original_device)


if __name__ == "__main__":
    unittest.main()
