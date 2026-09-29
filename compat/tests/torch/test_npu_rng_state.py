"""Complete RNG checkpointing must replay real ACL random work, not just seeds."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import jittor as jt

from _helpers import capability
from _helpers.child_process import run_child_script


@unittest.skipIf(not capability.check_accelerator('acl', backend=jt).enabled, 'No ACL found')
class TestNpuRngState(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(torch, '_torch_compat_install_context'))
        scope = jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback='error')
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.device = torch.npu.current_device()
        self.saved_cpu = torch.get_rng_state()
        self.saved_npu = torch.npu.get_rng_state_all()
        self.addCleanup(torch.npu.set_device, self.device)
        self.addCleanup(torch.set_rng_state, self.saved_cpu)
        self.addCleanup(torch.npu.set_rng_state_all, self.saved_npu)
        self.fallback_start = jt.core.backend_fallback_count()

    def tearDown(self):
        self.assertEqual(jt.flags.backend_fallback, 'error')
        self.assertEqual(jt.core.backend_fallback_count(), self.fallback_start)

    def draw(self, kind):
        device = 'npu:%d' % torch.npu.current_device()
        if kind == 'dropout':
            value = torch.nn.functional.dropout(torch.ones((1024,), device=device), p=0.3, training=True)
        else:
            value = getattr(torch, kind)((1024,), dtype=torch.float32, device=device)
        value.sync()
        jt.sync_all(True)
        self.assertEqual(value.location(), 'device')
        self.assertEqual(value.device_id, torch.npu.current_device())
        self.assertIn(value.placement_backend, (-1, 2))
        result = value.detach().cpu().numpy().copy()
        self.assertTrue(np.isfinite(result).all())
        return result

    def assert_state(self, state):
        self.assertEqual(state.dtype, torch.uint8)
        self.assertEqual(state.ndim, 1)
        self.assertEqual(state.device.type, 'cpu')
        self.assertGreater(state.numel(), 1)
        return state.numpy().copy()

    def test_progressed_state_replays_uniform_normal_and_nonzero_dropout(self):
        for kind in ('rand', 'randn', 'dropout'):
            with self.subTest(kind=kind):
                torch.npu.manual_seed(137)
                initial = self.assert_state(torch.npu.get_rng_state())
                first, second = self.draw(kind), self.draw(kind)
                state = torch.npu.get_rng_state()
                self.assertFalse(np.array_equal(first, second))
                self.assertFalse(np.array_equal(initial, self.assert_state(state)))
                expected = [self.draw(kind), self.draw(kind)]
                self.draw(kind)
                torch.npu.set_rng_state(state)
                for item in expected:
                    np.testing.assert_array_equal(self.draw(kind), item)
                # State object is a detached snapshot; repeated restoration is legal.
                torch.npu.set_rng_state(state)
                np.testing.assert_array_equal(self.draw(kind), expected[0])

    def test_torch_manual_seed_preserves_non_torch_rngs_and_full_seed(self):
        import random
        py_state = random.getstate()
        np_state = np.random.get_state()
        seed = (1 << 63) + 137
        torch.manual_seed(seed)
        self.assertEqual(torch.initial_seed(), seed)
        for device in range(torch.npu.device_count()):
            self.assertEqual(jt.core.rng_initial_seed('acl', device), seed)
        self.assertEqual(random.getstate(), py_state)
        after = np.random.get_state()
        self.assertEqual(after[0], np_state[0])
        np.testing.assert_array_equal(after[1], np_state[1])
        self.assertEqual(after[2:], np_state[2:])
        torch.npu.manual_seed(-1)
        self.assertEqual(torch.npu.initial_seed(), (1 << 64) - 1)
        with self.assertRaises(RuntimeError):
            torch.npu.manual_seed(1 << 64)

    def test_cpu_state_restore_does_not_reseed_acl(self):
        # This is state metadata only; it does not execute a CPU random draw.
        cpu = torch.get_rng_state()
        self.assert_state(cpu)
        self.draw('rand')
        before = self.assert_state(torch.npu.get_rng_state())
        torch.set_rng_state(cpu)
        after = self.assert_state(torch.npu.get_rng_state())
        np.testing.assert_array_equal(before, after)

    def test_invalid_state_and_all_count_leave_stream_unchanged(self):
        torch.npu.manual_seed(137)
        self.draw('rand')
        state = torch.npu.get_rng_state()
        before = self.assert_state(state)
        bad = torch.tensor([0, 1, 2], dtype=torch.uint8, device='cpu')
        with self.assertRaises((RuntimeError, ValueError)):
            torch.npu.set_rng_state(bad)
        np.testing.assert_array_equal(before, self.assert_state(torch.npu.get_rng_state()))
        with self.assertRaises(TypeError):
            torch.npu.set_rng_state(torch.tensor([1.0], dtype=torch.float32, device='cpu'))
        with self.assertRaises((RuntimeError, ValueError)):
            torch.npu.set_rng_state_all([])
        np.testing.assert_array_equal(before, self.assert_state(torch.npu.get_rng_state()))
        with self.assertRaises((RuntimeError, ValueError)):
            torch.npu.get_rng_state(torch.npu.device_count())
        with self.assertRaises(ValueError):
            torch.npu.get_rng_state('cpu')

    def test_all_states_are_real_distinct_device_streams(self):
        count = torch.npu.device_count()
        states = torch.npu.get_rng_state_all()
        self.assertEqual(len(states), count)
        for index, state in enumerate(states):
            np.testing.assert_array_equal(self.assert_state(state),
                                          self.assert_state(torch.npu.get_rng_state(index)))
        if count == 1:
            torch.npu.set_rng_state_all(states)
            return
        # Actual random draws on two real devices, not a synthetic two-rank claim.
        left, right = self.device, (self.device + 1) % count
        torch.npu.set_device(left)
        torch.npu.manual_seed(101)
        self.draw('rand')
        left_before = self.assert_state(torch.npu.get_rng_state(left))
        torch.npu.set_device(right)
        torch.npu.manual_seed(202)
        self.draw('randn')
        np.testing.assert_array_equal(left_before, self.assert_state(torch.npu.get_rng_state(left)))
        complete = torch.npu.get_rng_state_all()
        corrupt = list(complete)
        corrupt[-1] = torch.tensor([0], dtype=torch.uint8, device='cpu')
        with self.assertRaises((RuntimeError, ValueError)):
            torch.npu.set_rng_state_all(corrupt)
        for before, after in zip(complete, torch.npu.get_rng_state_all()):
            np.testing.assert_array_equal(self.assert_state(before), self.assert_state(after))
        torch.npu.set_rng_state_all(states)

    def test_fresh_process_replays_saved_continuation(self):
        for kind in ('rand', 'randn', 'dropout'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix='npu-rng-') as raw:
                root = Path(raw)
                torch.npu.manual_seed(137)
                self.draw(kind)
                state = torch.npu.get_rng_state()
                torch.save(state, root / 'state.pt')
                expected = [self.draw(kind), self.draw(kind)]
                source = '''
import os
from pathlib import Path
import numpy as np
import torch
import jittor as jt
assert hasattr(torch, '_torch_compat_install_context')
assert os.getpid() != PARENT_PID
root = Path(ROOT)
with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback='error'):
    torch.npu.set_device(DEVICE)
    def draw():
        device = 'npu:%d' % DEVICE
        if KIND == 'dropout':
            value = torch.nn.functional.dropout(torch.ones((1024,), device=device), p=0.3, training=True)
        else:
            value = getattr(torch, KIND)((1024,), dtype=torch.float32, device=device)
        value.sync()
        jt.sync_all(True)
        assert value.location() == 'device' and value.device_id == DEVICE
        assert value.placement_backend in (-1, 2)
        result = value.detach().cpu().numpy().copy()
        assert np.isfinite(result).all()
        return result
    torch.npu.manual_seed(84721)
    draw()
    torch.npu.set_rng_state(torch.load(root / 'state.pt', map_location='cpu', weights_only=False))
    np.savez(root / 'actual.npz', first=draw(), second=draw())
    assert jt.core.backend_fallback_count() == 0
print('NPU-RNG-FRESH-PROCESS-PASS', flush=True)
'''
                source = ('PARENT_PID=%r\nROOT=%r\nDEVICE=%r\nKIND=%r\n' %
                          (os.getpid(), str(root), self.device, kind)) + source
                result = run_child_script(source, text=True, timeout=240, name='npu_rng_' + kind)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('NPU-RNG-FRESH-PROCESS-PASS', result.stdout)
                with np.load(root / 'actual.npz', allow_pickle=False) as actual:
                    np.testing.assert_array_equal(actual['first'], expected[0])
                    np.testing.assert_array_equal(actual['second'], expected[1])

    def test_random_namespace_reexports_the_same_owners(self):
        for name in ('get_rng_state', 'set_rng_state', 'get_rng_state_all', 'set_rng_state_all',
                     'manual_seed', 'manual_seed_all', 'initial_seed', 'seed', 'seed_all'):
            self.assertIs(getattr(torch.npu, name), getattr(torch.npu.random, name))
