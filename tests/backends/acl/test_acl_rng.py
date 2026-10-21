"""Native ACL RNG state is per device and independent of the CPU engine."""
import unittest
import numpy as np
import jittor as jt


@unittest.skipUnless(jt.compiler.has_acl, 'requires the ACL provider')
class TestAclRng(unittest.TestCase):
    def setUp(self):
        self.device = jt.current_device()
        self.original_cpu = jt.get_cpu_rng_state()
        self.original_acl = {index: jt.core.get_device_rng_state('acl', index)
                             for index in range(jt.get_device_count())}

    def tearDown(self):
        for index, state in self.original_acl.items():
            jt.core.set_device_rng_state('acl', index, state)
        jt.set_cpu_rng_state(self.original_cpu)

    def _draw_on(self, device):
        value = jt.rand((37,), dtype='float32')
        value.sync()
        self.assertEqual(value.location(), 'device')
        self.assertEqual(value.device_id, device)
        return np.array(value.numpy(), copy=True)

    def _draw(self):
        return self._draw_on(self.device)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_uint64_seed_snapshot_replay_and_cpu_independence(self):
        seed = (1 << 63) + 1729
        jt.core.set_device_seed('acl', self.device, seed)
        self.assertEqual(jt.core.get_device_initial_seed('acl', self.device), seed)
        initial = jt.core.get_device_rng_state('acl', self.device)
        cpu = jt.get_cpu_rng_state()
        first = self._draw()
        advanced = jt.core.get_device_rng_state('acl', self.device)
        self.assertNotEqual(initial, advanced)
        self.assertEqual(jt.get_cpu_rng_state(), cpu)
        jt.set_cpu_seed(19)
        self.assertEqual(jt.core.get_device_rng_state('acl', self.device), advanced)
        jt.set_cpu_rng_state(cpu)
        self.assertEqual(jt.core.get_device_rng_state('acl', self.device), advanced)
        jt.core.set_device_rng_state('acl', self.device, initial)
        np.testing.assert_array_equal(self._draw(), first)
        self.assertEqual(jt.core.get_device_rng_state('acl', self.device), advanced)

    @jt.flag_scope(use_acl=1, use_cuda=1, lazy_execution=1, auto_flush_ops=0)
    def test_snapshot_flushes_pending_uniform_and_normal_graphs(self):
        jt.core.set_device_seed('acl', self.device, 101)
        before = jt.core.get_device_rng_state('acl', self.device)
        uniform = jt.rand((31,), dtype='float32')
        normal = jt.randn((19,), dtype='float32')
        after = jt.core.get_device_rng_state('acl', self.device)
        self.assertNotEqual(before, after)
        self.assertEqual(uniform.location(), 'device')
        self.assertEqual(normal.location(), 'device')
        expected_uniform = np.array(uniform.numpy(), copy=True)
        expected_normal = np.array(normal.numpy(), copy=True)
        jt.core.set_device_rng_state('acl', self.device, before)
        replay_uniform = jt.rand((31,), dtype='float32')
        replay_normal = jt.randn((19,), dtype='float32')
        # Submit the same pair of lazy roots, preserving execution ordering.
        self.assertEqual(jt.core.get_device_rng_state('acl', self.device), after)
        self.assertEqual(replay_uniform.location(), 'device')
        self.assertEqual(replay_normal.location(), 'device')
        np.testing.assert_array_equal(replay_uniform.numpy(), expected_uniform)
        np.testing.assert_array_equal(replay_normal.numpy(), expected_normal)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_invalid_snapshot_is_rejected_without_mutation(self):
        jt.core.set_device_seed('acl', self.device, 23)
        original = jt.core.get_device_rng_state('acl', self.device)
        for invalid in ('', 'bad', original+'trailing',
                        'JITTOR_ACL_RNG_V1\n18446744073709551616 0\n',
                        'JITTOR_ACL_RNG_V1\n23 -1\n',
                        'JITTOR_ACL_RNG_V1\n23 9223372036854775808\n'):
            with self.subTest(state=invalid):
                with self.assertRaisesRegex(RuntimeError, 'ACL RNG'):
                    jt.core.set_device_rng_state('acl', self.device, invalid)
                self.assertEqual(jt.core.get_device_rng_state('acl', self.device), original)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_legacy_global_seed_resets_acl_without_cpu_state_restore(self):
        jt.set_seed(29)
        first = self._draw()
        jt.set_seed(29)
        np.testing.assert_array_equal(self._draw(), first)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_visible_device_streams_have_independent_state(self):
        if jt.get_device_count() < 2:
            self.skipTest('requires two visible ACL devices for per-device state independence')
        other = 1 if self.device == 0 else 0
        saved = jt.core.get_device_rng_state('acl', other)
        try:
            state = jt.core.get_device_rng_state('acl', self.device)
            jt.core.set_device_seed('acl', other, 73)
            self.assertEqual(jt.core.get_device_rng_state('acl', self.device), state)
            self.assertEqual(jt.core.get_device_initial_seed('acl', other), 73)
            jt.core.set_device_seed('acl', self.device, 79)
            current_state = jt.core.get_device_rng_state('acl', self.device)
            current_first = self._draw()
            previous = jt.current_device()
            try:
                jt.set_device(other)
                other_first = self._draw_on(other)
                other_state = jt.core.get_device_rng_state('acl', other)
            finally:
                jt.set_device(previous)
            jt.core.set_device_rng_state('acl', self.device, current_state)
            np.testing.assert_array_equal(self._draw(), current_first)
            self.assertEqual(jt.core.get_device_rng_state('acl', other), other_state)
            jt.core.set_device_seed('acl', other, 73)
            try:
                jt.set_device(other)
                np.testing.assert_array_equal(self._draw_on(other), other_first)
            finally:
                jt.set_device(previous)
        finally:
            jt.core.set_device_rng_state('acl', other, saved)


if __name__ == '__main__':
    unittest.main()
