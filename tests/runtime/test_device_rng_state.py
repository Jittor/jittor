import unittest
import jittor as jt


class TestDeviceRngState(unittest.TestCase):
    def test_cpu_engine_is_not_an_accelerator_state_provider(self):
        before = jt.get_cpu_rng_state()
        with self.assertRaisesRegex(RuntimeError, 'no native device RNG state owner'):
            jt.core.get_device_rng_state('cpu', 0)
        self.assertEqual(jt.get_cpu_rng_state(), before)


if __name__ == '__main__':
    unittest.main()
