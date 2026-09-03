import unittest

import jittor as jt


class TestFusedAdamwInputErrors(unittest.TestCase):
    def test_mismatched_tensor_lists_fail_at_construction(self):
        parameter = jt.ones((1,), dtype="float32")
        step = jt.array(1.0)
        with jt.flag_scope(use_cuda=1):
            with self.assertRaisesRegex(RuntimeError, r"parameters.size\(\)"):
                jt.fused_adamw(
                    [parameter], [], [parameter], [parameter], step,
                    0.001, 0.9, 0.999, 0.0, 1e-8)


if __name__ == "__main__":
    unittest.main()
