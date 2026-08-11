import unittest

import jittor as jt
import jittor as torch


class TestTorchCompatCudaTF32(unittest.TestCase):
    def test_backend_allow_tf32_controls_cuda_flag(self):
        self.assertTrue(hasattr(jt.flags, "cuda_allow_tf32"))
        old_cuda = int(jt.flags.cuda_allow_tf32)
        old_acl = getattr(jt, "acl_allow_hf32", None)
        try:
            jt.flags.cuda_allow_tf32 = 0
            if hasattr(jt, "acl_allow_hf32"):
                jt.acl_allow_hf32 = False
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

            torch.backends.cuda.matmul.allow_tf32 = True
            self.assertEqual(int(jt.flags.cuda_allow_tf32), 1)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

            torch.set_float32_matmul_precision("highest")
            self.assertEqual(int(jt.flags.cuda_allow_tf32), 0)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

            torch.set_float32_matmul_precision("high")
            self.assertEqual(int(jt.flags.cuda_allow_tf32), 1)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            jt.flags.cuda_allow_tf32 = old_cuda
            if old_acl is not None:
                jt.acl_allow_hf32 = old_acl


if __name__ == "__main__":
    unittest.main()
