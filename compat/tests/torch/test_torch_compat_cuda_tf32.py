
from _helpers import capability as _test_capability
import unittest

import jittor as jt
import torch


# ``cuda_allow_tf32`` and ``cuda_allow_cudnn_tf32`` are registered by the CUDA
# backend, so a CPU-only build has nothing to control here. Skipping keeps the
# absence of a GPU from reading as a compatibility regression.
@unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "TF32 control needs a CUDA build")
class TestTorchCompatCudaTF32(unittest.TestCase):
    def test_backend_allow_tf32_controls_cuda_flag(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            self.assertTrue(hasattr(jt.introspection.policy.runtime, "cuda_allow_tf32"))
            old_cuda = int(jt.introspection.policy.runtime.cuda_allow_tf32)
            old_acl = getattr(jt, "acl_allow_hf32", None)
            try:
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_tf32=0))
                if hasattr(jt, "acl_allow_hf32"):
                    jt.acl_allow_hf32 = False
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

                torch.backends.cuda.matmul.allow_tf32 = True
                self.assertEqual(int(jt.introspection.policy.runtime.cuda_allow_tf32), 1)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

                torch.set_float32_matmul_precision("highest")
                self.assertEqual(int(jt.introspection.policy.runtime.cuda_allow_tf32), 0)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

                torch.set_float32_matmul_precision("high")
                self.assertEqual(int(jt.introspection.policy.runtime.cuda_allow_tf32), 1)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            finally:
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_tf32=old_cuda))
                if old_acl is not None:
                    jt.acl_allow_hf32 = old_acl

    def test_cudnn_allow_tf32_controls_independent_flag(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            self.assertTrue(hasattr(jt.introspection.policy.runtime, "cuda_allow_cudnn_tf32"))
            old_cudnn = int(jt.introspection.policy.runtime.cuda_allow_cudnn_tf32)
            old_matmul = int(jt.introspection.policy.runtime.cuda_allow_tf32)
            try:
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_cudnn_tf32=0))
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_tf32=0))
                self.assertFalse(torch.backends.cudnn.allow_tf32)

                torch.backends.cudnn.allow_tf32 = True
                self.assertEqual(int(jt.introspection.policy.runtime.cuda_allow_cudnn_tf32), 1)
                self.assertEqual(int(jt.introspection.policy.runtime.cuda_allow_tf32), 0)
                self.assertTrue(torch.backends.cudnn.allow_tf32)

                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_cudnn_tf32=0))
                self.assertFalse(torch.backends.cudnn.allow_tf32)
            finally:
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_cudnn_tf32=old_cudnn))
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_allow_tf32=old_matmul))


if __name__ == "__main__":
    unittest.main()
