"""Focused regressions for diffusers TextToVideoSDPipeline import/runtime gaps."""

from _helpers import capability as _test_capability

import unittest

import jittor as jt
import torch


class TestDiffusersVideoCompat(unittest.TestCase):
    def test_kornia_import_time_torch_api_surface(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
        result = torch.linalg.inv_ex(x)
        self.assertTrue(hasattr(result, "inverse"))
        self.assertTrue(hasattr(result, "info"))
        self.assertEqual(result.info.numpy().tolist(), [0])

        self.assertIs(torch.torch, torch)
        self.assertIsInstance(torch.ones(1), torch.torch.Tensor)
        self.assertEqual(torch.amp.custom_fwd(cast_inputs=torch.float32)(lambda v: v)(3), 3)
        self.assertEqual(torch.cuda.amp.custom_bwd()(lambda v: v)(4), 4)
        self.assertTrue(callable(torch.conv2d))
        self.assertTrue(callable(torch.conv3d))

    @unittest.skipUnless(_test_capability.check_accelerator('cuda', backend=jt).enabled, "requires CUDA")
    def test_layer_norm_fast_path_mixed_affine_dtype(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            prev_use_cuda = jt.introspection.policy.runtime.use_cuda
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
            try:
                x = torch.randn((4, 512)).float32()
                weight = torch.ones((512,), dtype=torch.float16)
                bias = torch.zeros((512,), dtype=torch.float16)
                with torch.no_grad():
                    y = torch.nn.functional.layer_norm(x, (512,), weight, bias, 1e-5)
                    y.sync()
                self.assertEqual(tuple(y.shape), (4, 512))
                self.assertEqual(str(y.dtype), "float32")
            finally:
                _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=prev_use_cuda))


if __name__ == "__main__":
    unittest.main()
