"""cuDNN FP16/BF16 SDPA reads packed-QKV views without QKV layout copies."""
import contextlib
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.cuda.kernels.nn import cudnn_attention_cuda as kernel


@unittest.skipUnless(capability.check_accelerator("cuda", backend=jt).enabled,
                     "needs a real CUDA runtime")
class TestCudnnPackedQkvViews(unittest.TestCase):
    def setUp(self):
        self.scope = contextlib.ExitStack()
        self.scope.enter_context(jt.flag_scope(use_cuda=1, amp_reg=0))
        self.scope.enter_context(forbid_backend_fallbacks())
        self.addCleanup(self.scope.close)

    def test_packed_qkv_unbind_and_chunk_parity(self):
        if kernel._library() is None:
            self.skipTest("cuDNN frontend library unavailable")
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (False, True):
                with self.subTest(dtype=dtype, causal=causal):
                    torch.manual_seed(0)
                    B, S, H, D = 2, 197, 12, 64
                    qkv = torch.randn(B, S, 3 * H * D, device="cuda", dtype=dtype)
                    with torch.no_grad():
                        # Pattern A: timm reshape + permute + unbind
                        q, k, v = qkv.reshape(B, S, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
                        self.assertIsNotNone(kernel._packed_qkv_view(q, k, v))
                        out_packed_a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                        out_contig = F.scaled_dot_product_attention(
                            q.contiguous(), k.contiguous(), v.contiguous(), is_causal=causal
                        )
                        # Pattern B: chunk + reshape + transpose
                        q2, k2, v2 = qkv.chunk(3, dim=-1)
                        q2 = q2.reshape(B, S, H, D).transpose(1, 2)
                        k2 = k2.reshape(B, S, H, D).transpose(1, 2)
                        v2 = v2.reshape(B, S, H, D).transpose(1, 2)
                        self.assertIsNotNone(kernel._packed_qkv_view(q2, k2, v2))
                        out_packed_b = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
                        jt.sync_all(True)

                    self.assertEqual(out_packed_a.device.type, "cuda")
                    self.assertEqual(out_packed_b.device.type, "cuda")
                    ref = out_contig.float().cpu().numpy()
                    np.testing.assert_allclose(
                        out_packed_a.float().cpu().numpy(), ref, rtol=1e-3, atol=1e-3
                    )
                    np.testing.assert_allclose(
                        out_packed_b.float().cpu().numpy(), ref, rtol=1e-3, atol=1e-3
                    )

    def test_packed_qkv_training_preserves_gradients(self):
        if kernel._library() is None:
            self.skipTest("cuDNN frontend library unavailable")
        torch.manual_seed(1)
        B, S, H, D = 1, 32, 4, 64
        qkv = torch.randn(B, S, 3 * H * D, device="cuda", dtype=torch.float16, requires_grad=True)
        q, k, v = qkv.reshape(B, S, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        jt.sync_all(True)
        self.assertIsNotNone(qkv.grad)
        self.assertEqual(qkv.grad.device.type, "cuda")
        self.assertTrue(bool(torch.isfinite(qkv.grad).all().item()))

    def test_masked_packed_qkv_inside_step_capture(self):
        if kernel._library() is None:
            self.skipTest("cuDNN frontend library unavailable")
        torch.manual_seed(2)
        B, S, H, D = 4, 49, 3, 32
        qkv = torch.randn(B, S, 3 * H * D, device="cuda", dtype=torch.float16)
        bias = torch.randn(1, H, S, S, device="cuda", dtype=torch.float16) * 0.1

        def run(x, m):
            q, k, v = x.reshape(B, S, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=m)

        with torch.no_grad():
            ref = run(qkv, bias)
            compiled = torch.compile(run, mode="reduce-overhead")
            _ = compiled(qkv, bias)
            out = compiled(qkv, bias)
            jt.sync_all(True)
        self.assertEqual(out.device.type, "cuda")
        np.testing.assert_allclose(
            out.float().cpu().numpy(), ref.float().cpu().numpy(), rtol=2e-3, atol=2e-3
        )


if __name__ == "__main__":
    unittest.main()
