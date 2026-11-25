"""CUDA inference LayerNorm reads permute(0, 2, 3, 1) NCHW views without contiguous copies."""
import contextlib
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.cuda.kernels.nn import layer_norm_cuda as kernel


@unittest.skipUnless(capability.check_accelerator("cuda", backend=jt).enabled,
                     "needs a real CUDA runtime")
class TestLayerNormNchwViewsCuda(unittest.TestCase):
    def setUp(self):
        self.scope = contextlib.ExitStack()
        self.scope.enter_context(jt.flag_scope(use_cuda=1, amp_reg=0))
        self.scope.enter_context(forbid_backend_fallbacks())
        self.addCleanup(self.scope.close)

    def test_nchw_permute_tile_and_block_paths_match_contiguous(self):
        # Covers both spatial >= 256 (32-spatial warp tile) and spatial < 256 (block per row),
        # plus non-multiple-of-32 spatial sizes (49, 196, 784, 3136) and all ConvNeXt channels.
        shapes = [
            (1, 96, 56, 56),   # spatial=3136 >= 256 (tile path)
            (2, 192, 28, 28),  # spatial=784 >= 256 (tile path, batch=2)
            (1, 384, 14, 14),  # spatial=196 < 256 (block-row path)
            (2, 768, 7, 7),    # spatial=49 < 256 (block-row path, batch=2)
            (1, 37, 17, 19),   # non-aligned channels & spatial >= 256
        ]
        for dtype, atol, rtol in (
            (torch.float32, 1e-5, 1e-5),
            (torch.float16, 2e-3, 2e-3),
            (torch.bfloat16, 2e-2, 2e-2),
        ):
            for n, c, h, w in shapes:
                with self.subTest(dtype=dtype, shape=(n, c, h, w)):
                    torch.manual_seed(0)
                    x_nchw = torch.randn(n, c, h, w, device="cuda", dtype=dtype)
                    weight = torch.randn(c, device="cuda", dtype=dtype)
                    bias = torch.randn(c, device="cuda", dtype=dtype)
                    with torch.no_grad():
                        x_nhwc_view = x_nchw.permute(0, 2, 3, 1)
                        self.assertIsNotNone(kernel._nchw_physical(x_nhwc_view))
                        out_fused = F.layer_norm(x_nhwc_view, (c,), weight, bias, 1e-6)
                        out_contig = F.layer_norm(x_nhwc_view.contiguous(), (c,), weight, bias, 1e-6)
                        jt.sync_all(True)
                    self.assertEqual(out_fused.device.type, "cuda")
                    self.assertEqual(tuple(out_fused.shape), (n, h, w, c))
                    np.testing.assert_allclose(
                        out_fused.float().cpu().numpy(),
                        out_contig.float().cpu().numpy(),
                        atol=atol,
                        rtol=rtol,
                    )


if __name__ == "__main__":
    unittest.main()
