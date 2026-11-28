"""CUDA LayerNorm retains FP32 arithmetic through an output cast."""

import jittor as jt
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_affine_layer_norm_fuses_fp32_casts(dtype):
    rng = np.random.default_rng(67)
    shape = (2, 37, 768)
    with torch.no_grad(), forbid_backend_fallbacks():
        x = torch.tensor(rng.normal(size=shape), dtype=dtype, device="cuda")
        w = torch.tensor(rng.normal(size=shape[-1]), dtype=torch.float32, device="cuda")
        b = torch.tensor(rng.normal(size=shape[-1]), dtype=torch.float32, device="cuda")
        jt.sync_all(True)
        wide = x.float()
        wide.sync()
        ref = F.layer_norm(wide, (768,), w, b)
        ref.sync()
        ref = ref.type_as(x)
        ref.sync()

        def forward():
            return F.layer_norm(x.float(), (768,), w, b).type_as(x)

        forward().sync()
        with jt.profile(device=True, memory=False) as profile:
            actual = forward()
            actual.sync()
        assert actual.dtype == dtype and actual.location() == "device"
        kernels = [row["name"] for row in profile.result.kernel_records]
        assert len(kernels) == 1 and "kernel_vec2b" in kernels[0]
        np.testing.assert_allclose(actual.float().cpu().numpy(), ref.float().cpu().numpy(),
                                   rtol=1e-3, atol=1e-3)
