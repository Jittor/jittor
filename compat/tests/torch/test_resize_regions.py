import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator
from jittor.compat.torch.vision import resize_regions


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_resize_regions_clips_without_changing_interpolation(device):
    if device == "cuda":
        require_accelerator("cuda")
    rng = np.random.default_rng(32)
    data = rng.normal(size=(4, 2, 3, 5)).astype(np.float32)
    rois = [[-2, 1, 3, 7], [4, -1, 7, 3], [20, 20, 22, 22], [1, 1, 1, 5]]
    height, width = 5, 6
    with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
        source = torch.tensor(data, device=device)
        boxes = torch.tensor(rois, device=device)
        before = jt.core._host_readback_count()
        out = resize_regions(source, boxes, (height, width))
        out.sync()
        assert jt.core._host_readback_count() == before
        if device == "cuda":
            assert out.location() == "device"
        expected = np.zeros((4, 2, height, width), dtype=np.float32)
        for i, (left, top, right, bottom) in enumerate(rois[:2]):
            scaled = torch.nn.functional.interpolate(source[i:i + 1], (bottom - top, right - left),
                                                       mode="bilinear", align_corners=False)
            x0, y0 = max(left, 0), max(top, 0)
            x1, y1 = min(right, width), min(bottom, height)
            expected[i, :, y0:y1, x0:x1] = scaled.cpu().numpy()[0, :,
                y0 - top:y1 - top, x0 - left:x1 - left]
        np.testing.assert_allclose(out.cpu().numpy(), expected, atol=2e-6, rtol=2e-6)
