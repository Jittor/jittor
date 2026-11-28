"""RoI Align contracts through the deployed torchvision facade."""

import numpy as np
import pytest
import torch
from torchvision.ops import roi_align

import jittor as jt
from _helpers.capability import require_accelerator
from jittor.compat.torch import vision


def _reference(features, rois, output_size, scale, ratio, aligned, math):
    pooled_h, pooled_w = output_size
    height, width = features.shape[-2:]
    output = np.zeros((len(rois), features.shape[1], pooled_h, pooled_w), dtype=math)
    shift = math(0.5 if aligned else 0.0)
    for k, (image, left, top, right, bottom) in enumerate(rois):
        left, top, right, bottom = (math(math(value) * math(scale) - shift)
                                    for value in (left, top, right, bottom))
        ext_h, ext_w = math(bottom - top), math(right - left)
        if not aligned:
            ext_h, ext_w = max(ext_h, math(1)), max(ext_w, math(1))
        bin_h, bin_w = math(ext_h / math(pooled_h)), math(ext_w / math(pooled_w))
        ny, nx = (ratio, ratio) if ratio > 0 else (int(np.ceil(bin_h)), int(np.ceil(bin_w)))
        for h, w in np.ndindex(output_size):
            total = np.zeros(features.shape[1], dtype=math)
            for iy, ix in np.ndindex((max(ny, 0), max(nx, 0))):
                y = math(top + math(h) * bin_h) + math(iy + 0.5) * bin_h / math(ny)
                x = math(left + math(w) * bin_w) + math(ix + 0.5) * bin_w / math(nx)
                if y < -1 or y > height or x < -1 or x > width:
                    continue
                y, x = np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)
                y0, x0 = int(y), int(x)
                y1, x1 = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
                dy, dx = math(y - y0), math(x - x0)
                hy, hx = math(1 - dy), math(1 - dx)
                f = features[int(image)]
                total += (hy * hx * f[:, y0, x0] + hy * dx * f[:, y0, x1]
                          + dy * hx * f[:, y1, x0] + dy * dx * f[:, y1, x1])
            output[k, :, h, w] = total / math(max(ny * nx, 1))
    return output


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("aligned, ratio", [(False, -1), (True, 0), (True, 2), (False, 3)])
def test_cuda_roi_align(dtype, aligned, ratio, monkeypatch):
    require_accelerator("cuda")
    def fail_composite(*args):
        pytest.fail("expected fused CUDA RoI Align")

    monkeypatch.setattr(vision, "_roi_align_composite", fail_composite)
    rng = np.random.default_rng(31)
    data = rng.normal(size=(2, 3, 7, 9)).astype(np.float32)
    rois = [[0, -3, -2, 8.5, 9], [1, 0.125, 1.25, 12, 13],
            [0, 3, 3, 3, 3], [1, 22, 22, 27, 30], [1, 9.25, 8, 10.5, 9]]
    with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
        features = torch.tensor(data.transpose(0, 1, 3, 2), dtype=getattr(torch, dtype),
                                device="cuda").transpose(-1, -2)
        boxes = torch.tensor(rois, dtype=getattr(torch, dtype), device="cuda")
        actual = roi_align(features, boxes, (2, 4), 0.75, ratio, aligned)
        actual.sync()
        assert actual.location() == "device"
        math = np.float16 if dtype == "float16" else np.float32
        expected = _reference(features.float().cpu().numpy().astype(math), boxes.float().cpu().numpy(),
                              (2, 4), 0.75, ratio, aligned, math)
        tolerance = {"float32": 2e-6, "float16": 2e-3, "bfloat16": 5e-3}[dtype]
        np.testing.assert_allclose(actual.float().cpu().numpy(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_roi_align_box_lists_empty_and_feature_gradient(device):
    if device == "cuda":
        require_accelerator("cuda")
    with jt.flag_scope(backend_fallback="error"):
        image = torch.tensor([[[[0., 1.], [2., 3.]]], [[[4., 5.], [6., 7.]]]],
                             device=device, requires_grad=True)
        empty = torch.empty((0, 4), device=device)
        box = torch.tensor([[0., 0., 2., 2.]], device=device)
        output = roi_align(image, [empty, box], 1, sampling_ratio=1, aligned=True)
        np.testing.assert_allclose(output.cpu().numpy(), [[[[5.5]]]])
        output.sum().backward()
        expected = np.zeros((2, 1, 2, 2), dtype=np.float32)
        expected[1] = 0.25
        np.testing.assert_allclose(image.grad.cpu().numpy(), expected)
        assert tuple(roi_align(image, [empty, empty], (2, 3)).shape) == (0, 1, 2, 3)
