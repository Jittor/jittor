import numpy as np
import pytest
import torch
from torchvision.ops import batched_nms, nms

import jittor as jt
from _helpers.capability import require_accelerator


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_grouped_nms_preserves_score_order_and_exact_category_ids(device):
    if device == "cuda":
        require_accelerator("cuda")
    with jt.flag_scope(backend_fallback="error"):
        boxes = torch.tensor([[0., 0., 3., 3.], [0., 0., 3., 3.], [1., 1., 4., 4.],
                              [0., 0., 3., 3.], [9., 9., 11., 11.]], device=device)
        scores = torch.tensor([0.8, 0.9, 0.75, 0.9, 0.75], device=device)
        classes = torch.tensor([2**50, 2**50, -7, 2**50 + 1, 2**50], device=device)
        result = batched_nms(boxes, scores, classes, 0.3)
        assert result.device.type == device
        assert result.dtype == torch.int64
        np.testing.assert_array_equal(result.cpu().numpy(), [1, 3, 2, 4])
        np.testing.assert_array_equal(nms(boxes, scores, 0.3).cpu().numpy(), [1, 2, 4])
        assert tuple(batched_nms(boxes[:0], scores[:0], classes[:0], 0.3).shape) == (0,)


@pytest.mark.cuda
def test_grouped_nms_handles_large_classes_with_bounded_scratch():
    require_accelerator("cuda")
    size = 2061
    with jt.flag_scope(backend_fallback="error"):
        boxes = torch.tensor([[0., 0., 2., 2.]], device="cuda").repeat(size, 1)
        scores = torch.arange(size, dtype=torch.float32, device="cuda")
        classes = torch.zeros(size, dtype=torch.int64, device="cuda")
        classes[1] = 1
        result = batched_nms(boxes, scores, classes, 0.5)
        result.sync()
        assert result.location() == "device"
        np.testing.assert_array_equal(result.cpu().numpy(), [size - 1, 1])
