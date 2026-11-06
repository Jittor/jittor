"""torch.unique_consecutive honours the input's device, not the ambient one."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unique_consecutive_on_cuda_input_under_host_ambient():
    # The Torch frontend runs with the host as its ambient placement; the
    # native implementation used to allocate its run mask there and mix
    # placements inside concat/setitem (transformers T5 test_determinism).
    x = torch.tensor([1, 1, 2, 2, 1, 3], device="cuda", dtype=torch.int32)
    out, inverse, counts = torch.unique_consecutive(x, return_inverse=True, return_counts=True)
    for value in (out, inverse, counts):
        assert str(value.device).startswith("cuda")
    assert out.cpu().numpy().tolist() == [1, 2, 1, 3]
    assert inverse.cpu().numpy().tolist() == [0, 0, 1, 1, 2, 3]
    assert counts.cpu().numpy().tolist() == [2, 2, 1, 1]
