"""Python None must remain a compile-time absent Triton pointer."""

import numpy as np


def test_optional_pointer_none_selects_absent_branch():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(output, optional):
        # Keep the failure safe: report the branch without dereferencing NULL.
        # vLLM uses this same guard before writing processed logits.
        if optional is not None:
            tl.store(output, 9)
        else:
            tl.store(output, 3)

    output = torch.zeros((1,), dtype=torch.int32, device="cuda")
    present = torch.zeros((1,), dtype=torch.int32, device="cuda")
    for optional, expected in ((None, 3), (present, 9), (None, 3)):
        kernel[(1,)](output, optional)
        assert output.device.type == "cuda"
        np.testing.assert_array_equal(output.cpu().numpy(), [expected])


def test_optional_pointer_present_performs_guarded_store():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(output, optional):
        if optional is not None:
            tl.store(optional, 17)
        tl.store(output, 5)

    output = torch.zeros((1,), dtype=torch.int32, device="cuda")
    present = torch.zeros((1,), dtype=torch.int32, device="cuda")
    kernel[(1,)](output, present)
    np.testing.assert_array_equal(output.cpu().numpy(), [5])
    np.testing.assert_array_equal(present.cpu().numpy(), [17])
