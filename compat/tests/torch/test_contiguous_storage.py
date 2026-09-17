"""Torch contiguous materializes real storage views without breaking autograd."""

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def _check_contiguous(device):
    for kind in ("transpose", "broadcast"):
        source_np = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        if kind == "broadcast":
            source_np = source_np[:1]
        source = torch.tensor(source_np, device=device, requires_grad=True)
        view = source.transpose(0, 1) if kind == "transpose" else source.expand(4, 3)
        was_contiguous = view.is_contiguous()
        if kind == "broadcast":
            assert not was_contiguous
        packed = view.contiguous()
        assert type(packed) is torch.Tensor
        assert (packed is view) == was_contiguous
        assert packed.is_contiguous()
        assert packed.contiguous() is packed
        gradient, = torch.autograd.grad((packed * packed).sum(), source)
        with forbid_backend_fallbacks():
            packed.sync()
            gradient.sync()
            assert packed.placement_backend == (2 if device == "npu" else 0)
            assert packed.location() == ("device" if device == "npu" else "cpu")
            actual = packed.detach().cpu().numpy()
            actual_gradient = gradient.detach().cpu().numpy()
        expected = source_np.T if kind == "transpose" else np.broadcast_to(source_np, (4, 3))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_gradient, 2 * source_np * (4 if kind == "broadcast" else 1))


@pytest.mark.cpu
def test_as_strided_rejects_a_view_that_reaches_past_the_tensor():
    """The view `as_strided` builds is a gather over this tensor's own elements.

    So a view described past them cannot be served from here, and serving it
    anyway reads outside the tensor: on a device tensor that is a sticky
    `cudaErrorIllegalAddress` at whatever CUDA call happens next, with nothing
    naming the call that asked for the view. torch rejects the same request
    ("setStorage: sizes ..., strides ..., storage_offset ... are too large").

    Measured on the H3 TP2 request before this check existed: the failing op was
    a `getitem` whose index tensor's top value was `388956160 // 64 * 65 + 1`,
    against a `388956160`-element buffer -- a view one sixty-fourth past the end,
    built by this function.
    """
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # a view that fits is still served, element for element
    got = t.as_strided((2, 3), (4, 1), 0)
    np.testing.assert_allclose(got.numpy(), np.arange(12, dtype=np.float32).reshape(3, 4)[:2, :3])
    # one row past the end
    with pytest.raises(ValueError, match="as_strided"):
        t.as_strided((4, 4), (4, 1), 0)
    # and a storage offset that walks off the front
    with pytest.raises(ValueError, match="as_strided"):
        t.as_strided((3, 4), (4, 1), -1)


def test_contiguous_storage_and_gradient_cpu():
    with jt.flag_scope(use_cuda=0):
        _check_contiguous("cpu")


@pytest.mark.npu
def test_contiguous_storage_and_gradient_npu():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        _check_contiguous("npu")
