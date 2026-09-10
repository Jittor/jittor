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
def test_contiguous_storage_and_gradient_cpu():
    with jt.flag_scope(use_cuda=0):
        _check_contiguous("cpu")


@pytest.mark.npu
def test_contiguous_storage_and_gradient_npu():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        _check_contiguous("npu")
