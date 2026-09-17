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


@pytest.mark.cpu
def test_a_storage_serves_a_shard_at_a_nonzero_offset():
    """`set_(storage, offset, size, stride)` is byte-addressed from the origin.

    vLLM-Omni's `PinnedModuleStager` snapshots a module group as one byte image
    of its storage (`torch.empty(0, dtype=torch.uint8).set_(storage, 0,
    (nbytes,), (1,))`) and re-creates each member with `set_(storage, offset,
    shape, stride)`. A weight sliced out of a fused one is a shard with a
    nonzero `storage_offset()`, so reading the offset as an index into the
    shard's *own* elements describes a buffer the shard runs past. That is how
    the H3 text encoder's fused-weight shard failed on rank 1.
    """
    for dtype in (torch.float32, torch.bfloat16, torch.uint8):
        big = torch.arange(128, dtype=torch.float32).to(dtype).reshape(16, 8)
        for shard in (big[:8], big[8:]):
            storage = shard.untyped_storage()
            image = torch.empty(0, dtype=torch.uint8, device=shard.device).set_(
                storage, 0, (storage.nbytes(),), (1,))
            restored = torch.empty(0, dtype=dtype, device=image.device).set_(
                image.untyped_storage(), shard.storage_offset(),
                tuple(shard.shape), tuple(shard.stride()))
            np.testing.assert_array_equal(restored.float().numpy(),
                                          shard.float().numpy())
    # A strided shard has no physical byte range, so it is refused rather than
    # packed at positions a later strided read does not look at.
    strided = big.reshape(-1)[1::2]
    with pytest.raises(ValueError, match="set_"):
        storage = strided.untyped_storage()
        torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))


def test_contiguous_storage_and_gradient_cpu():
    with jt.flag_scope(use_cuda=0):
        _check_contiguous("cpu")


@pytest.mark.npu
def test_contiguous_storage_and_gradient_npu():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        _check_contiguous("npu")
