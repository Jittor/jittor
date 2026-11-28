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
            assert packed.placement_backend == {"cpu": 0, "cuda": 1, "npu": 2}[device]
            assert packed.location() == ("cpu" if device == "cpu" else "device")
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


@pytest.mark.cpu
def test_a_view_with_no_elements_fits_any_tensor():
    """A zero-length axis addresses nothing, so no tensor is too small for it.

    torch requires a view's `storage_offset + sum((size - 1) * stride) + 1`
    elements only when the view *has* elements: `arange(10).as_strided((0,),
    (1,), 10)`, `(3, 0, 2)` shapes and every view of an empty tensor come back
    empty rather than raising. The bounds check added with the out-of-bounds
    gather fix computed that span unconditionally, so `s <= 0` axes dropped out
    of it while `storage_offset` stayed in and an empty view read as one
    starting past the end. It refused all three -- and with them `x.set_(y)`
    for an empty `y`, which is how layerwise offload swaps a parameter for a
    zero-element placeholder.
    """
    t = torch.arange(10, dtype=torch.float32)
    assert tuple(t.as_strided((0,), (1,), 0).shape) == (0,)
    assert tuple(t.as_strided((0,), (1,), 10).shape) == (0,)   # offset at the end
    assert tuple(t.as_strided((3, 0, 2), (1, 1, 1), 8).shape) == (3, 0, 2)
    empty = torch.empty(0, dtype=torch.float32)
    assert tuple(empty.as_strided((0,), (1,), 0).shape) == (0,)
    # so a tensor can be re-pointed at an empty one, as torch allows
    placeholder = torch.ones(3, dtype=torch.float32)
    placeholder.set_(empty)
    assert placeholder.numel() == 0
    # an empty storage holds no bytes -- a reach of "offset plus one element"
    # would hand a caller sizing a buffer from it an element that is not there
    assert empty.untyped_storage().nbytes() == 0
    # a storage offset before the origin is still refused, elements or not
    with pytest.raises(ValueError, match="as_strided"):
        t.as_strided((0,), (1,), -1)
    with pytest.raises(ValueError, match="as_strided"):
        t.as_strided((2,), (1,), -1)


@pytest.mark.cpu
def test_a_shard_reads_back_from_the_storage_that_carries_it():
    """`set_(storage, offset, ...)` addresses bytes the *carrier* actually holds.

    `storage_offset()` counts from the allocation's origin and `nbytes()` now
    reaches from that origin too, but the bytes behind the storage object are
    only the carrier tensor's own -- they begin at `storage_offset()`, not at
    the origin. Slicing them with an origin-relative address is off by exactly
    that offset, so re-reading a fused weight's tail from its own storage (what
    `PinnedModuleStager` does for every member) sliced past the end: an empty
    segment, then a gather of 64 indices into 0 elements, which is the
    out-of-bounds device read this file's other tests exist to refuse.
    """
    for dtype in (torch.float32, torch.bfloat16, torch.uint8):
        big = torch.arange(128, dtype=torch.float32).to(dtype).reshape(16, 8)
        for shard in (big[:8], big[8:]):
            storage = shard.untyped_storage()
            back = torch.empty(0, dtype=dtype, device=shard.device).set_(
                storage, shard.storage_offset(), tuple(shard.shape),
                tuple(shard.stride()))
            np.testing.assert_array_equal(back.float().numpy(),
                                          shard.float().numpy())
    # The rest of the allocation has no Python handle here, so a request for
    # bytes outside the carrier is refused rather than answered with the
    # carrier's own bytes read at the wrong place.
    tail = big[8:]
    with pytest.raises(ValueError, match="set_"):
        torch.empty(0, dtype=big.dtype).set_(
            tail.untyped_storage(), 0, (8, 8), (8, 1))


def test_contiguous_storage_and_gradient_cpu():
    with jt.flag_scope(use_cuda=0):
        _check_contiguous("cpu")


@pytest.mark.cuda
def test_contiguous_storage_and_gradient_cuda():
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        _check_contiguous("cuda")


@pytest.fixture
def cuda_contiguous_scope():
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0), forbid_backend_fallbacks():
        yield


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,axes", [
    pytest.param((7, 1), (1, 0), id="singleton-no-copy"),
    pytest.param((31, 7), (1, 0), id="matrix"),
    pytest.param((2, 32, 33), (0, 2, 1), id="b-c-hw-to-b-hw-c"),
    pytest.param((2, 65, 64), (0, 2, 1), id="b-hw-c-to-b-c-hw"),
    pytest.param((2, 33, 7, 31), (0, 2, 3, 1), id="nchw-nhwc"),
    pytest.param((2, 7, 31, 33), (0, 3, 1, 2), id="nhwc-nchw"),
    pytest.param((2, 7, 1, 33), (0, 2, 3, 1), id="singleton-spatial-axis"),
    pytest.param((2, 7, 33, 64), (0, 2, 1, 3), id="attention-heads-sequence"),
    pytest.param((2, 7, 33, 65), (0, 2, 3, 1), id="attention-heads-last"),
    pytest.param((2, 3, 1, 7, 33), (0, 2, 4, 1, 3), id="five-axes"),
    pytest.param((2, 2, 7, 3, 7, 33), (0, 5, 1, 3, 2, 4), id="swin-window-axes"),
])
def test_cuda_contiguous_transpose_values_and_gradients(cuda_contiguous_scope, dtype, shape, axes):
    rng = np.random.default_rng(0)
    values = rng.integers(-64, 64, size=shape).astype(np.float32)
    source = torch.tensor(values, dtype=dtype, device="cuda", requires_grad=True)
    view = source.permute(*axes)
    packed = view.contiguous()
    expected = values.transpose(axes)
    weights = rng.integers(-32, 32, size=expected.shape).astype(np.float32)
    cotangent = torch.tensor(weights, dtype=dtype, device="cuda")
    gradient, = torch.autograd.grad(packed, source, grad_outputs=cotangent)
    packed.sync()
    gradient.sync()
    assert packed.is_contiguous()
    assert packed.dtype == gradient.dtype == dtype
    assert packed.location() == gradient.location() == "device"
    np.testing.assert_array_equal(packed.detach().float().cpu().numpy(), expected)
    np.testing.assert_array_equal(gradient.detach().float().cpu().numpy(),
                                  weights.transpose(np.argsort(axes)))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,np_dtype,scale", [
    (torch.int32, np.int32, 123457),
    (torch.int64, np.int64, 274877906943),
])
def test_cuda_contiguous_transpose_integer_bits(cuda_contiguous_scope, dtype, np_dtype, scale):
    values = ((np.arange(2 * 33 * 65) - 2048) * scale).astype(np_dtype).reshape(2, 33, 65)
    packed = torch.tensor(values, dtype=dtype, device="cuda").transpose(1, 2).contiguous()
    packed.sync()
    assert packed.dtype == dtype
    assert packed.is_contiguous()
    assert packed.location() == "device"
    np.testing.assert_array_equal(packed.cpu().numpy(), values.transpose(0, 2, 1))


@pytest.mark.cuda
def test_cuda_contiguous_transpose_offset_and_gapped_rows(cuda_contiguous_scope):
    shape = (2, 33, 8, 65)
    values = (np.arange(np.prod(shape)) % 127).astype(np.float16).reshape(shape)
    source = torch.tensor(values, device="cuda", requires_grad=True)
    region = (slice(None), slice(1, None, 2), slice(1, None), slice(1, 64))
    axes = (0, 2, 3, 1)
    view = source[region].permute(*axes)
    assert view.storage_offset() % 8 != 0
    assert tuple(view.stride()) == (33 * 8 * 65, 65, 1, 2 * 8 * 65)
    packed = view.contiguous()
    packed.sync()
    assert packed.is_contiguous()
    assert packed.location() == "device"
    np.testing.assert_array_equal(packed.detach().cpu().numpy(), values[region].transpose(axes))
    gradient, = torch.autograd.grad(packed.float().sum(), source)
    expected_gradient = np.zeros_like(values)
    expected_gradient[region] = 1
    np.testing.assert_array_equal(gradient.detach().cpu().numpy(), expected_gradient)


@pytest.mark.cuda
def test_cuda_contiguous_transpose_broadcast_and_empty(cuda_contiguous_scope):
    values = np.arange(7 * 2 * 33, dtype=np.float32).reshape(1, 7, 2, 33)
    source = torch.tensor(values, device="cuda", requires_grad=True)
    view = source.permute(0, 2, 3, 1).expand(3, 2, 33, 7)
    assert view.stride()[0] == 0
    packed = view.contiguous()
    packed.sync()
    assert packed.is_contiguous()
    assert packed.location() == "device"
    np.testing.assert_array_equal(packed.detach().cpu().numpy(),
                                  np.broadcast_to(values.transpose(0, 2, 3, 1), (3, 2, 33, 7)))
    gradient, = torch.autograd.grad(packed.sum(), source)
    np.testing.assert_array_equal(gradient.detach().cpu().numpy(), np.full_like(values, 3))
    empty = source[:, :, :0, :].permute(0, 2, 3, 1).contiguous()
    empty.sync()
    assert tuple(empty.shape) == (1, 0, 33, 7)
    assert empty.is_contiguous()


@pytest.mark.cuda
def test_cuda_contiguous_transpose_more_tiles_than_blocks(cuda_contiguous_scope):
    values = np.arange(65537 * 4, dtype=np.int32).reshape(65537, 2, 2)
    packed = torch.tensor(values, device="cuda").transpose(1, 2).contiguous()
    packed.sync()
    assert packed.is_contiguous()
    assert packed.location() == "device"
    np.testing.assert_array_equal(packed.cpu().numpy(), values.transpose(0, 2, 1))


@pytest.mark.cuda
def test_cuda_contiguous_transpose_with_elementwise_consumer(cuda_contiguous_scope):
    values = (np.arange(2 * 7 * 31 * 33) % 127).astype(np.float32).reshape(2, 7, 31, 33)
    source = torch.tensor(values, dtype=torch.float16, device="cuda", requires_grad=True)
    result = source.permute(0, 3, 1, 2).contiguous() * 2 + 1
    result.sync()
    assert result.location() == "device"
    np.testing.assert_array_equal(result.detach().float().cpu().numpy(),
                                  values.transpose(0, 3, 1, 2) * 2 + 1)
    gradient, = torch.autograd.grad(result.float().sum(), source)
    np.testing.assert_array_equal(gradient.detach().float().cpu().numpy(), np.full_like(values, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("row_width,start,width", [
    (128, 8, 96),  # Aligned nonzero storage offset and aligned row strides.
    (128, 1, 96),  # Misaligned storage offset.
    (100, 0, 96),  # Consecutive rows do not all begin at vector boundaries.
    (128, 0, 95),  # Partial vector at the end of every row.
])
def test_contiguous_packs_permuted_rows(device, row_width, start, width):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    shape = (1, 2, 3, 3, 2, row_width)
    source_np = (np.arange(np.prod(shape)) % 127).astype(np.float16).reshape(shape)
    with jt.flag_scope(use_cuda=int(device == "cuda"), auto_graph_replay=0), forbid_backend_fallbacks():
        source = torch.tensor(source_np, device=device, requires_grad=True)
        view = source[..., start:start + width].permute(0, 1, 3, 2, 4, 5)
        assert not view.is_contiguous()
        packed = view.contiguous()
        packed.sync()
        assert packed.is_contiguous()
        assert packed.location() == ("cpu" if device == "cpu" else "device")
        expected = source_np[..., start:start + width].transpose(0, 1, 3, 2, 4, 5)
        np.testing.assert_array_equal(packed.detach().cpu().numpy(), expected)
        gradient, = torch.autograd.grad(packed.float().sum(), source)
        expected_gradient = np.zeros_like(source_np)
        expected_gradient[..., start:start + width] = 1
        np.testing.assert_array_equal(gradient.detach().cpu().numpy(), expected_gradient)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_contiguous_restores_context_and_respects_native_override(device, monkeypatch):
    if device == "cuda" and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    with jt.flag_scope(auto_graph_replay=0):
        source = torch.tensor([[1., 2.], [3., 4.]], device=device)
        view = source.t()
        type_token = jt.core._set_tensor_frontend_type(jt.Var)
        placement_token = jt.core._set_tensor_placement(1 if device == "cpu" else 0, 0)
        try:
            requested = jt.core._current_tensor_placement()
            packed = view.contiguous()
            assert type(packed) is torch.Tensor
            assert type(jt.core.ops.empty((0,), "float32")) is jt.Var
            assert jt.core._current_tensor_placement() == requested
        finally:
            jt.core._reset_tensor_placement(placement_token)
            jt.core._reset_tensor_frontend_type(type_token)
        packed.sync()
        assert packed.device.type == device
        assert packed.location() == ("cpu" if device == "cpu" else "device")
        packed.fill_(9)
        np.testing.assert_array_equal(source.numpy(), [[1., 2.], [3., 4.]])
        sentinel = object()
        monkeypatch.setattr(jt, "contiguous", lambda tensor: sentinel)
        assert view.contiguous() is sentinel
        with pytest.raises(NotImplementedError, match="contiguous_format"):
            view.contiguous(memory_format="channels_last")


def _check_repeat_is_a_dense_copy(device):
    """``Tensor.repeat`` copies; it is not an expanded view of the source.

    Jittor's native repeat is a broadcast view (stride 0 on the reproduced
    axes), so a write through it was refused -- PEFT's OFT Cayley expansion
    does ``torch.eye(b).repeat(n, 1, 1).add_(skew, alpha=2)`` -- and the
    result did not follow, or detach from, the source's own storage.
    """
    source = torch.eye(3, device=device)
    repeated = source.repeat(2, 1, 1)
    assert tuple(repeated.shape) == (2, 3, 3)
    assert repeated.is_contiguous()
    assert repeated.data_ptr() != source.data_ptr()
    repeated.add_(torch.full_like(repeated, 2.0))
    expected = np.array([[3., 2., 2.],
                         [2., 3., 2.],
                         [2., 2., 3.]], dtype=np.float32)
    np.testing.assert_array_equal(repeated.detach().cpu().numpy(),
                                  np.stack([expected, expected]))
    np.testing.assert_array_equal(source.detach().cpu().numpy(),
                                  np.eye(3, dtype=np.float32))
    # torch.tile is a copy too, and routes through repeat here.
    tiled = source.tile(2, 1, 1)
    assert tiled.is_contiguous()
    tiled.add_(1.0)
    np.testing.assert_array_equal(source.detach().cpu().numpy(),
                                  np.eye(3, dtype=np.float32))
    # repeat_interleave with an int count is expand + reshape natively; its
    # reshape is a stride-0 view. PEFT's HRA adapter makes a trainable
    # parameter out of it and trains in place.
    column = torch.arange(3, dtype=torch.float32, device=device).reshape(3, 1)
    interleaved = torch.repeat_interleave(column, 2, dim=1)
    assert tuple(interleaved.shape) == (3, 2)
    assert interleaved.is_contiguous()
    interleaved.add_(1.0)
    np.testing.assert_array_equal(column.detach().cpu().numpy(),
                                  np.arange(3, dtype=np.float32).reshape(3, 1))
    method_form = column.repeat_interleave(2, dim=1)
    assert method_form.is_contiguous()
    method_form.add_(1.0)
    np.testing.assert_array_equal(column.detach().cpu().numpy(),
                                  np.arange(3, dtype=np.float32).reshape(3, 1))


def test_deepcopy_owns_independent_storage():
    """`copy.deepcopy(p.data)` must not keep writing through `p`.

    The copy is rebuilt from bytes, as torch makes an independent tensor, so
    the `.data` owner/path bookkeeping of the original must not travel with
    it. PEFT's SHiRA adapter does `weight = copy.deepcopy(base.weight.data);
    weight += sparse_delta`; carrying the marker wrote the update into the
    base parameter and dropped the delta's gradient edge.
    """
    import copy
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    copied = copy.deepcopy(parameter.data)
    assert copied.data_ptr() != parameter.data_ptr()
    copied.add_(1.0)
    np.testing.assert_array_equal(parameter.detach().cpu().numpy(),
                                  np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(copied.detach().cpu().numpy(),
                                  np.full((2, 2), 2.0, dtype=np.float32))


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if capability.any_accelerator_enabled(backend=jt) else []))
def test_repeat_is_a_dense_copy(device):
    _check_repeat_is_a_dense_copy(device)


@pytest.mark.parametrize("in_channels", [16, 1])
def test_cudnn_inference_filter_materializes_dense_ohwi_for_torch_parameter(in_channels):
    """`_inference_filter` must produce dense OHWI storage for Torch parameters.

    Under the Torch frontend, `Tensor.clone()` defaults to `preserve_format`,
    which kept the transposed filter view's OIHW strides and caused every
    subsequent cuDNN conv2d call to launch a full weight `ContiguousOp` copy.
    """
    from jittor.nn.backends import cudnn as cudnn_backend

    rng = np.random.default_rng(0)
    values = rng.standard_normal((16, in_channels, 3, 3)).astype(np.float16)
    weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float16))
    x = torch.zeros((2, in_channels, 8, 8), dtype=torch.float16)
    jt.sync([weight, x])
    with torch.no_grad():
        first, layout1 = cudnn_backend._inference_filter(x, weight, 1)
        second, layout2 = cudnn_backend._inference_filter(x, weight, 1)
    assert layout1 == layout2 == "ohwi"
    assert first._storage_is_contiguous()
    assert second._storage_is_contiguous()
    assert tuple(first.shape) == (16, 3, 3, in_channels)
    expected_strides = (3 * 3 * in_channels, 3 * in_channels, in_channels, 1)
    assert tuple(second._storage_strides())[:3] == expected_strides[:3]
    if in_channels > 1:
        assert second._storage_strides()[3] == 1
    assert not weight.is_stop_grad()
    assert second.is_stop_grad()
    np.testing.assert_array_equal(first.detach().cpu().numpy(), values.transpose(0, 2, 3, 1))
    np.testing.assert_array_equal(second.detach().cpu().numpy(), values.transpose(0, 2, 3, 1))
    np.testing.assert_array_equal(weight.detach().cpu().numpy(), values)


@pytest.mark.npu
def test_contiguous_storage_and_gradient_npu():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        _check_contiguous("npu")
