"""Nested padding preserves payload placement and every original input gradient.

These small regressions isolate the real worker's jagged-to-dense value path.
Expected tensors and cotangents are independent NumPy fixtures; NumPy is used
only for test inputs/references and for observations after device assertions.
"""

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def _assert_placement(value, device):
    native = getattr(value, "_var", value)
    native.sync()
    # Native operations may inherit the active runtime (-1); residency after
    # sync below, not the optional explicit-placement tag, proves execution.
    assert native.placement_backend in (-1, 2 if device == "npu" else 0)
    assert native.location() == ("device" if device == "npu" else "cpu")
    if device == "npu":
        assert native.device_id >= 0


def _snapshot(value):
    return np.array(value.detach().cpu().numpy(), copy=True)


def _list_inputs(device, explicit_size):
    """Both source leaves must receive their own nonconstant dense cotangent."""
    with forbid_backend_fallbacks():
        left = torch.tensor([2.0, -3.0], device=device, requires_grad=True)
        right = torch.tensor([4.0, 5.0, -6.0], device=device, requires_grad=True)
        nested = torch.nested.as_nested_tensor([left, right], layout=torch.jagged)
        width = 5 if explicit_size else 3
        padded = nested.to_padded_tensor(-7.0, output_size=(2, width) if explicit_size else None)
        _assert_placement(padded, device)
        assert padded.requires_grad, "nested padding detached every original input"
        cotangent_np = np.arange(1, 2 * width + 1, dtype=np.float32).reshape(2, width)
        cotangent = torch.tensor(cotangent_np, device=device)
        gradients = torch.autograd.grad((padded * cotangent).sum(), (left, right))
        assert len(gradients) == 2 and all(gradient is not None for gradient in gradients)
        for value in (left, right, padded) + tuple(gradients):
            _assert_placement(value, device)
        actual = [_snapshot(value) for value in (padded,) + tuple(gradients)]

    expected = np.full((2, width), -7.0, dtype=np.float32)
    expected[0, :2] = [2.0, -3.0]
    expected[1, :3] = [4.0, 5.0, -6.0]
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[1], cotangent_np[0, :2])
    np.testing.assert_array_equal(actual[2], cotangent_np[1, :3])


def _jagged_roundtrip(device, explicit_size):
    """Feature-valued jagged→padded→jagged keeps the original packed leaf."""
    payload_np = np.arange(1, 11, dtype=np.float32).reshape(5, 2)
    width = 4 if explicit_size else 3
    cotangent_np = np.arange(1, 4 * width + 1, dtype=np.float32).reshape(2, width, 2)
    with forbid_backend_fallbacks():
        payload = torch.tensor(payload_np, device=device, requires_grad=True)
        offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device=device)
        nested = torch.nested.nested_tensor_from_jagged(payload, offsets, jagged_dim=1)
        padded = nested.to_padded_tensor(
            9.0, output_size=(2, width, 2) if explicit_size else None,
        )
        _assert_placement(padded, device)
        assert padded.requires_grad, "jagged padding detached its packed values leaf"
        # Reconstruct the exact packed values by dropping only the padding.
        packed_again = torch.cat((padded[0, :2, :], padded[1, :3, :]), dim=0)
        rebuilt = torch.nested.nested_tensor_from_jagged(packed_again, offsets, jagged_dim=1)
        dense_again = rebuilt.to_padded_tensor(
            9.0, output_size=(2, width, 2) if explicit_size else None,
        )
        _assert_placement(dense_again, device)
        cotangent = torch.tensor(cotangent_np, device=device)
        gradient, = torch.autograd.grad((dense_again * cotangent).sum(), (payload,))
        assert gradient is not None
        for value in (payload, padded, packed_again, dense_again, gradient):
            _assert_placement(value, device)
        actual = [_snapshot(value) for value in (padded, packed_again, dense_again, gradient)]
        actual_offsets = _snapshot(rebuilt.offsets())

    expected = np.full((2, width, 2), 9.0, dtype=np.float32)
    expected[0, :2] = payload_np[:2]
    expected[1, :3] = payload_np[2:]
    expected_gradient = np.concatenate((cotangent_np[0, :2], cotangent_np[1, :3]), axis=0)
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[1], payload_np)
    np.testing.assert_array_equal(actual[2], expected)
    np.testing.assert_array_equal(actual[3], expected_gradient)
    np.testing.assert_array_equal(actual_offsets, [0, 2, 5])


def _public_narrow(device, explicit_size):
    raw = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    width = 4 if explicit_size else 3
    with forbid_backend_fallbacks():
        source = torch.tensor(raw, device=device, requires_grad=True)
        nested = torch.nested.narrow(
            source, 1, torch.tensor([1, 0], device=device),
            torch.tensor([2, 3], device=device), layout=torch.jagged)
        assert not nested.is_contiguous()
        np.testing.assert_array_equal(_snapshot(nested.values()), raw.reshape(10, 2))
        np.testing.assert_array_equal(_snapshot(nested.offsets()), [1, 5, 8])
        np.testing.assert_array_equal(_snapshot(nested.lengths()), [2, 3])
        packed = nested.contiguous()
        assert packed.is_contiguous() and packed.lengths() is None
        np.testing.assert_array_equal(_snapshot(packed.values()), np.concatenate((raw[0, 1:3], raw[1, :3])))
        padded = torch.nested.to_padded_tensor(
            nested, -3.0, output_size=(2, width, 2) if explicit_size else None)
        _assert_placement(padded, device)
        weights = np.arange(1, 4 * width + 1, dtype=np.float32).reshape(2, width, 2)
        gradient, = torch.autograd.grad((padded * torch.tensor(weights, device=device)).sum(), (source,))
        _assert_placement(gradient, device)
        expected = np.full((2, width, 2), -3.0, dtype=np.float32)
        expected[0, :2], expected[1, :3] = raw[0, 1:3], raw[1, :3]
        expected_gradient = np.zeros_like(raw)
        expected_gradient[0, 1:3], expected_gradient[1, :3] = weights[0, :2], weights[1, :3]
        np.testing.assert_array_equal(_snapshot(padded), expected)
        np.testing.assert_array_equal(_snapshot(gradient), expected_gradient)


def _batch_partition(device, explicit_size):
    raw = np.arange(10, dtype=np.float32).reshape(5, 2)
    with forbid_backend_fallbacks():
        source = torch.tensor(raw, device=device, requires_grad=True)
        nested = torch.nested.nested_tensor_from_jagged(
            source, torch.tensor([0, 2, 5], device=device), jagged_dim=1)
        parts = nested.split([1, 1]) if explicit_size else nested.chunk(3)
        assert len(parts) == 2
        assert all(part.ndimension() == 3 and part.requires_grad and not part.is_meta for part in parts)
        combined = torch.cat([part.values() for part in parts])
        weights = torch.tensor(raw + 1, device=device)
        gradient, = torch.autograd.grad((combined * weights).sum(), (source,))
        _assert_placement(combined, device)
        _assert_placement(gradient, device)
        np.testing.assert_array_equal(_snapshot(combined), raw)
        np.testing.assert_array_equal(_snapshot(gradient), raw + 1)


@pytest.mark.cpu
@pytest.mark.parametrize("explicit_size", [False, True], ids=["inferred_size", "larger_output_size"])
@pytest.mark.parametrize("case", [_list_inputs, _jagged_roundtrip, _public_narrow, _batch_partition], ids=["all_list_input_grads", "jagged_roundtrip", "public_narrow", "batch_partition"])
def test_nested_padding_cpu(case, explicit_size):
    with jt.flag_scope(use_cuda=0):
        case("cpu", explicit_size)


@pytest.mark.npu
@pytest.mark.parametrize("explicit_size", [False, True], ids=["inferred_size", "larger_output_size"])
@pytest.mark.parametrize("case", [_list_inputs, _jagged_roundtrip, _public_narrow, _batch_partition], ids=["all_list_input_grads", "jagged_roundtrip", "public_narrow", "batch_partition"])
def test_nested_padding_npu(case, explicit_size):
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("No ACL found")
    with jt.flag_scope(use_acl=1, use_cuda=1):
        case("npu", explicit_size)


def _jagged_dim2_references():
    """Independent binary-rational inputs and FP64 cotangent mapping; no Torch."""
    import math

    packed = [[(row * 8 + column - 13) / 16.0 for column in range(8)] for row in range(4)]
    cotangent = [[[((batch * 20 + row * 5 + column) - 19) / 8.0
                   for column in range(5)] for row in range(4)] for batch in range(2)]
    dense = [[[packed[row][column] if column < 3 else -7.25
               for column in range(5)] for row in range(4)],
             [[packed[row][3 + column] for column in range(5)] for row in range(4)]]
    gradient = [cotangent[0][row][:3] + cotangent[1][row][:5] for row in range(4)]
    loss = math.fsum(dense[batch][row][column] * cotangent[batch][row][column]
                     for batch in range(2) for row in range(4) for column in range(5))
    return packed, cotangent, dense, gradient, loss


def _jagged_dim2_resident(value, device, ordinal):
    native = getattr(value, "_var", value)
    native.sync()
    assert jt.core.dispatch_context([native])[0] == device
    assert native.location() == ("device" if device == "cuda" else "cpu")
    if device == "cuda":
        assert native.device_id == ordinal
        assert torch.cuda.current_device() == ordinal


def _jagged_dim2_padding(device, ordinal=0):
    """Explicit ragged axis 2: full forward/storage/cotangent contract."""
    packed_ref, cotangent_ref, dense_ref, gradient_ref, loss_ref = _jagged_dim2_references()
    target = device if device == "cpu" else "cuda:%d" % ordinal
    before = int(jt.core.backend_fallback_count())
    with forbid_backend_fallbacks():
        packed = torch.tensor(packed_ref, dtype=torch.float32, device=target, requires_grad=True)
        offsets = torch.tensor([0, 3, 8], dtype=torch.int64, device=target)
        nested = torch.nested.nested_tensor_from_jagged(packed, offsets, jagged_dim=2)
        assert nested.dim() == 3 and tuple(nested.shape[:2]) == (2, 4)
        assert nested.requires_grad and not nested.is_meta
        values = nested.values()
        assert tuple(values.shape) == (4, 8)
        _jagged_dim2_resident(packed, device, ordinal)
        _jagged_dim2_resident(values, device, ordinal)
        # Public storage alias, not private NestedTensor implementation state.
        assert values.data_ptr() == packed.data_ptr()
        pieces = nested.unbind(0)
        assert len(pieces) == 2
        assert tuple(pieces[0].shape) == (4, 3)
        assert tuple(pieces[1].shape) == (4, 5)
        padded = torch.nested.to_padded_tensor(nested, -7.25, output_size=(2, 4, 5))
        assert tuple(padded.shape) == (2, 4, 5) and padded.requires_grad
        cotangent = torch.tensor(cotangent_ref, dtype=torch.float32, device=target)
        loss = (padded * cotangent).sum()
        gradient, = torch.autograd.grad(loss, (packed,))
        assert gradient is not None and tuple(gradient.shape) == (4, 8)
        observed = (packed, offsets, nested.offsets(), values, pieces[0], pieces[1], padded, cotangent, loss, gradient)
        for value in observed:
            _jagged_dim2_resident(value, device, ordinal)
        # Only copied host observations, after every original device assertion.
        actual = [_snapshot(value) for value in observed]
        for value in observed:
            _jagged_dim2_resident(value, device, ordinal)
        assert values.data_ptr() == packed.data_ptr()
    assert int(jt.core.backend_fallback_count()) - before == 0
    packed_np = np.asarray(packed_ref, dtype=np.float64)
    np.testing.assert_array_equal(actual[0].astype(np.float64), packed_np)
    np.testing.assert_array_equal(actual[1], [0, 3, 8])
    np.testing.assert_array_equal(actual[2], [0, 3, 8])
    np.testing.assert_array_equal(actual[3].astype(np.float64), packed_np)
    np.testing.assert_array_equal(actual[4].astype(np.float64), packed_np[:, :3])
    np.testing.assert_array_equal(actual[5].astype(np.float64), packed_np[:, 3:])
    np.testing.assert_array_equal(actual[6].astype(np.float64), np.asarray(dense_ref, dtype=np.float64))
    np.testing.assert_array_equal(actual[7].astype(np.float64), np.asarray(cotangent_ref, dtype=np.float64))
    np.testing.assert_array_equal(actual[8].astype(np.float64), loss_ref)
    np.testing.assert_array_equal(actual[9].astype(np.float64), np.asarray(gradient_ref, dtype=np.float64))


@pytest.mark.cpu
def test_jagged_dim2_padding_cpu():
    with jt.runtime.scope(use_cuda=0, backend_fallback="error"):
        _jagged_dim2_padding("cpu")


@pytest.mark.cuda
def test_jagged_dim2_padding_cuda():
    import os

    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    ordinal = int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0"))
    assert ordinal >= 0 and torch.cuda.is_available()
    torch.cuda.set_device(ordinal)
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"):
        _jagged_dim2_padding("cuda", ordinal)


def _packed_jagged_detach(device, ordinal=0):
    target = "cpu" if device == "cpu" else "cuda:%d" % ordinal
    before = int(jt.core.backend_fallback_count())
    with forbid_backend_fallbacks():
        for axis, holes in ((1, False), (2, False), (1, True)):
            raw = np.arange(32 if axis == 2 else 8, dtype=np.float32)
            if axis == 2:
                raw = raw.reshape(4, 8)
            values = torch.tensor(raw, device=target, requires_grad=True)
            offsets = torch.tensor([0, 5, 8] if holes else [0, 3, 8],
                                   dtype=torch.int32, device=target)
            lengths = torch.tensor([3, 3], dtype=torch.int64, device=target) if holes else None
            nested = torch.nested.nested_tensor_from_jagged(
                values, offsets, lengths=lengths, jagged_dim=axis)
            assert isinstance(nested, torch.Tensor)
            assert not isinstance(nested, torch.nn.Parameter)
            assert not isinstance(object(), torch.Tensor)
            assert not isinstance(jt.array([1.0]), torch.Tensor)
            detached = nested.detach()
            assert detached is not nested
            assert isinstance(detached, torch.Tensor)
            assert nested.requires_grad and values.requires_grad
            assert not detached.requires_grad and not detached.values().requires_grad
            assert detached.offsets() is offsets and detached.lengths() is lengths
            assert detached.offsets().dtype == torch.int32
            assert tuple(detached.values().shape) == tuple(raw.shape)
            assert detached.values().data_ptr() == values.data_ptr()
            pieces = detached.unbind(0)
            assert all(not piece.requires_grad for piece in pieces)
            observed = (values, offsets, detached.values()) + tuple(pieces)
            if lengths is not None:
                observed += (lengths,)
            for value in observed:
                _jagged_dim2_resident(value, device, ordinal)
            np.testing.assert_array_equal(_snapshot(detached.values()), raw)
            np.testing.assert_array_equal(_snapshot(offsets), [0, 5, 8] if holes else [0, 3, 8])
            weights = torch.tensor(raw + 1, device=target)
            gradient, = torch.autograd.grad((values * weights).sum(), (values,))
            _jagged_dim2_resident(gradient, device, ordinal)
            np.testing.assert_array_equal(_snapshot(gradient), raw + 1)
            for value in observed:
                _jagged_dim2_resident(value, device, ordinal)
    assert int(jt.core.backend_fallback_count()) == before == 0


@pytest.mark.cpu
def test_packed_jagged_detach_cpu():
    with jt.runtime.scope(use_cuda=0, backend_fallback="error"):
        _packed_jagged_detach("cpu")


@pytest.mark.cuda
def test_packed_jagged_detach_cuda():
    import os
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    ordinal = int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0"))
    assert ordinal >= 0 and torch.cuda.is_available()
    torch.cuda.set_device(ordinal)
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"):
        _packed_jagged_detach("cuda", ordinal)


def _packed_jagged_pickle(device, ordinal=0):
    import pickle
    target = "cpu" if device == "cpu" else "cuda:%d" % ordinal
    before = int(jt.core.backend_fallback_count())
    with forbid_backend_fallbacks():
        for case in ("input_int32", "position_axis2", "floating", "holes"):
            axis = 2 if case == "position_axis2" else 1
            trainable = case in ("floating", "holes")
            raw = np.arange(32 if axis == 2 else 8, dtype=np.float32 if trainable else np.int64)
            if axis == 2: raw = raw.reshape(4, 8)
            offsets_np = np.array([0, 5, 8] if case == "holes" else [0, 3, 8],
                                  dtype=np.int32 if case == "input_int32" else np.int64)
            dtype = torch.float32 if trainable else torch.int64
            offset_dtype = torch.int32 if case == "input_int32" else torch.int64
            values = torch.tensor(raw, dtype=dtype, device=target, requires_grad=trainable)
            offsets = torch.tensor(offsets_np, dtype=offset_dtype, device=target)
            lengths = (torch.tensor([3, 3], dtype=torch.int64, device=target)
                       if case == "holes" else None)
            nested = torch.nested.nested_tensor_from_jagged(
                values, offsets=offsets, lengths=lengths, jagged_dim=axis)
            originals = (values, offsets) + (() if lengths is None else (lengths,))
            for value in originals: _jagged_dim2_resident(value, device, ordinal)
            # Joint ordinary pickle must retain memo identity for the public
            # components, including the complete buffer behind holes.
            restored, restored_values, restored_offsets, restored_lengths = pickle.loads(
                pickle.dumps((nested, values, offsets, lengths), protocol=5))
            assert restored.values() is restored_values and restored.offsets() is restored_offsets
            assert restored.lengths() is restored_lengths
            assert restored.dim() == nested.dim() and restored.shape[0] == 2
            if axis == 2: assert restored.shape[1] == 4
            assert restored.dtype == dtype and restored.requires_grad is trainable
            assert restored_values.dtype == dtype and restored_offsets.dtype == offset_dtype
            decoded = (restored_values, restored_offsets) + (() if lengths is None else (restored_lengths,))
            references = (raw, offsets_np) + (() if lengths is None else (np.array([3, 3], dtype=np.int64),))
            for value, reference in zip(decoded, references):
                _jagged_dim2_resident(value, device, ordinal)
                actual = _snapshot(value)
                assert actual.dtype == reference.dtype and actual.shape == reference.shape
                np.testing.assert_array_equal(actual, reference)
            if trainable:
                cotangent_np = np.arange(1, raw.size + 1, dtype=np.float32).reshape(raw.shape)
                cotangent = torch.tensor(cotangent_np, device=target)
                gradient, = torch.autograd.grad((restored_values * cotangent).sum(), (restored_values,))
                _jagged_dim2_resident(gradient, device, ordinal)
                np.testing.assert_array_equal(_snapshot(gradient), cotangent_np)
                _jagged_dim2_resident(gradient, device, ordinal)
            for value in originals + decoded: _jagged_dim2_resident(value, device, ordinal)
    assert int(jt.core.backend_fallback_count()) == before == 0


@pytest.mark.cpu
def test_packed_jagged_pickle_cpu():
    with jt.runtime.scope(use_cuda=0, backend_fallback="error"):
        _packed_jagged_pickle("cpu")


@pytest.mark.cuda
def test_packed_jagged_pickle_cuda():
    import os
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    ordinal = int(os.environ.get("JITTOR_TEST_DEVICE_INDEX", "0"))
    assert ordinal >= 0 and torch.cuda.is_available()
    torch.cuda.set_device(ordinal)
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"):
        _packed_jagged_pickle("cuda", ordinal)
