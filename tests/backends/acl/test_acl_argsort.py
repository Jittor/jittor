"""Native/public ACL sorting keeps full permutations and their value gradients."""
import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor._runtime.fallback import forbid_backend_fallbacks


def _bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def _values(case):
    if case == "ties":
        return np.asarray([[2, 1, 2, -3, 0, -3, 1],
                           [4, 4, -2, 0, -2, 4, 1]], dtype=np.float32), -1
    if case == "nonlast":
        return (np.arange(42, dtype=np.float32).reshape(3, 2, 7) % 19) - 9, 1
    if case == "special":
        return np.asarray([[np.nan, -np.inf, 3, np.inf, -2, np.nan, -np.inf],
                           [np.inf, 0, np.nan, -np.inf, np.inf, -1, 0]], dtype=np.float32), -1
    if case in ("vocab", "masked_vocab"):
        rng = np.random.default_rng(821)
        data = np.stack([rng.permutation(151936), rng.permutation(151936)]).astype(np.float32) - 75968
        if case == "masked_vocab":
            data[:, np.arange(data.shape[1]) % 31 != 0] = -np.inf
        return data, -1
    return np.asarray([[3, -1, 7, 0, -4, 2, 5],
                       [6, 2, -3, 1, 0, 8, -5]], dtype=np.float32), -1


def _assert_placement(tensors, device):
    # Host extraction can offload an identity FP32 cast; verify the actual
    # execution result before any numpy()/CPU read changes its storage.
    for tensor in tensors:
        tensor.sync()
        assert tensor.location() == ("device" if device=="npu" else "cpu")
        assert tuple(jt.core.dispatch_context([tensor])) == (("acl", 0) if device=="npu" else ("cpu", -1))


_FINITE_CASES = ("unique", "ties", "nonlast", "vocab")
_NPU_CASES = _FINITE_CASES + ("special", "masked_vocab")


@pytest.mark.parametrize("device,dtype,case", [
    ("cpu", "float32", case) for case in _FINITE_CASES
] + [
    pytest.param("npu", dtype, case, marks=pytest.mark.npu)
    for dtype in ("float16", "float32", "bfloat16") for case in _NPU_CASES
])
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize("index_dtype", ("int32", "int64"))
def test_argsort_full_values_indices_and_gradient(device, dtype, case, descending, index_dtype):
    if device == "npu":
        require_accelerator("acl")
    data, dim = _values(case)
    if dtype == "float16":
        # Keep large-vocabulary finite keys inside FP16 range; form the
        # reference from the actual quantized values, including its ties.
        if case in ("vocab", "masked_vocab"):
            data = data / np.float32(4)
        data = data.astype(np.float16).astype(np.float32)
    elif dtype == "bfloat16":
        data = _bf16(data)
    axis = dim % data.ndim
    expected_values = np.sort(data, axis=axis)
    if descending:
        expected_values = np.flip(expected_values, axis=axis)
    with jt.flag_scope(use_cuda=int(device=="npu")), forbid_backend_fallbacks():
        source = jt.array(data, dtype=dtype).start_grad()
        indices, values = jt.argsort(source, dim, descending, index_dtype)
        indices.sync()
        values.sync()
        _assert_placement((source, indices, values), device)
        actual_indices = indices.numpy().copy()
        actual_values = values.float32().numpy().copy()
        assert str(indices.dtype) == index_dtype and str(values.dtype) == dtype
        np.testing.assert_array_equal(actual_values, expected_values)
        np.testing.assert_array_equal(
            np.take_along_axis(data, actual_indices.astype(np.int64), axis), actual_values)
        shape = [1] * data.ndim
        shape[axis] = data.shape[axis]
        expected_indices = np.broadcast_to(np.arange(data.shape[axis]).reshape(shape), data.shape)
        np.testing.assert_array_equal(np.sort(actual_indices, axis=axis), expected_indices)
        # Default unstable sort does not promise a particular order within an
        # equal-key run. Its full permutation must still scatter every distinct
        # cotangent to the index that this invocation actually returned.
        cotangent = (np.arange(data.size, dtype=np.float32).reshape(data.shape) % 97 + 1) / np.float32(128)
        weight = jt.array(cotangent, dtype="float32").stop_grad()
        gradient = jt.grad((values.float32() * weight).sum(), source)
        gradient.sync()
        _assert_placement((gradient,), device)
        expected_gradient = np.empty_like(data)
        np.put_along_axis(expected_gradient, actual_indices.astype(np.int64),
                          (_bf16(cotangent) if dtype=="bfloat16"
                           else cotangent.astype(np.float16).astype(np.float32)
                           if dtype=="float16" else cotangent), axis)
        np.testing.assert_array_equal(gradient.float32().numpy(), expected_gradient)


@pytest.mark.npu
@pytest.mark.parametrize("index_dtype", ("int32", "int64"))
def test_raw_argsort_forward_preserves_default_index_dtype(index_dtype):
    require_accelerator("acl")
    data, dim = _values("unique")
    with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
        source = jt.array(data).stop_grad()
        indices, values = jt.ops.argsort(source, dim, False, index_dtype)
        indices.sync()
        values.sync()
        assert str(indices.dtype) == index_dtype
        np.testing.assert_array_equal(values.numpy(), np.sort(data, axis=-1))
        np.testing.assert_array_equal(np.take_along_axis(data, indices.numpy(), -1), values.numpy())
        assert tuple(jt.core.dispatch_context([indices, values])) == ("acl", 0)
