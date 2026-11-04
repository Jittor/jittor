"""Torch non-tuple sequences can spell multiple indexing axes."""

from contextlib import ExitStack
import warnings

import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with ExitStack() as stack:
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        yield request.param
        if hasattr(torch, "_torch_compat_install_context"):
            assert jt.core.backend_fallback_count() == before


def check(value, expected, device):
    assert value.device.type == device
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        value.sync()
        assert value.location() == ("device" if device == "cuda" else "cpu")
        assert jt.core.dispatch_context([value])[0] == device
    np.testing.assert_array_equal(value.detach().cpu().numpy(), expected)


@pytest.mark.parametrize("index_device", ["cpu", "same"])
def test_tensor_sequence_read_write_add_and_gradient(device, index_device):
    target = device if index_device == "same" else "cpu"
    rows = torch.tensor([0, 1], dtype=torch.long, device=target)
    cols = torch.tensor([1, 3], dtype=torch.long, device=target)
    indices = [rows, cols]
    source = torch.tensor(np.arange(8, dtype=np.float32).reshape(2, 4),
                          device=device, requires_grad=True)
    with pytest.warns(UserWarning, match="non-tuple sequence"):
        result = source[indices]
    check(result, [1., 7.], device)
    result.sum().backward()
    check(source.grad, [[0., 1., 0., 0.], [0., 0., 0., 1.]], device)
    values = source.detach().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values[indices] = torch.tensor([10., 20.], device=device)
        values[indices] += torch.tensor([2., 3.], device=device)
    check(values, [[0., 12., 2., 3.], [4., 5., 6., 23.]], device)
    check(rows, [0, 1], target)
    check(cols, [1, 3], target)
    assert indices[0] is rows and indices[1] is cols


@pytest.mark.parametrize("kind", ["nested", "slice", "ellipsis", "none"])
def test_short_sequence_multiaxis_spellings(device, kind):
    source = torch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=device)
    index, expected = {
        "nested": ([[0, 1], [2, 0]], [2., 3.]),
        "slice": ([slice(None), 1], [1., 4.]),
        "ellipsis": ([Ellipsis, 1], [1., 4.]),
        "none": ([None, 1], [[3., 4., 5.]]),
    }[kind]
    with pytest.warns(UserWarning, match="non-tuple sequence"):
        result = source[index]
    check(result, expected, device)


@pytest.mark.parametrize("index", [[1, 0], [], [[0]] * 32])
def test_single_axis_sequences_keep_their_shape(device, index):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = torch.tensor(values, device=device)
    expected = values[np.asarray(index, dtype=np.int64)]
    with warnings.catch_warnings(record=True) as emitted:
        result = source[index]
    assert not [item for item in emitted if "non-tuple sequence" in str(item.message)]
    check(result, expected, device)
