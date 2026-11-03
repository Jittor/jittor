"""CPU advanced indices may select accelerator data without moving the source."""

from contextlib import ExitStack

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


def assert_device(value, expected):
    assert str(value.device).split(":")[0] == expected
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        value.sync()
        assert value.location() == ("device" if expected == "cuda" else "cpu")
        assert jt.core.dispatch_context([value])[0] == expected


@pytest.mark.parametrize("kind", ["bool", "empty_bool", "long", "int32", "tuple_bool", "tuple_mixed"])
def test_cpu_indices_preserve_values_gradients_and_source_placement(device, kind):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = torch.tensor(values, device=device, requires_grad=True)
    if kind in ("bool", "empty_bool"):
        raw = np.array([kind == "bool", False])
        index = torch.tensor(raw, device="cpu")
        selection, expected_index = index, raw
    elif kind in ("long", "int32"):
        raw = np.array([1, 0, 1], dtype=np.int64 if kind == "long" else np.int32)
        index = torch.tensor(raw, device="cpu")
        selection, expected_index = index, raw
    elif kind == "tuple_bool":
        raw = np.array([True, False, True])
        index = torch.tensor(raw, device="cpu")
        selection, expected_index = (slice(None), index), (slice(None), raw)
    else:
        raw = np.array([1, 0, 1], dtype=np.int64)
        index = torch.tensor(raw, device="cpu")
        columns = torch.tensor([0, 2, 0], dtype=torch.long, device=device)
        selection, expected_index = (index, columns), (raw, np.array([0, 2, 0]))
    result = source[selection]
    assert_device(result, device)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), values[expected_index])
    result.sum().backward()
    expected_gradient = np.zeros_like(values)
    np.add.at(expected_gradient, expected_index, 1.)
    assert_device(source.grad, device)
    np.testing.assert_array_equal(source.grad.detach().cpu().numpy(), expected_gradient)
    assert_device(index, "cpu")
    np.testing.assert_array_equal(index.numpy(), raw)


@pytest.mark.parametrize("kind", ["bool", "long", "tuple"])
def test_native_var_cpu_indices_are_accepted_by_torch_tensor(device, kind):
    if not hasattr(torch, "_torch_compat_install_context"):
        pytest.skip("Native Jittor Var interoperability requires the Jittor runtime")
    import jittor as jt
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = torch.tensor(values, device=device, requires_grad=True)
    raw = np.array([True, False]) if kind == "bool" else np.array([1, 0, 1], dtype=np.int64)
    index = jt.array(raw, dtype=str(raw.dtype)).to("cpu")
    assert type(index) is jt.Var
    assert isinstance(index.device, str)
    index.sync()
    before = (index.device, index.location(), index.placement_backend,
              jt.core.dispatch_context([index]))
    assert before[:2] == ("cpu", "cpu")
    selection = (index, slice(None)) if kind == "tuple" else index
    result = source[selection]
    assert_device(result, device)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), values[raw])
    result.sum().backward()
    expected_gradient = np.zeros_like(values)
    np.add.at(expected_gradient, raw, 1.)
    assert_device(source.grad, device)
    np.testing.assert_array_equal(source.grad.detach().cpu().numpy(), expected_gradient)
    # Native legacy Vars can retain ambient dispatch with placement_backend=-1
    # despite physical CPU residency. Preserve that original metadata too;
    # only the copied index and gathered data belong to the target device.
    index.sync()
    assert (index.device, index.location(), index.placement_backend,
            jt.core.dispatch_context([index])) == before
    np.testing.assert_array_equal(index.numpy(), raw)


def test_nll_loss_internal_native_index_values_and_gradient(device):
    # The Jittor implementation constructs a native jt.index for rows, then
    # indexes a Torch Tensor with that Var and the Torch target together.
    values = np.array([[-.5, -1.5, -2.5], [-3.5, -4.5, -5.5]], dtype=np.float32)
    source = torch.tensor(values, device=device, requires_grad=True)
    targets = torch.tensor([2, 0], device=device, dtype=torch.long)
    result = torch.nn.functional.nll_loss(source, targets)
    assert_device(result, device)
    assert result.item() == pytest.approx(3.)
    result.backward()
    assert_device(source.grad, device)
    np.testing.assert_array_equal(source.grad.detach().cpu().numpy(),
                                  [[0., 0., -.5], [-.5, 0., 0.]])


@pytest.mark.cuda
def test_cpu_source_rejects_cuda_advanced_index():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = torch.tensor([1., 2.], device="cpu")
    index = torch.tensor([1, 0], device="cuda", dtype=torch.long)
    with pytest.raises(RuntimeError):
        source[index]


@pytest.mark.cuda
def test_other_cuda_device_index_is_not_silently_moved():
    if torch.cuda.device_count() < 2:
        pytest.skip("Need two CUDA devices")
    source = torch.tensor([1., 2.], device="cuda:0")
    index = torch.tensor([1, 0], device="cuda:1", dtype=torch.long)
    with pytest.raises(RuntimeError):
        source[index]
