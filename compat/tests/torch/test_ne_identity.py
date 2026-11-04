"""Real-tensor self inequality uses the canonical NaN predicate on its device."""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


_ACCELERATOR_DEVICE = "npu" if jt.compiler.has_acl else "cuda"
_DEVICES = ["cpu"] + ([_ACCELERATOR_DEVICE] if _test_capability.any_accelerator_enabled(backend=jt) else [])
_PLACEMENT_SPECS = [("cpu", "cpu")] + (
    [("cpu", _ACCELERATOR_DEVICE), (_ACCELERATOR_DEVICE, _ACCELERATOR_DEVICE),
     (_ACCELERATOR_DEVICE, "cpu")] if len(_DEVICES) > 1 else [])


@pytest.fixture(autouse=True)
def _device_scope_and_no_fallbacks(request):
    device = request.node.callspec.params["device"]
    with jt.runtime.scope(use_cuda=int(device != "cpu")), forbid_backend_fallbacks():
        yield


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("spelling", ["operator", "method"])
def test_same_real_tensor_ne_preserves_nan_inf_zero_and_dtype(device, dtype, spelling):
    source = (np.asarray([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-7, 1.0], dtype=dtype)
              if dtype.startswith("float") else np.asarray([0, 1, 0, 1], dtype=dtype))
    value = torch.tensor(source, dtype=getattr(torch, dtype), device=device)
    actual = value != value if spelling == "operator" else value.ne(value)
    assert actual.dtype == torch.bool
    assert actual.device.type == device
    assert tuple(actual.shape) == source.shape
    assert not actual.requires_grad
    np.testing.assert_array_equal(actual.cpu().numpy(), np.isnan(source))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("spelling", ["operator", "method", "function"])
def test_different_holders_and_values_keep_elementwise_ne(device, spelling):
    # Finite inputs also cover the untouched function spelling, without
    # claiming to repair the separate function owner's device infinity bug.
    source = np.asarray([[-0.0, 1.0, 2.0], [3.0, -4.0, 0.0]], dtype="float32")
    value = torch.tensor(source, device=device)
    alias = value.view_as(value)
    assert alias is not value
    different = torch.tensor(source + np.asarray([0.0, 1.0, -1.0]),
                             dtype=torch.float32, device=device)
    for other, reference in ((alias, source != source),
                             (different, source != (source + [0.0, 1.0, -1.0]))):
        if spelling == "operator":
            actual = value != other
        elif spelling == "method":
            actual = value.ne(other)
        else:
            actual = torch.ne(value, other)
        assert actual.dtype == torch.bool and actual.device.type == device
        np.testing.assert_array_equal(actual.cpu().numpy(), reference)


@pytest.mark.parametrize("device", _DEVICES)
def test_self_inequality_result_stays_fresh_after_source_mutation(device):
    value = torch.tensor([1.0, float("nan"), -0.0], device=device)
    first = value != value
    np.testing.assert_array_equal(first.cpu().numpy(), [False, True, False])
    with torch.no_grad():
        value.copy_(torch.tensor([float("nan"), 2.0, float("inf")], device=device))
    second = value.ne(value)
    np.testing.assert_array_equal(second.cpu().numpy(), [True, False, False])
    np.testing.assert_array_equal(first.cpu().numpy(), [False, True, False])


def _comparison_graph_values(result):
    seen, stack = set(), [result]
    while stack:
        value = stack.pop()
        if value.var_ptr in seen:
            continue
        seen.add(value.var_ptr)
        yield value
        if not value.is_finished:
            stack.extend(value._inputs())


@pytest.mark.parametrize("device,ambient", _PLACEMENT_SPECS)
@pytest.mark.parametrize("dtype", ["float16", "float32", "int32", "int64", "bool"])
@pytest.mark.parametrize("scalar_name", ["zero", "half", "numpy_int", "bool"])
@pytest.mark.parametrize("spelling", ["operator", "method"])
def test_ne_python_scalar_is_constructed_on_tensor_device(device, ambient, dtype, scalar_name, spelling):
    from jittor.compat.torch.frontend import tensor_frontend
    scalar = {"zero": 0, "half": 0.5, "numpy_int": np.int64(1), "bool": True}[scalar_name]
    source = np.asarray([-2, 0, 1, 3], dtype=dtype)
    with jt.runtime.scope(use_cuda=int(ambient != "cpu")), jt.no_grad():
        value = torch.tensor(source, dtype=getattr(torch, dtype), device=device)
        with tensor_frontend(torch.Tensor, device=ambient):
            result = value != scalar if spelling == "operator" else value.ne(scalar)
        if device == "cpu":
            # Check construction before executing: accepting an ACL scalar and
            # silently copying it to CPU would produce correct numbers while
            # introducing a real device dependency into a host-only mask.
            for node in _comparison_graph_values(result):
                assert node.placement_backend in (-1, 0)
                if node._producer_name() == "array" and not tuple(node.shape):
                    assert node.placement_backend == 0
        assert result.dtype == torch.bool and result.device.type == device
        np.testing.assert_array_equal(result.cpu().numpy(), source != scalar)
        assert value.device.type == device


@pytest.mark.parametrize("device", _DEVICES)
def test_ne_existing_tensor_operand_keeps_its_placement_and_holder(device):
    from jittor.compat.torch.frontend import tensor_frontend
    source = np.asarray([0., 1., -2.], dtype="float32")
    with jt.no_grad():
        value = torch.tensor(source, device=device)
        other = torch.tensor([0., 2., -2.], device=device)
        before = other.var_ptr
        with tensor_frontend(torch.Tensor, device="cpu"):
            result = value != other
        np.testing.assert_array_equal(result.cpu().numpy(), [False, True, False])
        assert other.var_ptr == before and other.device.type == device
