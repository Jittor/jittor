"""Boolean basic assignment retains shape, broadcasting, snapshots and views."""
import sys
import numpy as np
import pytest
import torch

CASES = ("prefix", "none_bounds", "row_broadcast", "column_broadcast",
         "snapshot_overlap", "view_alias", "step", "newaxis", "numeric_rhs")


def resident(value, device):
    assert value.dtype == torch.bool and value.device.type == device
    jt = sys.modules.get("jittor")
    if jt is not None:
        value.sync()
        assert value.location() == ("device" if device == "npu" else "cpu")
        assert tuple(jt.core.dispatch_context([value])) == (("acl", 0) if device == "npu" else ("cpu", -1))


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
@pytest.mark.parametrize("case", CASES)
def test_bool_slice_assignment_exact_and_alias(device, case):
    if device == "npu" and not (getattr(torch, "npu", None) and torch.npu.is_available()):
        pytest.skip("NPU is unavailable in this runtime")
    data = (np.arange(14).reshape(2, 7) % 3 == 0)
    expected = data.copy()
    value = torch.tensor(data, dtype=torch.bool, device=device)
    alias = value
    extras = []
    if case == "prefix":
        value[..., :1] = False
        expected[..., :1] = False
    elif case == "none_bounds":
        value[:, slice(None, None, None)] = True
        expected[:, slice(None, None, None)] = True
    elif case == "row_broadcast":
        source = np.array([True, False, True, False])
        value[:, 1:5] = torch.tensor(source, dtype=torch.bool, device=device)
        expected[:, 1:5] = source
    elif case == "column_broadcast":
        source = np.array([[False], [True]])
        value[:, 1:4] = torch.tensor(source, dtype=torch.bool, device=device)
        expected[:, 1:4] = source
    elif case == "snapshot_overlap":
        source = value[:, :-1].clone()
        value[:, 1:] = source
        expected[:, 1:] = data[:, :-1].copy()
        extras.append((source, data[:, :-1]))
    elif case == "view_alias":
        view = value[:, 1:6]
        view[:, :2] = False
        expected[:, 1:3] = False
        extras.append((view, expected[:, 1:6]))
    elif case == "step":
        source = np.array([[True], [False]])
        value[:, ::2] = torch.tensor(source, dtype=torch.bool, device=device)
        expected[:, ::2] = source
    elif case == "newaxis":
        value[None, :, 2:4] = True
        expected[None, :, 2:4] = True
    elif case == "numeric_rhs":
        source = np.array([0., .5, -.25, 256.], dtype=np.float32)
        value[:, 1:5] = torch.tensor(source, dtype=torch.float32, device=device)
        expected[:, 1:5] = source
    assert alias is value and tuple(value.shape) == data.shape
    for tensor, reference in [(value, expected), (alias, expected), *extras]:
        resident(tensor, device)
        np.testing.assert_array_equal(tensor.detach().cpu().numpy(), reference)
