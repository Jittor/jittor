"""FSDP FULL restore uses public placement and keeps reference dtype policy."""

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "_torch_compat_install_context"),
    reason="Jittor internal FSDP checkpoint placement owner",
)


@pytest.mark.parametrize("preserve_dtype", [False, True])
@pytest.mark.parametrize("backend,index,expected", [
    ("cpu", 0, "cpu"), ("cuda", 2, "cuda:2"),
    ("acl", 0, "npu:0"), ("acl", 3, "npu:3"),
])
def test_checkpoint_factory_public_device_routing(
        monkeypatch, preserve_dtype, backend, index, expected):
    # Routing only: the factory returns pre-existing real CPU storage. Real
    # accelerator allocation and fresh restore require the on-device gate.
    import jittor as jt
    from jittor.compat.fsdp2 import _state_dict as owner
    from jittor.compat.torch.context import get_install_context

    namespace = get_install_context(jt).target_namespace
    value = torch.tensor([1.25, -.5], dtype=torch.float64, device="cpu")
    reference = torch.tensor([0., 0.], dtype=torch.float32, device="cpu")
    wanted = value.dtype if preserve_dtype else reference.dtype
    sentinel = torch.tensor([1.25, -.5], dtype=wanted, device="cpu", requires_grad=True)
    observed = []

    def factory(data, *, dtype, device):
        observed.append((data, dtype, device))
        assert data is value and dtype == wanted
        assert device == expected
        return sentinel

    with monkeypatch.context() as patch:
        patch.setattr(jt.core, "dispatch_context", lambda inputs: (backend, index))
        patch.setattr(namespace, "tensor", factory)
        restored = owner._checkpoint_tensor(
            value, reference, preserve_dtype=preserve_dtype)
    assert len(observed) == 1
    assert restored.dtype == wanted and not restored.requires_grad
    np.testing.assert_array_equal(restored.detach().numpy(), [1.25, -.5])


@pytest.mark.parametrize("preserve_dtype", [False, True])
def test_checkpoint_tensor_real_cpu_values_dtype_and_detach(preserve_dtype):
    from jittor.compat.fsdp2 import _state_dict as owner

    value = torch.tensor([1.25, -.5], dtype=torch.float64, device="cpu", requires_grad=True)
    reference = torch.tensor([0., 0.], dtype=torch.float32, device="cpu")
    restored = owner._checkpoint_tensor(
        value, reference, preserve_dtype=preserve_dtype)
    assert isinstance(restored, torch.Tensor)
    assert restored.device.type == "cpu" and not restored.requires_grad
    assert restored.dtype == (value.dtype if preserve_dtype else reference.dtype)
    np.testing.assert_array_equal(restored.detach().numpy(), [1.25, -.5])
