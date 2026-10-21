"""``torch.npu`` answers about the ACL runtime, not with placeholders.

Every device namespace the CUDA installer creates used to share the same
stubs, so on a machine with eight 910Bs ``torch.npu.is_available()`` was
False and ``device_count()`` was 0. A library that asks before choosing a
device believed it and went to the host.

Both halves are asserted here: on a CPU-only process the facade still refuses
rather than inventing a device, and on ACL it selects, computes and reports
the device that was selected.
"""

import os

import pytest
import torch

import jittor as jt


def _on_acl():
    return bool(getattr(jt.compiler, "has_acl", 0)) and bool(jt.flags.use_cuda)


def test_npu_namespace_refuses_rather_than_inventing_a_device():
    if _on_acl():
        pytest.skip("this is the CPU half; the ACL half is the next case")
    assert torch.npu.is_available() is False
    assert torch.npu.device_count() == 0
    for operation in (torch.npu.current_device,
                      lambda: torch.npu.set_device(0),
                      torch.npu.synchronize,
                      torch.npu.empty_cache):
        with pytest.raises(RuntimeError, match="ACL"):
            operation()


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_discovery_selection_and_synchronization():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    count = torch.npu.device_count()
    assert torch.npu.is_available() and count > 0
    assert torch.npu.is_bf16_supported() is True

    original = torch.npu.current_device()
    try:
        # Every spelling of a device torch accepts, including the bare type
        # and None, which both mean "the current one".
        for argument in (0, "npu:%d" % (count - 1), torch.device("npu:0"),
                         None, torch.device("npu")):
            torch.npu.set_device(argument)
            selected = torch.npu.current_device()
            x = torch.tensor([selected + 1.0, selected + 2.0],
                             dtype=torch.float32, device="npu")
            y = x.square() + 3
            torch.npu.synchronize(torch.device("npu", selected))
            assert torch.npu.current_device() == selected
            assert x.device.type == y.device.type == "npu"
            assert y.cpu().tolist() == [(selected + 1.0) ** 2 + 3,
                                        (selected + 2.0) ** 2 + 3]
    finally:
        torch.npu.set_device(original)

    # An ordinal past the end is a mistake, not a device.
    for operation in (torch.npu.set_device, torch.npu.synchronize):
        with pytest.raises(ValueError, match="out of range"):
            operation(count)
    with pytest.raises(TypeError, match="bool"):
        torch.npu.set_device(True)
    assert torch.npu.current_device() == original


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_device_scope_restores_the_previous_device():
    original = torch.npu.current_device()
    other = (original + 1) % torch.npu.device_count()
    with torch.npu.device(other):
        assert torch.npu.current_device() == other
    assert torch.npu.current_device() == original


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_torch_npu_primitives_match_their_definitions():
    import torch_npu

    assert torch_npu.__jittor_acl_facade__ is True
    assert torch_npu.npu is torch.npu

    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0]], device="npu")
    weight = torch.tensor([0.5, 1.5, 2.5, 3.5], device="npu")
    normalized, auxiliary = torch_npu.npu_rms_norm(hidden, weight, 1e-6)
    assert auxiliary is None
    reference = hidden * torch.rsqrt((hidden * hidden).mean(-1, keepdim=True) + 1e-6) * weight
    assert torch.allclose(normalized, reference, atol=1e-5)

    cos = torch.full((1, 4), 0.6, device="npu")
    sin = torch.full((1, 4), 0.8, device="npu")
    rotated = torch_npu.npu_rotary_mul(hidden, cos, sin)
    half = hidden.shape[-1] // 2
    expected = hidden * cos + torch.cat((-hidden[..., half:], hidden[..., :half]), -1) * sin
    assert torch.allclose(rotated, expected, atol=1e-5)

    paired = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="npu")
    gate, value = paired.chunk(2, dim=-1)
    assert torch.allclose(torch_npu.npu_swiglu(paired),
                          gate * torch.sigmoid(gate) * value, atol=1e-5)

    # Fused attention is refused, not quietly replaced with an unfused stand-in.
    with pytest.raises(NotImplementedError):
        torch_npu.npu_fusion_attention()


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_unsupported_npu_apis_refuse_instead_of_reporting_zero():
    for name in ("memory_allocated", "max_memory_allocated", "memory_stats",
                 "get_device_properties", "current_stream", "mem_get_info"):
        with pytest.raises(NotImplementedError):
            getattr(torch.npu, name)()
