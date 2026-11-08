"""Autocast chains keep FP32 master weights differentiable on Ascend."""
import sys
import numpy as np
import pytest
import torch


def _fixed(shape, offset):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return ((values + offset) % 17 - 8) / 32


def _resident(value):
    assert value.device.type == "npu"
    jt = sys.modules.get("jittor")
    if jt is not None:
        value.sync()
        assert value.location() == "device"
        assert tuple(jt.core.dispatch_context([value])) == ("acl", 0)


@pytest.mark.npu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("input_low", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_npu_autocast_linear_chain_fp32_master_full_grad(dtype_name, input_low, lora, cache_enabled):
    dtype = getattr(torch, dtype_name)
    input_dtype = dtype if input_low else torch.float32
    first = torch.nn.Linear(8, 4, bias=False, device="npu", dtype=torch.float32)
    second = torch.nn.Linear(4, 6, bias=False, device="npu", dtype=torch.float32)
    with torch.no_grad():
        first.weight.copy_(torch.tensor(_fixed((4, 8), 1), device="npu"))
        second.weight.copy_(torch.tensor(_fixed((6, 4), 3), device="npu"))
    x = torch.tensor(_fixed((2, 3, 8), 7), device="npu", dtype=input_dtype,
                     requires_grad=True)
    base = torch.tensor(_fixed((6, 8), 5), device="npu", dtype=dtype)
    cotangent = torch.tensor(_fixed((2, 3, 6), 9), device="npu")
    reference_x = x.detach().clone().requires_grad_(True)
    reference_a = first.weight.detach().clone().requires_grad_(True)
    reference_b = second.weight.detach().clone().requires_grad_(True)
    with torch.autocast("npu", dtype=dtype, cache_enabled=cache_enabled):
        hidden = first(x)
        actual = second(hidden)
        if lora:
            actual = torch.nn.functional.linear(x, base) + actual * .5
    assert hidden.dtype == dtype and actual.dtype == dtype
    actual_grads = torch.autograd.grad((actual.float() * cotangent).sum(),
                                       [x, first.weight, second.weight])
    with torch.autocast("npu", enabled=False):
        reference = torch.nn.functional.linear(reference_x.to(dtype), reference_a.to(dtype))
        reference = torch.nn.functional.linear(reference, reference_b.to(dtype))
        if lora:
            reference = torch.nn.functional.linear(reference_x.to(dtype), base) + reference * .5
    reference_grads = torch.autograd.grad((reference.float() * cotangent).sum(),
                                          [reference_x, reference_a, reference_b])
    if lora and not input_low and cache_enabled:
        # Autocast backward accumulates the two FP32-leaf branch contributions
        # before its single compute-dtype rounding. Explicit forward casts round
        # each branch independently, so they are not a valid input-grad oracle.
        with torch.autocast("npu", enabled=False), torch.no_grad():
            low_a = first.weight.detach().to(dtype).float()
            low_b = second.weight.detach().to(dtype).float()
            low_base = base.detach().float()
            grad_out = cotangent.to(dtype).float().reshape(-1, 6)
            base_dx = grad_out @ low_base
            chain_dh = ((grad_out * .5).to(dtype).float()) @ low_b
            chain_dh = chain_dh.to(dtype).float()
            chain_dx = chain_dh @ low_a
            expected_x = (base_dx + chain_dx).to(dtype).to(input_dtype).reshape_as(x)
        reference_grads = (expected_x, *reference_grads[1:])
    pairs = [(actual, reference), *zip(actual_grads, reference_grads)]
    for value in [first.weight, second.weight, *[v for pair in pairs for v in pair]]:
        _resident(value)
    assert first.weight.dtype == second.weight.dtype == torch.float32
    assert actual_grads[0].dtype == input_dtype
    assert actual_grads[1].dtype == actual_grads[2].dtype == torch.float32
    for value, expected in pairs:
        assert value.dtype == expected.dtype and tuple(value.shape) == tuple(expected.shape)
        np.testing.assert_array_equal(value.detach().float().cpu().numpy(),
                                      expected.detach().float().cpu().numpy())
    jt = sys.modules.get("jittor")
    if jt is not None:
        assert jt.core.backend_fallback_count() == 0


@pytest.mark.npu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_npu_raw_mapped_matmul_retains_mixed_dtype_rejection(dtype_name):
    jt = sys.modules.get("jittor")
    if jt is None:
        pytest.skip("Native torch_npu has no Jittor raw mapped_matmul API")
    left = torch.ones((2, 8), dtype=getattr(torch, dtype_name), device="npu")
    weight = torch.ones((6, 8), dtype=torch.float32, device="npu")
    with jt.flag_scope(amp_reg=0):
        with pytest.raises(RuntimeError, match="same dtype"):
            jt.core.ops.mapped_matmul(left, weight, False, True, False)


@pytest.mark.npu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_npu_autocast_clear_cache_refreshes_updated_master_weight(dtype_name):
    dtype = getattr(torch, dtype_name)
    module = torch.nn.Linear(4, 3, bias=False, device="npu", dtype=torch.float32)
    x = torch.ones((2, 4), device="npu", requires_grad=True)
    with torch.no_grad():
        module.weight.fill_(.125)
    with torch.autocast("npu", dtype=dtype, cache_enabled=True):
        first = module(x)
        _resident(first)
        with torch.no_grad():
            module.weight.fill_(.25)
        torch.clear_autocast_cache()
        second = module(x)
        _resident(second)
    assert first.dtype == second.dtype == dtype
    np.testing.assert_array_equal(first.detach().float().cpu().numpy(), np.full((2, 3), .5))
    np.testing.assert_array_equal(second.detach().float().cpu().numpy(), np.ones((2, 3)))
    dx, dw = torch.autograd.grad(second.float().sum(), [x, module.weight])
    _resident(dx)
    _resident(dw)
    assert dx.dtype == dw.dtype == torch.float32
    np.testing.assert_array_equal(dx.detach().cpu().numpy(), np.full((2, 4), .75))
    np.testing.assert_array_equal(dw.detach().cpu().numpy(), np.full((3, 4), 2.))
