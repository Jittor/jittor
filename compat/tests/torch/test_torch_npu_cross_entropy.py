"""Unweighted hard-label NPU CE preserves device execution and public dtypes."""
from contextlib import nullcontext
import sys

import numpy as np
import pytest
import torch


_TOLERANCES = {
    "float32": (1.3e-6, 1e-5),
    "float16": (1e-3, 1e-5),
    "bfloat16": (1.6e-2, 1e-5),
}


def _resident(value):
    assert value.device.type == "npu"
    backend = sys.modules.get("jittor")
    if backend is not None:
        value.sync()
        assert tuple(backend.core.dispatch_context([value])) == ("acl", 0)


def _array(value):
    return value.detach().float().cpu().numpy().copy()


def _quantize(values, dtype_name):
    values = np.asarray(values, dtype=np.float32)
    if dtype_name == "float16":
        return values.astype(np.float16).astype(np.float32)
    if dtype_name == "bfloat16":
        bits = values.view(np.uint32)
        rounded = bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
        return np.asarray(rounded & np.uint32(0xffff0000), dtype=np.uint32).view(np.float32)
    return values


def _run_case(dtype_name, reduction, module, autocast=False):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires a real ACL/NPU device")
    dtype = getattr(torch, dtype_name)
    values = np.random.RandomState(183).normal(0, 1.25, (5, 11)).astype(np.float32)
    labels = np.array([0, 3, -100, 10, 7], dtype=np.int64)
    x = torch.tensor(values, device="npu", dtype=dtype, requires_grad=True)
    target = torch.tensor(labels, device="npu")
    # Reference mathematics uses the actual quantized input, independently of
    # the CANN decomposition and the Torch compatibility implementation.
    wide = _array(x).astype(np.float64)
    shifted = wide - wide.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    log_probs = shifted - np.log(exponential.sum(axis=1, keepdims=True))
    # Independent torch_npu probes establish a narrow log-softmax boundary,
    # including when autocast promotes the following NLL stage to FP32.
    if dtype_name != "float32":
        log_probs = _quantize(log_probs, dtype_name).astype(np.float64)
    valid = labels != -100
    safe = np.where(valid, labels, 0)
    losses = -log_probs[np.arange(5), safe]
    losses[~valid] = 0
    expected_grad = np.exp(log_probs)
    expected_grad[np.arange(5), safe] -= 1
    expected_grad[~valid] = 0
    expected = losses
    if reduction == "sum":
        expected = losses.sum()
    elif reduction == "mean":
        expected = losses.sum() / valid.sum()
        expected_grad /= valid.sum()

    if not autocast and dtype_name != "float32":
        expected = _quantize(expected, dtype_name)

    backend = sys.modules.get("jittor")
    before = backend.core.backend_fallback_count() if backend is not None else None
    region = torch.autocast("npu", dtype=dtype) if autocast else nullcontext()
    with region:
        if module:
            loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction=reduction)(x, target)
        else:
            loss = torch.nn.functional.cross_entropy(
                x, target, ignore_index=-100, reduction=reduction)
    grad, = torch.autograd.grad(loss.sum(), [x])
    _resident(loss)
    _resident(grad)
    assert loss.dtype == (torch.float32 if autocast else dtype)
    assert grad.dtype == dtype
    if backend is not None:
        assert backend.core.backend_fallback_count() == before
    output_tolerance = _TOLERANCES["float32" if autocast else dtype_name]
    np.testing.assert_allclose(_array(loss), expected, rtol=output_tolerance[0],
                               atol=output_tolerance[1], equal_nan=False)
    gradient_tolerance = _TOLERANCES[dtype_name]
    np.testing.assert_allclose(_array(grad), expected_grad, rtol=gradient_tolerance[0],
                               atol=gradient_tolerance[1], equal_nan=False)


@pytest.mark.npu
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("module", [False, True])
def test_npu_hard_label_ce_loss_and_gradient(dtype_name, reduction, module):
    _run_case(dtype_name, reduction, module)


@pytest.mark.npu
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("module", [False, True])
def test_npu_autocast_ce_promotes_loss_and_preserves_input_gradient(dtype_name, module):
    _run_case(dtype_name, "mean", module, autocast=True)
