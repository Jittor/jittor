"""ROCm training regressions for AMP and multi-tensor AdamW."""

import numpy as np
import pytest
import torch
import jittor as jt
from _helpers.rocm import native_rocm
from jittor._runtime.dispatch import override_kernel, registered_kernel

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason="ROCm build required")


def test_rocm_default_adamw_uses_fused_provider_and_matches_opt_out():
    from jittor.backends.rocm.kernels import optim
    raw = np.random.RandomState(196).normal(size=(17, 23)).astype("float32")
    seed = torch.tensor(np.random.RandomState(197).normal(size=raw.shape).astype("float32"),
                        device="cuda")

    def train(fused=None):
        parameter = torch.nn.Parameter(torch.tensor(raw, device="cuda"))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=.01, fused=fused)
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            (parameter.square() * seed).sum().backward()
            optimizer.step()
        parameter.sync()
        assert parameter.location() == "device"
        return parameter.detach().cpu().numpy().copy()

    calls = []
    original = registered_kernel("optim.adamw_fused", "rocm")
    assert original is not None
    def traced(*args):
        calls.append(True)
        return original(*args)
    with override_kernel("optim.adamw_fused", "rocm", traced, supports=optim._supports_adamw):
        reference = train(False)
        assert calls == []
        actual = train()
    assert len(calls) == 4
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_amp_conv2d_forward_backward(dtype):
    raw = np.arange(2*3*7*8, dtype="float32").reshape(2,3,7,8) % 7 / 8
    weight = np.arange(4*3*3*2, dtype="float32").reshape(4,3,3,2) % 5 / 8
    x = torch.tensor(raw, device="cuda", requires_grad=True)
    w = torch.tensor(weight, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        y = torch.nn.functional.conv2d(x, w, stride=(2,1), padding=(1,0))
    y.sync()
    assert y.location() == "device" and y.dtype == dtype
    y.float().sum().backward()
    for value in (x.grad, w.grad):
        value.sync()
        assert value.location() == "device" and value.dtype == torch.float32
        assert np.isfinite(value.cpu().numpy()).all()
    with torch.no_grad():
        reference = torch.nn.functional.conv2d(x.float(), w.float(), stride=(2,1), padding=(1,0))
        np.testing.assert_allclose(y.float().cpu().numpy(), reference.cpu().numpy(),
                                   rtol=0.008, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_amp_layer_norm_and_cross_entropy_use_fp32(dtype):
    raw = np.random.RandomState(198).normal(size=(4, 17)).astype("float32")
    labels = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.long)
    x = torch.tensor(raw, device="cuda", requires_grad=True)
    weight = torch.ones(17, device="cuda", requires_grad=True)
    bias = torch.zeros(17, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=dtype):
        y = torch.nn.functional.layer_norm(x, (17,), weight, bias)
        loss = torch.nn.functional.cross_entropy(y, labels)
    assert y.dtype == torch.float32 and loss.dtype == torch.float32
    loss.backward()
    grad = x.grad.detach().cpu().numpy().copy()
    z = torch.tensor(raw, device="cuda", requires_grad=True)
    reference = torch.nn.functional.cross_entropy(
        torch.nn.functional.layer_norm(z, (17,), weight.detach(), bias.detach()), labels)
    reference.backward()
    np.testing.assert_allclose(loss.detach().cpu().numpy(), reference.detach().cpu().numpy(),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad, z.grad.cpu().numpy(), rtol=2e-5, atol=2e-6)


def test_rocm_bfloat16_unary_math_on_torch_frontend():
    raw = np.array([.25, .5, 1., 2., 4.], dtype="float32")
    x = torch.tensor(raw, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for op, reference in ((torch.log, np.log), (torch.sqrt, np.sqrt), (torch.exp, np.exp)):
            y = op(x)
            y.sync()
            assert y.location() == "device" and y.dtype == torch.bfloat16
            np.testing.assert_allclose(y.float().cpu().numpy(), reference(raw), rtol=0.008, atol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_amp_batchnorm_preserves_activation_dtype(dtype):
    rng = np.random.RandomState(201)
    x = torch.tensor(rng.normal(size=(2, 4, 5, 7)).astype("float32"),
                     device="cuda", dtype=dtype, requires_grad=True)
    w = torch.ones(4, device="cuda", requires_grad=True)
    b = torch.zeros(4, device="cuda", requires_grad=True)
    mean = torch.zeros(4, device="cuda")
    var = torch.ones(4, device="cuda")
    with torch.autocast(device_type="cuda", dtype=dtype):
        y = torch.nn.functional.batch_norm(x, mean, var, w, b, training=True)
    y.sync()
    assert y.location() == "device" and y.dtype == dtype
    raw = x.float().detach().cpu().numpy()
    expected = (raw-raw.mean((0,2,3),keepdims=True))/np.sqrt(raw.var((0,2,3),keepdims=True)+1e-5)
    np.testing.assert_allclose(y.float().cpu().numpy(), expected, rtol=0.008, atol=0.002)
    y.float().sum().backward()
    assert x.grad is not None
    x.grad.sync()
    assert x.grad.location() == "device"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rocm_low_precision_maxpool_backward(dtype):
    raw = np.arange(2*3*8*8,dtype="float32").reshape(2,3,8,8) % 19
    x = torch.tensor(raw, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.nn.functional.max_pool2d(x, 2, stride=2)
    y.sync()
    assert y.location() == "device" and y.dtype == dtype
    expected = raw.reshape(2,3,4,2,4,2).max((3,5))
    np.testing.assert_array_equal(y.float().cpu().numpy(), expected)
    y.float().sum().backward()
    x.grad.sync()
    assert x.grad.location() == "device"
    np.testing.assert_array_equal(x.grad.float().cpu().numpy().sum((2,3)),
                                  np.full((2,3),16))


def test_rocm_bf16_mixed_ternary_backward():
    x = torch.tensor(np.linspace(-1,1,128,dtype="float32"),device="cuda",requires_grad=True)
    mask = x > 0
    with torch.autocast("cuda",dtype=torch.bfloat16):
        narrow = x.to(torch.bfloat16)
        y = torch.where(mask, narrow, x.float())
        loss = (y.float().square()).sum()
    loss.backward()
    x.grad.sync()
    assert x.grad.location() == "device"
    assert np.isfinite(x.grad.cpu().numpy()).all()
