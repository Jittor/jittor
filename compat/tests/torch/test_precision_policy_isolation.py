"""Native/Torch policy isolation, delayed execution, and an independent RNN oracle."""
from contextlib import contextmanager

import numpy as np
import pytest


@contextmanager
def _policies(cuda=False):
    import jittor as jt
    import torch
    from jittor.compat.torch.installers.cuda.api import _cuda_runtime
    state = _cuda_runtime()
    saved = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
             state.matmul_refinement)
    if cuda:
        inventory = jt.introspection.capabilities.devices("cuda")
        if not inventory.capability.enabled or not inventory.count:
            pytest.skip("accelerator prerequisite: real CUDA required")
    options = dict(float32_matmul_precision="highest", use_tensorcore=0,
                   cuda_allow_tf32=0, cuda_allow_cudnn_tf32=0, auto_flush_ops=0)
    if cuda:
        options["use_cuda"] = 1
    with jt.runtime.scope(**options):
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(saved[0])
            torch.backends.cudnn.allow_tf32 = saved[1]
            state.matmul_refinement = saved[2]


def test_native_combined_setter_and_frontend_domains_restore_independently():
    import jittor as jt
    import torch
    from jittor.compat.torch.frontend import tensor_frontend
    with _policies():
        assert tuple(jt.core.float32_precision_state()) == ("highest", "highest")
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = False
        assert tuple(jt.core.float32_precision_state()) == ("highest", "highest")
        with jt.runtime.scope(float32_matmul_precision="high"):
            assert tuple(jt.core.float32_precision_state()) == ("high", "high")
            assert torch.get_float32_matmul_precision() == "medium"
            assert not torch.backends.cudnn.allow_tf32
            with pytest.raises(RuntimeError, match="scope probe"):
                with tensor_frontend(torch.Tensor):
                    assert tuple(jt.core.float32_precision_state()) == ("medium", "highest")
                    raise RuntimeError("scope probe")
            assert tuple(jt.core.float32_precision_state()) == ("high", "high")
        assert tuple(jt.core.float32_precision_state()) == ("highest", "highest")
        torch.backends.cudnn.allow_tf32 = True
        assert torch.get_float32_matmul_precision() == "medium"
        torch.set_float32_matmul_precision("highest")
        assert torch.backends.cudnn.allow_tf32


def _conv_reference(x, weight):
    out = np.empty((x.shape[0], weight.shape[0], x.shape[2]-weight.shape[2]+1,
                    x.shape[3]-weight.shape[3]+1), dtype=np.float64)
    for y in range(out.shape[2]):
        for z in range(out.shape[3]):
            out[:, :, y, z] = np.einsum("nchw,ochw->no", x[:, :, y:y+weight.shape[2], z:z+weight.shape[3]], weight)
    return out


def test_pending_matmul_conv_keep_their_frontend_policy_after_switches():
    import jittor as jt
    import torch
    from _helpers.logs import find_log_with_re
    rng = np.random.RandomState(7319)
    a, b = (rng.randn(32, 32).astype(np.float32) for _ in range(2))
    x = rng.randn(1, 2, 5, 5).astype(np.float32)
    w = rng.randn(3, 2, 3, 3).astype(np.float32)
    with _policies(cuda=True):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        strict_mm = torch.matmul(torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda"))
        strict_conv = torch.nn.functional.conv2d(torch.tensor(x, device="cuda"), torch.tensor(w, device="cuda"))
        # Both graphs remain pending while the creating frontend and native
        # default change. Their library calls must still use the old policy.
        assert strict_mm.location() == strict_conv.location() == "none"
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        medium_mm = torch.matmul(torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda"))
        with jt.runtime.scope(float32_matmul_precision="high"):
            native_mm = jt.matmul(jt.array(a), jt.array(b))
            native_conv = jt.nn.conv2d(jt.array(x), jt.array(w))
            with jt.log_capture_scope(log_silent=1, log_v=0,
                    log_vprefix="cublas_matmul=100,cudnn_conv=100") as logs:
                arrays = jt.fetch_sync([strict_mm, strict_conv, native_mm, native_conv, medium_mm])
        gemm = find_log_with_re(logs, r"algo select: precision=(\S+) computeType=(\S+) algo=(\S+)")
        conv = find_log_with_re(logs, r"precision select: precision=(\S+) computeType=(\S+) mathType=(\S+)")
        assert {item[0] for item in gemm} == {"highest", "high", "medium"}, gemm
        assert {item[2] for item in conv} == {"CUDNN_FMA_MATH", "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION"}, conv
        np.testing.assert_allclose(arrays[0], a.astype(np.float64) @ b.astype(np.float64), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(arrays[1], _conv_reference(x.astype(np.float64), w.astype(np.float64)), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(arrays[2], a @ b, rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(arrays[3], _conv_reference(x, w), rtol=5e-3, atol=5e-3)
        assert np.max(np.abs(arrays[4] - a @ b)) / max(1., np.max(np.abs(a @ b))) < 1e-2
        assert torch.get_float32_matmul_precision() == "medium"
        assert torch.backends.cudnn.allow_tf32


def _lstm_oracle(x, weights, coefficient):
    x, coefficient = x.astype(np.float64), coefficient.astype(np.float64)
    wi, wh, bi, bh = [item.astype(np.float64) for item in weights]
    hidden = wh.shape[1]
    h = np.zeros((x.shape[1], hidden))
    c = np.zeros_like(h)
    cache, output = [], []
    for value in x:
        pre = value @ wi.T + h @ wh.T + bi + bh
        ai, af, ag, ao = np.split(pre, 4, axis=1)
        i, f, g, o = 1/(1+np.exp(-ai)), 1/(1+np.exp(-af)), np.tanh(ag), 1/(1+np.exp(-ao))
        old_h, old_c = h, c
        c = f*c + i*g
        h = o*np.tanh(c)
        cache.append((old_h, old_c, i, f, g, o, c))
        output.append(h)
    dwi, dwh, db = np.zeros_like(wi), np.zeros_like(wh), np.zeros_like(bi)
    dx, dh, dc = np.zeros_like(x), np.zeros_like(h), np.zeros_like(c)
    for index in reversed(range(len(x))):
        old_h, old_c, i, f, g, o, c = cache[index]
        dh = dh + coefficient[index]
        tc = np.tanh(c)
        dc = dc + dh*o*(1-tc*tc)
        da = np.concatenate((dc*g*i*(1-i), dc*old_c*f*(1-f),
                             dc*i*(1-g*g), dh*tc*o*(1-o)), axis=1)
        dwi += da.T @ x[index]
        dwh += da.T @ old_h
        db += da.sum(axis=0)
        dx[index] = da @ wi
        dh, dc = da @ wh, dc*f
    return np.stack(output), (dx, dwi, dwh, db, db.copy())


def test_lstm_cpu_oracle_backward_has_a_finite_difference_control():
    rng = np.random.RandomState(720)
    x = rng.randn(2, 1, 2).astype(np.float64)
    weights = [rng.randn(8, 2)*0.1, rng.randn(8, 2)*0.1,
               rng.randn(8)*0.1, rng.randn(8)*0.1]
    coefficient = rng.randn(2, 1, 2)
    _, grads = _lstm_oracle(x, weights, coefficient)
    for variable, gradient in zip([x, *weights], grads):
        original = variable.flat[1]
        variable.flat[1] = original + 1e-5
        positive = np.sum(_lstm_oracle(x, weights, coefficient)[0]*coefficient)
        variable.flat[1] = original - 1e-5
        negative = np.sum(_lstm_oracle(x, weights, coefficient)[0]*coefficient)
        variable.flat[1] = original
        np.testing.assert_allclose(gradient.flat[1], (positive-negative)/2e-5, rtol=1e-7, atol=1e-9)


def _run_lstm(frontend, x, weights, coefficient, is_torch, backward_tf32=None):
    import jittor as jt
    from _helpers.logs import find_log_with_re
    model = frontend.nn.LSTM(x.shape[-1], weights[1].shape[-1], num_layers=1)
    names = ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0")
    creator = frontend.tensor if is_torch else jt.array
    for name, value in zip(names, weights):
        getattr(model, name).assign(creator(value, dtype="float32"))
    value = creator(x, dtype="float32")
    value.start_grad()
    coefficient = creator(coefficient, dtype="float32")
    with jt.log_capture_scope(log_silent=1, log_v=0, log_vprefix="cudnn_rnn_descriptor=100") as logs:
        output, _ = model(value)
        loss = (output * coefficient).sum()
        targets = [value] + [getattr(model, name) for name in names]
        if backward_tf32 is not None:
            frontend.backends.cudnn.allow_tf32 = backward_tf32
        grads = frontend.autograd.grad(loss, targets) if is_torch else jt.grad(loss, targets)
        arrays = jt.fetch_sync([output, *grads])
    selection = find_log_with_re(logs, r"rnn precision select: precision=(\S+) mathType=(\S+)")
    assert selection, "the RNN must execute actual cuDNN descriptors"
    return arrays, selection


def test_fp32_rnn_native_default_and_torch_cudnn_policy_match_cpu_recurrence():
    import jittor as jt
    import torch
    rng = np.random.RandomState(72019)
    x = rng.randn(5, 4, 32).astype(np.float32)
    weights = [(rng.randn(256, 32)*0.12).astype(np.float32),
               (rng.randn(256, 64)*0.12).astype(np.float32),
               (rng.randn(256)*0.03).astype(np.float32),
               (rng.randn(256)*0.03).astype(np.float32)]
    coefficient = rng.randn(5, 4, 64).astype(np.float32)
    expected_output, expected_grads = _lstm_oracle(x, weights, coefficient)
    reference = [expected_output, *expected_grads]
    with _policies(cuda=True):
        assert torch.backends.cudnn.allow_tf32 is True
        native, native_selection = _run_lstm(jt, x, weights, coefficient, False)
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = False
        strict, strict_selection = _run_lstm(torch, x, weights, coefficient, True)
        torch.backends.cudnn.allow_tf32 = True
        fast, fast_selection = _run_lstm(torch, x, weights, coefficient, True)
        assert all(item == ("highest", "CUDNN_FMA_MATH") for item in native_selection + strict_selection)
        assert all(item == ("high", "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION") for item in fast_selection)
        errors = []
        for actuals in (native, strict, fast):
            errors.append(max(float(np.max(np.abs(actual-wanted)) / max(1., np.max(np.abs(wanted))))
                              for actual, wanted in zip(actuals, reference)))
        assert errors[0] < 1e-6 and errors[1] < 1e-6, errors
        assert errors[2] < 5e-3, errors
        print("RNN relative max errors: native_highest=%g torch_ieee=%g torch_tf32=%g" % tuple(errors))


def test_grouped_rnn_backward_retains_forward_precision_after_policy_switch():
    import torch
    rng = np.random.RandomState(71920)
    x = rng.randn(5, 4, 32).astype(np.float32)
    weights = [(rng.randn(256, 32)*0.12).astype(np.float32),
               (rng.randn(256, 64)*0.12).astype(np.float32),
               (rng.randn(256)*0.03).astype(np.float32),
               (rng.randn(256)*0.03).astype(np.float32)]
    coefficient = rng.randn(5, 4, 64).astype(np.float32)
    output, gradients = _lstm_oracle(x, weights, coefficient)
    with _policies(cuda=True):
        torch.backends.cudnn.allow_tf32 = False
        # CudnnRnnOp implements grouped grads(), not the single-input grad().
        # Change policy after constructing the forward, before this entry runs.
        actual, selection = _run_lstm(torch, x, weights, coefficient, True,
                                      backward_tf32=True)
        assert all(item == ("highest", "CUDNN_FMA_MATH") for item in selection), selection
        error = max(float(np.max(np.abs(got-want)) / max(1., np.max(np.abs(want))))
                    for got, want in zip(actual, [output, *gradients]))
        assert error < 1e-6, error
        assert torch.backends.cudnn.allow_tf32 is True
