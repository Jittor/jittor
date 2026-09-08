"""The native Python matrix/conv entry points execute their registered selections."""

from _helpers import capability as _test_capability

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("use_cuda", [0, 1])
def test_batched_matmul_calls_selected_library_and_filters_dtype(monkeypatch, use_cuda):
    import jittor as jt
    from jittor.nn.functional import matrix

    if use_cuda and not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    calls = []
    lookups = []
    result = object()

    def matmul_call(a, b, trans_a, trans_b):
        calls.append((tuple(a.shape), tuple(b.shape), trans_a, trans_b))
        return result

    def library(name, *, load=False):
        lookups.append((name, load))
        return SimpleNamespace(cublas_batched_matmul=matmul_call,
                               mkl_batched_matmul=matmul_call)

    monkeypatch.setattr(matrix, "get_library_ops", library)
    with jt.flag_scope(use_cuda=use_cuda):
        a = jt.array(np.zeros((2, 3, 4), dtype=np.float32))
        b = jt.array(np.zeros((2, 4, 5), dtype=np.float32))
        assert matrix.matmul(a, b) is result
        assert calls == [((2, 3, 4), (2, 4, 5), False, False)]
        assert {name for name, _ in lookups} == {"cublas" if use_cuda else "mkl"}
        if not use_cuda:
            assert ("mkl", True) in lookups
        lookups.clear()
        integer_a, integer_b = a.int32(), b.int32()
        assert matrix.select_kernel("batched_matmul", integer_a, integer_b, False, False) is None
        assert lookups == []


@pytest.mark.parametrize("use_cuda", [0, 1])
def test_conv2d_priority_and_explicit_depthwise_disable(monkeypatch, use_cuda):
    import jittor as jt
    from jittor._runtime.dispatch import select_kernel
    from jittor.nn.backends import cudnn
    from jittor.nn.modules import depthwise

    if use_cuda and not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA runtime required")
    monkeypatch.setattr(cudnn, "get_library_ops", lambda name: object())
    with jt.flag_scope(use_cuda=use_cuda):
        x = jt.array(np.zeros((1, 2, 5, 5), dtype=np.float32))
        weight = jt.array(np.zeros((2, 1, 3, 3), dtype=np.float32))
        args = (x, weight, None, (1, 1), (0, 0), (1, 1), 2)
        selected = select_kernel("conv2d", *args)
        disabled = select_kernel("conv2d", *args, _depthwise_fast_path=False)
        if use_cuda:
            assert selected is depthwise._depthwise_conv2d
            assert disabled is cudnn._try_cudnn_conv2d.__wrapped__
        else:
            assert selected is None and disabled is None


def test_projected_rnn_declines_library_without_loading_it(monkeypatch):
    import jittor as jt
    from jittor.nn.modules import recurrent_base

    loaded = []
    monkeypatch.setattr(recurrent_base, "get_library", lambda name: loaded.append(name))
    assert not recurrent_base._supports_cudnn_rnn(
        None, None, SimpleNamespace(proj_size=2), ())
    assert loaded == []


@pytest.mark.parametrize("provide_hidden", [False, True])
def test_rnn_internal_factories_follow_input_device(provide_hidden):
    import jittor as jt

    if not _test_capability.check_accelerator('cuda', backend=jt).enabled or _test_capability.device_count('cuda', backend=jt) < 2:
        pytest.skip("two CUDA devices are required")
    with jt.flag_scope(use_cuda=1, device_id=1, no_grad=1):
        model = jt.nn.RNN(3, 4, nonlinearity="tanh")
        model.eval()
        x = jt.array(np.arange(6, dtype=np.float32).reshape(2, 1, 3) / 8)
        hx = jt.zeros((1, 1, 4), dtype="float32") if provide_hidden else None
        output, hidden = model(x, hx)
        jt.sync_all(True)
        assert output.device_id == hidden.device_id == 1
        expected_output = output.numpy().copy()
        expected_hidden = hidden.numpy().copy()

        with jt.flag_scope(device_id=0):
            actual_output, actual_hidden = model(x, hx)
            jt.sync_all(True)
            assert jt.introspection.policy.runtime.device_id == 0
            for value in (actual_output, actual_hidden):
                assert value.device_id == 1
                assert value.location() == "device"
            np.testing.assert_allclose(actual_output.numpy(), expected_output,
                                       rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(actual_hidden.numpy(), expected_hidden,
                                       rtol=1e-5, atol=1e-6)
