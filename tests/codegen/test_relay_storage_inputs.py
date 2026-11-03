"""Dense-kernel relay candidates must stay inside their fused execution plan."""

import numpy as np
import pytest

import jittor as jt
from _helpers.native_runtime import native_execution_scope
from _helpers.onednn import requires_onednn


@pytest.fixture(autouse=True)
def native_cpu():
    with jt.flag_scope(use_cuda=0, amp_reg=0, enable_tuner=1), native_execution_scope(jt):
        requires_onednn()
        yield


@pytest.mark.parametrize('strided', [False, True])
def test_depthwise_conv_relay_keeps_original_input_and_weight_gradients(strided):
    random = np.random.RandomState(71)
    source = random.randn(1, 8, 64).astype('float32')
    values = source.transpose(0, 2, 1)
    weights = random.randn(64, 1, 3).astype('float32')
    seed = random.randn(1, 64, 8).astype('float32')
    root = jt.array(source if strided else values.copy())
    x = root._storage_permute((0, 2, 1)) if strided else root
    w = jt.array(weights)
    # Freeze the inputs as real allocations so constructor normalization must
    # handle the physical strides, rather than a pending transpose expression.
    jt.sync([x, w])
    assert x._storage_is_contiguous() != strided
    output = jt.nn.conv1d(x, w, padding=1, groups=64)
    dx, dw = jt.grad((output * jt.array(seed)).sum(), [root, w])

    expected = np.zeros_like(seed)
    expected_dx = np.zeros_like(values)
    expected_dw = np.zeros_like(weights)
    for position in range(8):
        for kernel in range(3):
            index = position + kernel - 1
            if 0 <= index < 8:
                expected[:, :, position] += values[:, :, index] * weights[:, 0, kernel]
                expected_dx[:, :, index] += seed[:, :, position] * weights[:, 0, kernel]
                expected_dw[:, 0, kernel] += (seed[:, :, position] * values[:, :, index]).sum(0)
    np.testing.assert_allclose(output.numpy(), expected, atol=1e-5, rtol=1e-5)
    root_gradient = expected_dx.transpose(0, 2, 1) if strided else expected_dx
    np.testing.assert_allclose(dx.numpy(), root_gradient, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(dw.numpy(), expected_dw, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('strided', [False, True])
def test_matmul_relay_rejection_never_publishes_negative_candidate(strided):
    random = np.random.RandomState(79)
    values = random.randn(16, 24).astype('float32')
    right = random.randn(24, 12).astype('float32')
    seed = random.randn(16, 12).astype('float32')
    root = jt.array(values.T.copy() if strided else values)
    a = root._storage_permute((1, 0)) if strided else root
    b = jt.array(right)
    jt.sync([a, b])
    with jt.log_capture_scope(log_v=0, log_vprefix='tuner_manager=1000') as logs:
        product = (a.broadcast((16, 24, 12), (2,)) * b.broadcast((16, 24, 12), (0,))).sum(1)
        da, db = jt.grad((product * jt.array(seed)).sum(), [root, b])
        np.testing.assert_allclose(product.numpy(), values @ right, atol=1e-5, rtol=1e-5)
        expected_da = seed @ right.T
        np.testing.assert_allclose(da.numpy(), expected_da.T if strided else expected_da,
                                   atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(db.numpy(), values.T @ seed, atol=1e-5, rtol=1e-5)
    assert not any('relay-1' in str(record) for record in logs)


def test_dense_mkl_matmul_provider_remains_executable():
    ops = requires_onednn()
    a = jt.array(np.arange(6, dtype='float32').reshape(2, 3))
    b = jt.array(np.arange(12, dtype='float32').reshape(3, 4))
    with jt.profile_scope() as report:
        result = ops.mkl_matmul(a, b, False, False)
        np.testing.assert_array_equal(result.numpy(), a.numpy() @ b.numpy())
    assert any('mkl_matmul' in row[0] for row in report[1:]), report
