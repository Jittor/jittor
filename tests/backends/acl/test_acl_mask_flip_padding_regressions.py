"""Public ACL assignment, multi-axis flip and padding regression cases."""
import numpy as np
import pytest
import jittor as jt
from _helpers import capability


@pytest.fixture(autouse=True)
def _acl_device():
    if not capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip('real Ascend device required')
    with jt.flag_scope(use_acl=1, use_cuda=1):
        yield
        jt.sync_all(True)


def _array(value):
    value.sync()
    assert value.location() == 'device'
    return value.numpy()


def test_public_mask_assignment_with_negative_integer():
    x = jt.zeros((2, 3))
    x[-1, jt.array([True, False, True])] = 7
    np.testing.assert_array_equal(_array(x), [[0, 0, 0], [7, 0, 7]])


@pytest.mark.parametrize('index', [2, -3])
def test_public_mask_assignment_rejects_out_of_bounds_without_writing(index):
    x = jt.zeros((2, 3))
    with pytest.raises(IndexError, match='out of bounds'):
        x[index, jt.array([True, False, True])] = 7
    np.testing.assert_array_equal(_array(x), np.zeros((2, 3)))


def test_public_mask_assignment_rejects_mismatched_mask():
    x = jt.zeros((2, 3))
    with pytest.raises(IndexError, match='mask shape'):
        x[jt.array([True])] = 7
    np.testing.assert_array_equal(_array(x), np.zeros((2, 3)))


def test_public_rank_two_mask_preserves_trailing_axis():
    x = jt.zeros((2, 2, 3))
    mask = np.array([[True, False], [False, True]])
    x[jt.array(mask)] = 5
    expected = np.zeros((2, 2, 3))
    expected[mask] = 5
    np.testing.assert_array_equal(_array(x), expected)


def test_multi_axis_flip_forward_and_backward():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    weights = np.arange(24, dtype=np.float32).reshape(data.shape) / 16
    x = jt.array(data).start_grad()
    out = jt.flip(x, (0, 2))
    grad = jt.grad((out * jt.array(weights)).sum(), x)
    np.testing.assert_array_equal(_array(out), np.flip(data, (0, 2)))
    np.testing.assert_array_equal(_array(grad), np.flip(weights, (0, 2)))


@pytest.mark.parametrize('mode', ['constant', 'replicate'])
def test_padding_forward_and_backward(mode):
    data = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    x = jt.array(data).start_grad()
    out = jt.nn.pad(x, (1, 2, 1, 0), mode=mode)
    grad = jt.grad(out.sum(), x)
    widths = ((0, 0), (0, 0), (1, 0), (1, 2))
    expected = np.pad(data, widths, mode='edge' if mode == 'replicate' else 'constant')
    counts = np.ones_like(data) if mode == 'constant' else np.array([[[[4, 2, 6], [2, 1, 3]]]])
    np.testing.assert_array_equal(_array(out), expected)
    np.testing.assert_array_equal(_array(grad), counts)
