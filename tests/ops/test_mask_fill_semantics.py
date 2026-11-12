"""Index validation for the device-independent masked-fill composition."""
import numpy as np
import pytest
import jittor as jt
from jittor.ops.indexing import _bool_mask_fill


def test_negative_integer_and_boolean_mask_select_the_last_row():
    x = jt.zeros((2, 3))
    _bool_mask_fill(x, (-1, jt.array([True, False, True])), 7)
    np.testing.assert_array_equal(x.numpy(), [[0, 0, 0], [7, 0, 7]])


@pytest.mark.parametrize('index', [2, -3])
def test_invalid_integer_index_raises_before_mutating(index):
    x = jt.zeros((2, 3))
    with pytest.raises(IndexError, match='out of bounds'):
        _bool_mask_fill(x, (index, jt.array([True, False, True])), 7)
    np.testing.assert_array_equal(x.numpy(), np.zeros((2, 3)))


def test_boolean_mask_is_not_broadcast_to_a_different_axis_length():
    x = jt.zeros((2, 3))
    with pytest.raises(IndexError, match='mask shape'):
        _bool_mask_fill(x, (jt.array([True]),), 7)
    np.testing.assert_array_equal(x.numpy(), np.zeros((2, 3)))


def test_two_dimensional_mask_keeps_trailing_dimensions():
    x = jt.zeros((2, 2, 3))
    mask = np.array([[True, False], [False, True]])
    _bool_mask_fill(x, jt.array(mask), 5)
    expected = np.zeros((2, 2, 3))
    expected[mask] = 5
    np.testing.assert_array_equal(x.numpy(), expected)


@pytest.fixture(autouse=True, params=['cpu', 'cuda'])
def _device(request):
    from _helpers import capability
    from jittor._runtime.fallback import forbid_backend_fallbacks
    device = request.param
    if device == 'cuda' and not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('real CUDA device required')
    with jt.flag_scope(use_cuda=int(device == 'cuda')), forbid_backend_fallbacks():
        probe = jt.ones(1).sync()
        if device == 'cuda':
            assert probe.location() == 'device'
        yield
        jt.sync_all(True)
