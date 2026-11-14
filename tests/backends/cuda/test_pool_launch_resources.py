"""The NHWC max-pool launch fits device limits in forward and backward."""

import numpy as np
import pytest
import jittor as jt

from _helpers.capability import require_accelerator
from jittor.nn.functional.pooling.core_2d import _max_pool2d_nhwc


@pytest.mark.cuda
def test_wide_nhwc_pool_has_exact_values_and_gradients():
    require_accelerator('cuda')
    shape = (2, 112, 112, 64)
    data = np.arange(np.prod(shape), dtype='float32').reshape(shape) * 2**-20
    with jt.flag_scope(use_cuda=1, backend_fallback='error'):
        x = jt.array(data).start_grad()
        y = _max_pool2d_nhwc(x, 'maximum', (3, 3), (2, 2), (1, 1), 56, 56)
        gradient = jt.grad(y.sum(), x)
        jt.sync([y, gradient])
        assert y.location() == gradient.location() == 'device'
        # Strictly increasing rows/columns put every maximum at an odd
        # row and column; every selected element receives exactly one unit.
        expected = data[:, 1::2, 1::2, :].transpose(0, 3, 1, 2)
        expected_gradient = np.zeros(shape, dtype='float32')
        expected_gradient[:, 1::2, 1::2, :] = 1
        np.testing.assert_array_equal(y.numpy(), expected)
        np.testing.assert_array_equal(gradient.numpy(), expected_gradient)
