"""Native RMSprop decay uses the parameter's actual backend and dtype."""

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        not jt.has_cuda, reason='requires CUDA',
    )),
])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('global_decay,group_decay', [(.25, None), (.25, 0.), (0., .5)])
def test_native_rmsprop_decay_location_state_and_gradient(device, dtype, global_decay, group_decay):
    with jt.flag_scope(use_cuda=device == 'cuda'), forbid_backend_fallbacks():
        initial = np.array([1., -2., .5], dtype=dtype)
        gradient = np.array([.5, -.25, .125], dtype=dtype)
        parameter = jt.array(initial, dtype=dtype)
        group = {'params': [parameter]}
        if group_decay is not None:
            group['weight_decay'] = group_decay
        optimizer = jt.optim.RMSprop(
            [group], lr=.03125, alpha=.875, eps=.125, weight_decay=global_decay,
        )
        expected, square = initial.copy(), np.zeros_like(initial)
        decay = global_decay if group_decay is None else group_decay
        tolerance = 2e-6 if dtype == 'float32' else 1e-12
        for _ in range(3):
            loss = (parameter * jt.array(gradient, dtype=dtype)).sum()
            np.testing.assert_array_equal(jt.grad(loss, parameter).numpy(), gradient)
            effective = gradient + decay * expected
            square = .875 * square + .125 * effective * effective
            expected -= .03125 * effective / (np.sqrt(square) + .125)
            optimizer.step(loss)
            state = optimizer.param_groups[0]['values'][0]
            parameter.sync()
            state.sync()
            for tensor in (parameter, state):
                assert str(tensor.dtype) == dtype
                assert tensor.location() == ('cpu' if device == 'cpu' else 'device')
            np.testing.assert_allclose(parameter.numpy(), expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(state.numpy(), square, rtol=tolerance, atol=tolerance)
        jt.sync_all(True)
