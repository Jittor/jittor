"""Torch Adam state boundaries do not depend on which result is read first."""
import numpy as np
import pytest
import jittor as jt

from jittor.optim.algorithms.adam import adam_update


def _update(dtype, decoupled, states_first, gradient_first=False):
    rng = np.random.RandomState(13)
    parameter, gradient, second_moment, first_moment = (
        rng.uniform(low, high, 332).astype(dtype)
        for low, high in ((-.2, .3), (-.04, .04), (1e-5, 5e-4), (-.003, .003)))
    parameter = jt.array(parameter, dtype=dtype).start_grad()
    gradient = jt.array(gradient, dtype=dtype).start_grad()
    gradient = gradient * .37 + .0007
    second_moment = jt.array(second_moment, dtype=dtype).stop_grad()
    first_moment = jt.array(first_moment, dtype=dtype).stop_grad()
    if gradient_first:
        gradient.sync()
    result = adam_update(
        parameter, gradient, second_moment, first_moment,
        lr=.003, eps=1e-5, weight_decay=.02, betas=(.8, .93), step=3,
        decoupled_weight_decay=decoupled, torch_math=True)
    if states_first:
        first_moment.sync()
        second_moment.sync()
    result.sync()
    return tuple(np.array(value.numpy(), copy=True)
                 for value in (result, first_moment, second_moment))


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("decoupled", (False, True), ids=("Adam", "AdamW"))
def test_torch_adam_cpu_state_and_parameter_fetch_order(dtype, decoupled):
    # A fresh checkpoint materializes states; an uninterrupted step may consume
    # them in the same fused kernel as the parameter. Both must round alike.
    with jt.flag_scope(use_cuda=0):
        parameters_first = _update(dtype, decoupled, states_first=False)
        states_first = _update(dtype, decoupled, states_first=True)
    for left, right in zip(parameters_first, states_first):
        assert left.dtype == right.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("decoupled", (False, True), ids=("Adam", "AdamW"))
def test_torch_adam_cpu_gradient_materialization(dtype, decoupled):
    with jt.flag_scope(use_cuda=0):
        lazy = _update(dtype, decoupled, states_first=False)
        materialized = _update(dtype, decoupled, states_first=False, gradient_first=True)
    for left, right in zip(lazy, materialized):
        assert left.dtype == right.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(left, right)
