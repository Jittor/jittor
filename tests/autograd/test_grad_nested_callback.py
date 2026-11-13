"""Nested differentiation inside a callback keeps the outer graph intact."""
import numpy as np
import pytest

import jittor as jt
from _helpers import capability


@pytest.mark.parametrize("use_cuda", [False, True], ids=["cpu", "cuda"])
def test_nested_grad_callback_preserves_shared_multioutput_gradients(use_cuda):
    if use_cuda and not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA is unavailable in this build")
    callbacks = []

    class Pair(jt.Function):
        def execute(self, x, weight):
            self.x, self.weight = x, weight
            return x * weight, x + weight

        def grad(self, product_grad, sum_grad):
            # This independent backward has its own traversal. Materializing it
            # also enters the executor while the outer backward is being built.
            probe = jt.array([2., -3.])
            inner = jt.grad((probe * probe).sum(), probe)
            np.testing.assert_array_equal(inner.numpy(), [4., -6.])
            callbacks.append(True)
            # A new consumer must not join the outer loss's recorded paths.
            unused = self.x * probe
            unused.stop_grad()
            return (self.weight * product_grad + sum_grad,
                    self.x * product_grad + sum_grad)

    with jt.flag_scope(use_cuda=int(use_cuda), missing_grad_error=1):
        x = jt.array([1., 2.])
        weight = jt.array([3., 4.])
        a, b = Pair()(x, weight)
        c, d = Pair()(x, weight)
        loss = (a + 2*b + 3*c + 4*d + x*x).sum()
        dx, dw = jt.grad(loss, [x, weight])
        np.testing.assert_array_equal(dx.numpy(), [20., 26.])
        np.testing.assert_array_equal(dw.numpy(), [10., 14.])
        assert len(callbacks) == 2
        if use_cuda:
            assert dx.location() == dw.location() == "device"
