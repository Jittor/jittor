"""CPU BF16 SGD follows the scalar and accumulation boundaries of Torch."""

import numpy as np

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.optim.algorithms.sgd import sgd_update


def _round_bfloat16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def test_native_cpu_bfloat16_sgd_rounds_momentum_and_alpha_like_torch():
    # These values discriminate the fused Jittor expression from the CPU
    # torch.optim.SGD sequence: BF16 momentum product, add gradient, and
    # add_(direction, alpha=-lr). The latter rounds alpha to the parameter
    # dtype before its FP32 opmath accumulation.
    p0 = np.array(
        [-0.220703125, -0.1083984375, 0.0036468505859375, -0.01324462890625],
        dtype=np.float32,
    )
    grad1 = np.array([-0.3359375, 0.3203125, -0.462890625, -0.99609375], dtype=np.float32)
    grad2 = np.array([0.400390625, 0.73828125, 0.96875, -0.22265625], dtype=np.float32)
    lr, momentum = 0.005, 0.9
    alpha = _round_bfloat16([-lr])[0]

    first_velocity = _round_bfloat16(grad1)
    first_parameter = _round_bfloat16(p0 + alpha * first_velocity)
    second_velocity = _round_bfloat16(_round_bfloat16(momentum * first_velocity) + grad2)
    second_parameter = _round_bfloat16(first_parameter + alpha * second_velocity)

    with jt.flag_scope(use_cuda=0), forbid_backend_fallbacks():
        parameter = jt.array(p0).bfloat16().stop_grad()
        velocity = jt.zeros_like(parameter).stop_grad()
        first_grad = jt.array(grad1).bfloat16().stop_grad()
        next_grad = jt.array(grad2).bfloat16().stop_grad()

        parameter = sgd_update(
            parameter,
            first_grad,
            velocity,
            lr=lr,
            momentum=momentum,
            momentum_initialized=False,
        )
        jt.sync_all()
        np.testing.assert_array_equal(parameter.float32().numpy(), first_parameter)
        np.testing.assert_array_equal(velocity.float32().numpy(), first_velocity)

        parameter = sgd_update(
            parameter,
            next_grad,
            velocity,
            lr=lr,
            momentum=momentum,
            momentum_initialized=True,
        )
        jt.sync_all()
        np.testing.assert_array_equal(velocity.float32().numpy(), second_velocity)
        np.testing.assert_array_equal(parameter.float32().numpy(), second_parameter)
