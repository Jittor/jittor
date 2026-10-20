"""Persistent flat ACL AdamW: it computes the same update as the per-tensor path.

The flat layout re-points every parameter and state at one buffer, so the
comparison is between two independently built models, not two code paths on
one model: re-pointing is not reversible after the fact.
"""

import numpy as np
import torch
import jittor as jt

from _helpers.device_types import instantiate_device_type_tests
from jittor._runtime.fallback import forbid_backend_fallbacks

import jittor.optim.algorithms._acl_flat as flat_module

_ORIGINAL_SETUP = flat_module.setup_group


def _build(device):
    values = [
        (torch.tensor([.5, -.25, .125, -.0625, .0625, -.125], device=device,
                      dtype=torch.float32).reshape(2, 3),
         [.25, -.5, .75, -.125, .5, -.25]),
        (torch.tensor([.75, -.375], device=device, dtype=torch.float32),
         [-.25, .5]),
        (torch.tensor([.25, .5, -.75, .125, -.25, .375], device=device,
                      dtype=torch.float32).reshape(3, 2),
         [.5, .25, -.125, -.5, .125, .75]),
    ]
    params = [torch.nn.Parameter(value) for value, _ in values]
    coefficients = [
        torch.tensor(coefficient, device=device, dtype=torch.float32).reshape(
            tuple(value.shape))
        for value, coefficient in values
    ]
    return params, coefficients


def _run(device, flatten, steps=4):
    flat_module.setup_group = _ORIGINAL_SETUP if flatten else (lambda *a, **k: None)
    try:
        params, coefficients = _build(device)
        opt = torch.optim.AdamW(params, lr=.01, eps=1e-4, weight_decay=.1)
        states = getattr(opt, "_acl_flat_states", None)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = sum((param * coefficient).sum()
                       for param, coefficient in zip(params, coefficients))
            loss.backward()
            opt.step()
        jt.sync_all()
        arrays = [param.detach().float().numpy().copy() for param in params]
        moments = [(opt.state[param]["exp_avg"].float().numpy().copy(),
                    opt.state[param]["exp_avg_sq"].float().numpy().copy())
                   for param in params]
        return arrays, moments, states
    finally:
        flat_module.setup_group = _ORIGINAL_SETUP


class TestAclFlatAdamw:
    def test_flat_layout_matches_per_tensor(self, device):
        with torch.enable_grad(), forbid_backend_fallbacks():
            flat_params, flat_moments, states = _run(device, True)
            assert states, "the ACL flat layout declined a plain parameter group"
            state = next(iter(states.values()))
            assert state.broken is False
            reference_params, reference_moments, reference_states = _run(
                device, False)
            assert not reference_states

            for actual, expected in zip(flat_params, reference_params):
                np.testing.assert_array_equal(actual, expected)
            for actual, expected in zip(flat_moments, reference_moments):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
            # The layout is what the step relies on: every parameter and
            # state still names a slice of its flat buffer.
            for value in state.params:
                assert value.data_ptr() != 0
                assert value._shares_storage_with(state.flat_params)
            for value in state.moments:
                assert value._shares_storage_with(state.flat_moments)
            for value in state.variances:
                assert value._shares_storage_with(state.flat_variances)

    def test_flat_state_is_acl_only(self, device):
        with torch.enable_grad(), forbid_backend_fallbacks():
            _, _, states = _run(device, True)
            state = next(iter(states.values()))
            # Host readbacks migrate a Var for inspection, so compare the
            # explicit placement rather than `location()`.
            for value in (state.flat_params, state.flat_moments,
                          state.flat_variances):
                assert value.placement_backend == state.params[0].placement_backend
                assert value.device_id == state.params[0].device_id
            assert int(state.flat_params.numel()) == sum(
                int(value.numel()) for value in state.params)


instantiate_device_type_tests(TestAclFlatAdamw, globals(), only_for=("npu",))
