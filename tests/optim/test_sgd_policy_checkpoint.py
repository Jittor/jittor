"""Native SGD preserves its numerical policy across real checkpoint I/O."""

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


def _make(value, policy):
    parameter = jt.array(value, dtype='float32').start_grad()
    optimizer = jt.optim.SGD(
        [parameter], lr=.01, momentum=.9, weight_decay=.03,
        dampening=.2, torch_math=policy,
    )
    return parameter, optimizer


def _snapshot(parameter, optimizer):
    # Native clone shares storage; numpy() on that clone would migrate the
    # live parameter too. Read an explicit independent host copy instead.
    return [jt.device_copy(value, -1).numpy().copy() for value in (
        parameter, optimizer.param_groups[0]['values'][0],
    )]


def _advance(parameter, optimizer, start, end, device):
    rows = []
    for step in range(start, end):
        factor = (1., -.5, 1.5, .25, -1.)[step]
        gradient = jt.array(np.array([-.3359375, -.0220947265625,
                                     -.019775390625, -.052001953125],
                                    dtype='float32') * factor).stop_grad()
        optimizer.step((parameter * gradient).sum())
        for value in (parameter, optimizer.param_groups[0]['values'][0]):
            value.sync()
            assert str(value.dtype) == 'float32'
            assert value.location() == ('device' if device == 'cuda' else 'cpu')
        rows.append(_snapshot(parameter, optimizer))
    return rows


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        not jt.has_cuda, reason='requires CUDA',
    )),
])
@pytest.mark.parametrize('policy', [False, True])
def test_native_sgd_policy_checkpoint(tmp_path, device, policy):
    with jt.flag_scope(use_cuda=device == 'cuda'), forbid_backend_fallbacks():
        initial = [.0172119140625, -.0023345947265625,
                   -.004791259765625, -.0089111328125]
        parameter, optimizer = _make(initial, policy)
        assert jt.optim.SGD([jt.array(initial)], lr=.01).torch_math is False
        _advance(parameter, optimizer, 0, 3, device)
        saved = _snapshot(parameter, optimizer)
        path = tmp_path / 'sgd.pkl'
        jt.save({'parameter': parameter, 'optimizer': optimizer.state_dict()}, str(path))
        checkpoint = jt.load(str(path))
        assert checkpoint['optimizer']['defaults']['torch_math'] is policy
        baseline = _advance(parameter, optimizer, 3, 5, device)

        restored, new_optimizer = _make(checkpoint['parameter'], not policy)
        new_optimizer.load_state_dict(checkpoint['optimizer'])
        assert new_optimizer.torch_math is policy
        for actual, expected in zip(_snapshot(restored, new_optimizer), saved):
            np.testing.assert_array_equal(actual, expected)
        resumed = _advance(restored, new_optimizer, 3, 5, device)
        for actual_row, expected_row in zip(resumed, baseline):
            for actual, expected in zip(actual_row, expected_row):
                np.testing.assert_array_equal(actual, expected)

        legacy = {'defaults': dict(checkpoint['optimizer']['defaults'])}
        del legacy['defaults']['torch_math']
        _, legacy_optimizer = _make(initial, not policy)
        legacy_optimizer.load_state_dict(legacy)
        assert legacy_optimizer.torch_math is not policy
