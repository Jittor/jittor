"""Refused captures finish existing values while preserving ordinary autograd."""

import numpy as np
import pytest

import jittor as jt
from jittor._runtime import step_capture
from _helpers.capability import require_accelerator


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_presynced_refused_forward_retains_first_and_second_derivatives(device):
    if device == 'cuda':
        require_accelerator('cuda')
    data = np.array([.2, -.7], dtype='float32')
    with jt.flag_scope(use_cuda=int(device == 'cuda'), auto_graph_replay=0,
                       backend_fallback='error'):
        x = jt.array(data).start_grad()

        def forward():
            output = x.tanh()
            output.sync()
            step_capture.refuse('the test requests ordinary autograd')
            return output

        runner = jt.capture_step(forward, record=False)
        try:
            runner()
            output = runner()
            assert runner.refused == 'the test requests ordinary autograd'
            first = jt.grad(output.sum(), x)
            second = jt.grad(first.sum(), x)
            y = np.tanh(data.astype('float64'))
            for actual, expected in ((first, 1 - y * y),
                                     (second, -2 * y * (1 - y * y))):
                actual.sync()
                assert actual.location() == ('device' if device == 'cuda' else 'cpu')
                np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-6)
        finally:
            runner.invalidate()


def test_amortized_recapture_across_sessions_does_not_refuse():
    with jt.flag_scope(use_cuda=0, auto_graph_replay=0), jt.no_grad():
        def step(x, cache):
            return x + cache

        runner = jt.capture_step(step, record=False)
        try:
            x = jt.array(np.ones((2,), dtype="float32"))
            # 10 sessions (> _GIVE_UP_AFTER=8) with different cache shapes, each
            # reusing its capture across 10 steps, must keep capturing.
            for session in range(10):
                cache = jt.array(np.full((session + 1, 2), float(session), dtype="float32"))
                for _ in range(12):
                    out = runner(x, cache)
                np.testing.assert_allclose(out.numpy(), np.full((session + 1, 2), float(session + 1), dtype="float32"))
            assert runner.refused is None
            assert runner.stats["captured"] == 10
            assert runner.stats["replayed"] == 100
        finally:
            runner.invalidate()
