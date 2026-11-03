"""Momentum-free SGD must not allocate or migrate unused parameter-sized state."""

from unittest.mock import patch

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.dispatch import dispatch_context
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.optim.algorithms import sgd


def _host(value):
    return jt.device_copy(value, -1).numpy().copy()


@pytest.mark.parametrize("fused", [False, True])
def test_plain_sgd_never_allocates_momentum(fused):
    with forbid_backend_fallbacks():
        parameter = jt.array([1., -2., 3., -4.]).start_grad()
        other = jt.array([2., 3., 4., 5.]).start_grad()
        with patch.object(sgd, "_momentum_buffer", wraps=sgd._momentum_buffer) as allocate:
            optimizer = jt.optim.SGD([parameter], lr=.1, fused=fused, weight_decay=.2)
            optimizer.add_param_group({"params": [other]})
            assert allocate.call_count == 0, "plain SGD allocated momentum at construction"
            for _ in range(2):
                optimizer.step((parameter * 2).sum() + (other * 3).sum())
                parameter.sync()
                other.sync()
            # CANN's fused ABI still requires a temporary velocity argument;
            # CPU/CUDA need no momentum allocation at all.
            if not (fused and dispatch_context(parameter).backend == "acl"):
                assert allocate.call_count == 0, "plain SGD allocated unused momentum storage"
        np.testing.assert_allclose(_host(parameter), [.5644, -2.3168, 2.4852, -4.2376], atol=1e-6)
        np.testing.assert_allclose(_host(other), [1.3268, 2.2872, 3.2476, 4.208], atol=1e-6)
        for group in optimizer.param_groups:
            assert all(value is None for value in group["values"])
            assert group["momentum_initialized"] == [False]


@pytest.mark.parametrize("fused", [False, True])
def test_late_momentum_and_paused_checkpoint(tmp_path, fused):
    with forbid_backend_fallbacks():
        parameter = jt.array([1., 2., 3., 4.]).start_grad()
        optimizer = jt.optim.SGD([parameter], lr=.1, dampening=.25, fused=fused)
        optimizer.step(parameter.sum())
        group = optimizer.param_groups[0]
        assert all(value is None for value in group["values"])
        group["momentum"] = .5
        optimizer.step((parameter * 2).sum())
        np.testing.assert_array_equal(_host(group["values"][0]), [2., 2., 2., 2.])
        group["momentum"] = 0
        optimizer.step((parameter * 3).sum())
        path = str(tmp_path / "paused-sgd.pkl")
        jt.save({"parameter": parameter, "optimizer": optimizer.state_dict()}, path)
        saved = jt.load(path)
        restored = jt.array(saved["parameter"]).start_grad()
        resumed = jt.optim.SGD([restored], lr=.1, dampening=.25, fused=fused)
        resumed.load_state_dict(saved["optimizer"])
        for value, opt in ((parameter, optimizer), (restored, resumed)):
            opt.param_groups[0]["momentum"] = .5
            opt.step((value * 4).sum())
        np.testing.assert_array_equal(_host(parameter), _host(restored))
        np.testing.assert_array_equal(_host(resumed.param_groups[0]["values"][0]), [4., 4., 4., 4.])


def test_plain_checkpoint_keeps_lazy_state(tmp_path):
    parameter = jt.array([1., 2.]).start_grad()
    optimizer = jt.optim.SGD([parameter], lr=.1, fused=False)
    optimizer.step(parameter.sum())
    path = str(tmp_path / "plain-sgd.pkl")
    jt.save({"parameter": parameter, "optimizer": optimizer.state_dict()}, path)
    saved = jt.load(path)
    restored = jt.array(saved["parameter"]).start_grad()
    resumed = jt.optim.SGD([restored], lr=.1, fused=False)
    resumed.load_state_dict(saved["optimizer"])
    with patch.object(sgd, "_momentum_buffer", wraps=sgd._momentum_buffer) as allocate:
        resumed.step(restored.sum())
        restored.sync()
        assert allocate.call_count == 0
        resumed.param_groups[0]["momentum"] = .9
        resumed.step((restored * 2).sum())
        restored.sync()
        assert allocate.call_count == 1
    np.testing.assert_allclose(_host(restored), [.6, 1.6], atol=1e-6)
    np.testing.assert_array_equal(_host(resumed.param_groups[0]["values"][0]), [2., 2.])


def test_plain_sgd_capture_replays_without_momentum_storage():
    parameter = jt.array([1., 2., 3., 4.]).start_grad()
    optimizer = jt.optim.SGD([parameter], lr=.125, fused=True)

    def step():
        loss = (parameter * parameter).sum()
        optimizer.step(loss)
        return loss

    captured = jt.capture_step(step)
    with forbid_backend_fallbacks():
        for _ in range(5):
            captured().sync()
        np.testing.assert_array_equal(_host(parameter), np.array([1., 2., 3., 4.]) * .75 ** 5)
    assert captured.refused is None
    assert captured.stats["replayed"] > 0
    assert all(value is None for value in optimizer.param_groups[0]["values"])
