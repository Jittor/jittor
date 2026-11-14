"""A CPU prefix remains live while the device training step is recorded."""
from unittest.mock import patch

import numpy as np
import pytest
import jittor as jt
from _helpers.capability import require_accelerator
from _helpers.devices import explicit_tensor_placement as placement
from jittor._runtime.step_capture import StepCapture


def array(values, backend):
    with placement(backend):
        value = jt.array(np.asarray(values, dtype='float32'))
    value.sync(True, False)
    return value


def training_step(weight, backend, opaque=False):
    def step(x, host):
        with placement(0):
            if opaque:
                scale = jt.code(host.shape, host.dtype, [host],
                                cpu_src='@out(0)=@in0(0)+2;')
            else:
                scale = jt.sqrt(host*host+2)
        scale = jt.device_copy(scale, 0 if backend else -1)
        loss = ((weight*x)/scale).sqr().sum()
        gradient = jt.grad(loss, weight)
        weight.update(weight-.01*gradient)
        return loss.detach().stop_grad()
    return step


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
def test_dynamic_prefix_training_state_and_retained_outputs_match_eager(device):
    if device == 'cuda':
        require_accelerator('cuda')
    backend = int(device == 'cuda')
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=backend, auto_graph_replay=0, backend_fallback='error'):
        eager_weight = array([1., 2., 3., 4.], backend).start_grad()
        captured_weight = array([1., 2., 3., 4.], backend).start_grad()
        eager = training_step(eager_weight, backend)
        captured = jt.capture_step(training_step(captured_weight, backend))
        expected, observed = [], []
        try:
            for index in range(12):
                x = array(np.arange(4)+index+1., backend).stop_grad()
                host = array([index%5-2.], 0).stop_grad()
                expected.append(eager(x, host))
                observed.append(captured(x, host))
            jt.sync(expected+observed, True, False)
            np.testing.assert_allclose([x.numpy() for x in observed], [x.numpy() for x in expected],
                                       atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(captured_weight.numpy(), eager_weight.numpy(), atol=2e-6, rtol=2e-6)
            assert captured.refused is None
            if backend:
                assert captured.stats['graph'] > 0, captured._graph_refused
                assert captured._graph_has_host_prefix
                assert captured._graph_host_prefix_bytes > 0
        finally:
            captured.invalidate()
        assert not captured._graph_has_host_prefix
    assert jt.core.backend_fallback_count() == before


@pytest.mark.cuda
def test_failed_prefix_launch_releases_graph_and_takes_exactly_one_step():
    require_accelerator('cuda')
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback='error'):
        a = array([1., 2., 3., 4.], 1).start_grad()
        b = array([1., 2., 3., 4.], 1).start_grad()
        eager, captured = training_step(a, 1), jt.capture_step(training_step(b, 1))
        try:
            for index in range(7):
                x = array(np.arange(4)+1., 1).stop_grad(); host = array([index+1.], 0).stop_grad()
                eager(x, host).sync(True, False); captured(x, host).sync(True, False)
            assert captured._graph_has_host_prefix, captured._graph_refused
            handle = captured._cuda_graph
            release = jt.core.graph_release
            with patch.object(jt.core, 'graph_launch_host_prefix', return_value=False), \
                    patch.object(jt.core, 'graph_release', wraps=release) as released:
                x = array(np.arange(4)+2., 1).stop_grad(); host = array([3.], 0).stop_grad()
                expected, observed = eager(x, host), captured(x, host)
                jt.sync([expected, observed], True, False)
                released.assert_called_once_with(handle)
                assert captured._cuda_graph == 0
                np.testing.assert_allclose(observed.numpy(), expected.numpy(), atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(b.numpy(), a.numpy(), atol=2e-6, rtol=2e-6)
        finally:
            captured.invalidate()


@pytest.mark.cuda
def test_prefix_budget_and_opaque_code_remain_recording_barriers():
    require_accelerator('cuda')
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback='error'):
        weight = array([1., 2., 3., 4.], 1).start_grad()
        limited = StepCapture(training_step(weight, 1), max_retained_bytes=65536)
        try:
            with patch.object(jt.core, 'graph_host_prefix_bytes', return_value=2**20):
                for index in range(7):
                    limited(array(np.ones(4), 1).stop_grad(), array([index+1.], 0).stop_grad()).sync(True, False)
            assert limited.stats['graph'] == 0
            assert 'host prefix' in limited._graph_refused
        finally:
            limited.invalidate()
        opaque = jt.capture_step(training_step(array([1., 2., 3., 4.], 1).start_grad(), 1, opaque=True))
        try:
            for index in range(7):
                opaque(array(np.ones(4), 1).stop_grad(), array([index+1.], 0).stop_grad()).sync(True, False)
            assert opaque.stats['graph'] == 0
            assert opaque._graph_refused is not None
        finally:
            opaque.invalidate()


@pytest.mark.cuda
@pytest.mark.parametrize('kind', ['pure', 'opaque', 'random', 'setitem'])
def test_dynamic_index_prefix_requires_explicit_opt_in_and_read_only_dependencies(kind):
    require_accelerator('cuda')
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0, auto_flush_ops=0,
                       backend_fallback='error'):
        source = array(np.arange(4)+1., 1).stop_grad()
        if kind == 'opaque':
            condition = jt.code((4,), 'float32', [source],
                                cuda_src='cudaMemcpyAsync(out0_p, in0_p, '
                                         '4*sizeof(float), cudaMemcpyDeviceToDevice);')
        elif kind == 'random':
            condition = jt.rand((4,))
        elif kind == 'setitem':
            condition = source.clone()
            condition[0] = -1
        else:
            condition = source*2-1
        # The native dynamic-shape converter resolves Where immediately.
        # Preserve its graph, as the training capture does while tracing.
        with jt.flag_scope(keep_graph=2):
            outputs = jt.where(condition > 0)
        assert jt.core.graph_host_prefix_bytes(outputs) == -1
        count = jt.core.graph_host_prefix_bytes(outputs, True)
        if kind == 'pure':
            assert count >= 4*8  # Upper bound of four int64 indices.
        else:
            assert count == -1
        for output in outputs:
            output._release_kept()
        jt.sync(outputs, True, False)


@pytest.mark.cuda
def test_dynamic_indices_with_changing_cpu_scale_preserve_queued_training_steps():
    require_accelerator('cuda')
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback='error'):
        feeds = []
        for index in range(16):
            labels = np.ones(4, dtype='float32')
            labels[index % 4] = -1
            if index >= 9:
                labels[(index+1) % 4] = -1
            feeds.append((array(np.arange(4)+1., 1).stop_grad(),
                          array([index % 5+1.], 0).stop_grad(),
                          array(labels, 1).stop_grad()))

        def step_for(weight):
            def step(x, host, labels):
                with placement(0):
                    scale = jt.sqrt(host*host+2)
                scale = jt.device_copy(scale, 0)
                selected = (weight*x)[jt.where(labels >= 0)[0]]
                loss = (selected/scale).sqr().mean()
                weight.update(weight-.01*jt.grad(loss, weight))
                return loss.detach().stop_grad()
            return step

        a = array([1., 2., 3., 4.], 1).start_grad()
        b = array([1., 2., 3., 4.], 1).start_grad()
        eager = step_for(a)
        expected = [eager(*feed) for feed in feeds]
        jt.sync(expected, True, False)
        captured = jt.capture_step(step_for(b))
        try:
            # Retain and queue outputs: no caller synchronization between
            # training calls may accidentally mask a host-staging race.
            observed = [captured(*feed) for feed in feeds]
            jt.sync(observed, True, False)
            np.testing.assert_allclose([v.numpy() for v in observed],
                                       [v.numpy() for v in expected], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(b.numpy(), a.numpy(), atol=2e-6, rtol=2e-6)
            assert captured.stats['graph'] >= 3, captured._graph_refused
            assert 'dynamic prefix' in captured.last_invalidated
        finally:
            captured.invalidate()


@pytest.mark.cuda
@pytest.mark.parametrize('record', [True, False])
def test_hook_cannot_change_dynamic_extent_before_an_update(record):
    require_accelerator('cuda')
    from jittor._runtime.step_capture import on_replay
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback='error'):
        weight = array([1., 2., 3., 4.], 1).start_grad()
        feed = array([1., 1., 1., 1.], 1).stop_grad()
        change = [False]

        def step(labels):
            def hook():
                if change[0]:
                    labels._write_inplace(np.full(4, -1., dtype='float32'))
            on_replay(hook)
            loss = weight[jt.where(labels >= 0)[0]].sqr().mean()
            weight.update(weight-.01*jt.grad(loss, weight))
            return loss.detach().stop_grad()

        runner = jt.capture_step(step, record=record)
        try:
            for _ in range(8):
                runner(feed).sync(True, False)
            assert runner.stats['replayed'] >= 4
            if record:
                assert runner.stats['graph'] >= 3, runner._graph_refused
            before = weight.numpy().copy()
            change[0] = True
            with pytest.raises(RuntimeError, match='hook changed the dynamic prefix'):
                runner(feed)
            np.testing.assert_array_equal(weight.numpy(), before)
        finally:
            runner.invalidate()
