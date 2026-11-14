"""A captured optimizer step must apply its update exactly once.

The capture's post-trace sync used to be forced whenever the traced step
published gradients, even when the call had already materialized every root.
That re-ran the kept graph and applied the optimizer update a second time:
AdamW's fused update materializes its state roots inside the traced call, so
a four-step loss trajectory skipped a step. These tests compare the captured
trajectory against an eager twin on the same initial weights.
"""

import numpy as np
import pytest
import torch

from jittor._runtime import step_capture
import jittor as jt


@pytest.mark.cuda
def test_captured_adam_counters_remain_live_when_only_some_are_observed():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    p = torch.nn.Parameter(torch.tensor([.2, -.3], device='cuda'))
    q = torch.nn.Parameter(torch.tensor([.4, .1], device='cuda'))
    optimizer = torch.optim.AdamW([p, q], lr=.003)

    def step():
        loss = (p*2+q).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        return loss.detach()

    runner = torch.compile(step, mode='reduce-overhead', dynamic=False)
    try:
        for _ in range(8):
            assert np.isfinite(runner().item())
        assert runner.stats['graph'] >= 3, runner._graph_refused
        observed = optimizer.state[p]['step']
        assert observed.item() == 8
        # The other parameter remains unobserved across another replay.
        observed.data.fill_(8)
        runner().item()
        assert observed.item() == 9
        later = optimizer.state[q]['step']
        assert later.item() == 9
        # Both exposed aliases become authoritative without recapturing.
        observed.data.fill_(99); later.data.fill_(99)
        runner().item()
        assert observed.item() == later.item() == 100
        assert optimizer.state[p]['step'] is observed
        assert optimizer.state[q]['step'] is later
    finally:
        runner.invalidate()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('record', [False, True])
@jt.flag_scope(backend_fallback='error')
def test_captured_adam_returns_preupdate_gradient_snapshots(device, record):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required')

    def trajectory(captured):
        p = torch.nn.Parameter(torch.tensor([.7, -.4], device=device))
        q = torch.nn.Parameter(torch.tensor([.2, .5], device=device))
        optimizer = torch.optim.AdamW([p, q], lr=.02)

        def step(x):
            loss = ((x * p + q).square() * p).mean()
            loss.backward()
            snapshots = tuple(v.grad.detach().clone() for v in (p, q))
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach(), snapshots

        runners = [step_capture.StepCapture(step, record=record) for _ in range(2)]
        history = []
        try:
            for index in range(12):
                data = np.arange((3 + index % 2) * 2, dtype='float32').reshape(-1, 2) / 10 + .5
                x = torch.tensor(data, device=device)
                loss, gradients = runners[index % 2](x) if captured else step(x)
                history.append((loss.item(), *[
                    value.detach().cpu().numpy().copy() for value in (*gradients, p, q)]))
            if captured:
                for runner in runners:
                    if device == 'cpu':
                        assert runner.refused == 'the per-parameter Adam update bakes its step count in'
                        assert runner.stats['replayed'] == 0
                        continue
                    assert runner.refused is None
                    assert runner.stats['replayed'] >= 4
                    if record and device == 'cuda':
                        assert runner.stats['graph'] > 0, runner._graph_refused
            return history
        finally:
            for runner in runners:
                runner.invalidate()

    for expected, actual in zip(trajectory(False), trajectory(True)):
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(right, left, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('change_at', [2, 8])
@pytest.mark.parametrize('opaque', [False, True])
@jt.flag_scope(backend_fallback='error')
def test_nonzero_prefix_refreshes_indices_and_shape_change_takes_one_adam_step(
        device, record, change_at, opaque):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    def trajectory(capture):
        parameter = torch.nn.Parameter(torch.tensor([.2, -.3, .5, 1.], device=device))
        optimizer = torch.optim.AdamW([parameter], lr=.003)

        def step(x, labels):
            if opaque:
                labels = jt.code(labels.shape, labels.dtype, [labels],
                                 cpu_src='std::memcpy(out0_p, in0_p, out0->size);',
                                 cuda_src='cudaMemcpyAsync(out0_p, in0_p, '
                                          'out0->size, cudaMemcpyDeviceToDevice);')
            indices = (labels >= 0).nonzero().reshape(-1)
            loss = (parameter*x).index_select(0, indices).square().mean()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return loss.detach()

        runner = step_capture.StepCapture(step, record=record) if capture else step
        values = []
        try:
            for index in range(17):
                labels = np.ones(4, dtype='int64')
                labels[index % 4] = -1
                if index >= change_at:
                    labels[(index+1) % 4] = -1
                x = torch.tensor([1., -2., 3., .5], device=device)
                values.append(float(runner(x, torch.tensor(labels, device=device)).item()))
                if capture and device == 'cuda' and record and not opaque and change_at == 8 and index == 7:
                    assert runner.stats['graph'] > 0, runner._graph_refused
                    assert runner._graph_has_host_prefix
            if capture:
                if device == 'cuda' and not opaque:
                    assert runner.stats['captured'] == 2, runner.last_invalidated
                    assert 'dynamic prefix' in runner.last_invalidated
                    if record:
                        assert runner.stats['graph'] >= 5, runner._graph_refused
                else:
                    assert runner.refused is not None
                    assert runner.stats['replayed'] == 0
                if not record:
                    assert runner.stats['graph'] == 0
            assert int(optimizer.state[parameter]['step']) == 17
            return values, parameter.detach().cpu().numpy().copy()
        finally:
            if capture:
                runner.invalidate()
    expected, weights = trajectory(False)
    observed, actual = trajectory(True)
    np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual, weights, atol=2e-6, rtol=2e-6)


def _inputs(device):
    generator = np.random.RandomState(0)
    x = torch.from_numpy(generator.randn(16, 4).astype("float32")).to(device)
    y = torch.from_numpy(generator.randn(16, 1).astype("float32")).to(device)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1).to(device)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(
            (0.1 * generator.randn(1, 4)).astype("float32")))
        model.bias.copy_(torch.from_numpy(np.zeros(1, "float32")))
    return x, y, model


def _trajectory(kind, device, captured, steps=4):
    x, y, model = _inputs(device)
    if kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    def step():
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss

    runner = step_capture.StepCapture(step) if captured else step
    losses, params = [], []
    for _ in range(steps):
        losses.append(float(runner().numpy()))
        params.append(model.weight.detach().cpu().numpy().copy())
    return losses, params, getattr(runner, "refused", None), getattr(runner, "stats", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ["adamw", "sgd"])
def test_captured_optimizer_step_matches_eager(kind):
    eager_losses, eager_params, _, _ = _trajectory(kind, "cuda", False)
    losses, params, refused, stats = _trajectory(kind, "cuda", True)
    assert refused is None
    assert stats["replayed"] >= 2
    np.testing.assert_allclose(losses, eager_losses, rtol=1e-6, atol=1e-7)
    for captured, eager in zip(params, eager_params):
        np.testing.assert_allclose(captured, eager, rtol=1e-6, atol=1e-7)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('kind', ['adam', 'adamw'])
@jt.flag_scope(backend_fallback='error')
def test_scheduler_betas_and_other_live_adam_values_do_not_recapture_each_step(kind):
    def trajectory(capture):
        parameter = torch.nn.Parameter(torch.tensor([.2, -.3, .5, 1.], device='cuda'))
        cls = torch.optim.Adam if kind == 'adam' else torch.optim.AdamW
        optimizer = cls([parameter], lr=.01, weight_decay=0.)

        def step(x):
            optimizer.zero_grad()
            loss = (parameter*x).square().sum()
            loss.backward(); optimizer.step()
            return loss.detach()

        runner = step_capture.StepCapture(step) if capture else step
        values = []
        try:
            for index in range(14):
                group = optimizer.param_groups[0]
                group['lr'] = .003+.0002*index
                group['betas'] = (.85+.005*index, .99+.0001*index)
                group['eps'] = 1e-8 if index < 7 else 1e-6
                if kind == 'adamw':
                    group['weight_decay'] = .01+.001*index
                x = torch.tensor([1., -2., 3., .5], device='cuda')
                values.append(float(runner(x).item()))
            final = parameter.detach().cpu().numpy().copy()
            if capture:
                assert runner.stats['captured'] == 1, runner.last_invalidated
                assert runner.stats['graph'] >= 5, runner._graph_refused
            return values, final
        finally:
            if capture:
                runner.invalidate()
    expected, weights = trajectory(False)
    observed, actual = trajectory(True)
    np.testing.assert_allclose(observed, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual, weights, atol=2e-6, rtol=2e-6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@jt.flag_scope(backend_fallback='error')
def test_transposed_batched_product_metadata_does_not_block_training_graph():
    def trajectory(capture):
        parameter = torch.nn.Parameter(torch.ones((2, 4, 3), device='cuda')/4)
        optimizer = torch.optim.SGD([parameter], lr=.01)
        def step(x):
            loss = torch.bmm(x, parameter.transpose(1, 2)).square().mean()
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            return loss.detach()
        runner = step_capture.StepCapture(step) if capture else step
        try:
            values = [float(runner(torch.ones((2, 5, 3), device='cuda')*(1+i*.01)).item())
                      for i in range(10)]
            final = parameter.detach().cpu().numpy().copy()
            if capture:
                assert runner.stats['graph'] > 0, runner._graph_refused
            return values, final
        finally:
            if capture:
                runner.invalidate()
    expected, weights = trajectory(False)
    observed, actual = trajectory(True)
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual, weights, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize('record', [False, True])
@pytest.mark.parametrize('readout', ['loss', 'detached', 'post_update'])
@pytest.mark.parametrize('presync', [False, True])
@pytest.mark.parametrize('kind', ['adamw', 'sgd'])
def test_alternating_captures_share_optimizer_state_without_extra_updates(record, readout, presync, kind):
    def trajectory(captured):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
        optimizer_type = torch.optim.AdamW if kind == 'adamw' else torch.optim.SGD
        optimizer = optimizer_type([parameter], lr=.1)

        def step(x):
            optimizer.zero_grad()
            loss = (parameter * x).sum()
            if presync:
                torch.cuda.synchronize()
            loss.backward()
            optimizer.step()
            if readout == 'post_update':
                return (parameter * 2).sum().detach()
            return loss.detach() if readout == 'detached' else loss

        runners = [step_capture.StepCapture(step, record=record) for _ in range(2)] if captured else [step, step]
        history = []
        for index in range(12):
            x = torch.full((2 + index % 2, 2), 1. + .1 * index, device='cuda')
            output = runners[index % 2](x)
            torch.cuda.synchronize()
            history.append((output.item(), parameter.detach().cpu().numpy().copy()))
        if captured:
            for runner in runners:
                assert runner.refused is None
                assert runner.stats['replayed'] >= 4
                if record:
                    assert runner.stats['graph'] >= 2
        return history

    reference, candidate = trajectory(False), trajectory(True)
    for (want_loss, want_parameter), (loss, parameter) in zip(reference, candidate):
        np.testing.assert_allclose(loss, want_loss, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(parameter, want_parameter, rtol=2e-6, atol=1e-6)
