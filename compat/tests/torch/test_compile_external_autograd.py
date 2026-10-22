"""Whole-step replay must preserve a caller's external autograd boundary."""
import numpy as np
import pytest
import jittor as jt
import torch
from _helpers.device_types import instantiate_device_type_tests


class _Model(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 3.], device=device))

    def forward(self, x):
        return (self.weight * x).sum()


def _compiled(model, kind):
    if kind == 'bound':
        return torch.compile(model.__call__, mode='reduce-overhead')
    if kind == 'closure':
        return torch.compile(lambda x: model(x), mode='reduce-overhead')
    return torch.compile(lambda x, weight: (weight * x).sum(), mode='reduce-overhead')


class TestCompileExternalAutograd:
    def test_published_accumulated_gradient_uses_one_ready_owner(self, device):
        parameter = torch.nn.Parameter(torch.tensor([2.], device=device))
        optimizer = torch.optim.SGD([parameter], lr=.1, foreach=False, fused=False)
        x = torch.tensor([.5], device=device)

        def step(value):
            optimizer.zero_grad(set_to_none=True)
            first = (parameter * value).sum()
            first.backward()
            second = (parameter * (2. * value)).sum()
            second.backward()
            optimizer.step()
            return second

        run = torch.compile(step, mode='reduce-overhead')
        losses = []
        for index in range(6):
            losses.append(float(run(x)))
            gradient = parameter.grad
            assert gradient is optimizer.param_groups[0]['grads'][0]
            assert gradient._mem_ptr_now
            np.testing.assert_allclose(float(gradient), 1.5, atol=1e-6)
            np.testing.assert_allclose(float(gradient.detach().cpu().numpy()[0]),
                                       1.5, atol=1e-6)
            if index in (1, 5):
                before = float(parameter.detach().cpu().numpy()[0])
                np.testing.assert_allclose(before, 2. - .15 * (index + 1), atol=1e-6)
                assert float(parameter.detach().cpu().numpy()[0]) == before
        assert losses[-1] < losses[0]
        assert run.stats['captured'] == 1
        assert run.stats['graph'] > 0

    def test_published_gradient_read_cannot_repeat_captured_update(self, device):
        # A live eager graph can already have published unrelated gradients.
        # Only gradients from this backward belong in the captured step.
        unrelated = torch.nn.Parameter(torch.tensor([7.], device=device))
        (unrelated * 2.).sum().backward()
        assert float(unrelated.grad) == 2.
        p = torch.nn.Parameter(torch.tensor([2.], device=device))
        q = torch.nn.Parameter(torch.tensor([3.], device=device))
        x = torch.tensor([.5], device=device)
        opt = torch.optim.SGD([p, q], lr=.01, foreach=False, fused=False)

        def step(value):
            opt.zero_grad()
            loss = ((p + q) * value).square().sum()
            loss.backward()
            opt.step()
            return loss

        run = torch.compile(step, mode='reduce-overhead')
        losses = []
        for index in range(6):
            losses.append(float(run(x)))
            if index not in (1, 5):
                continue
            expected_gradient = 2.5 * .99 ** index
            expected_parameter = 2. - .025 * sum(.99 ** k for k in range(index + 1))
            before = np.array([float(v.detach().cpu().numpy()[0]) for v in (p, q)])
            np.testing.assert_allclose(before, [expected_parameter, expected_parameter + 1.],
                                       rtol=1e-5, atol=1e-6)
            gradient = p.grad
            assert gradient is opt.param_groups[0]['grads'][0]
            assert gradient._mem_ptr_now
            for observed in (float(gradient), float(gradient),
                             float(gradient.detach().cpu().numpy()[0])):
                np.testing.assert_allclose(observed, expected_gradient,
                                           rtol=1e-5, atol=1e-6)
            after = np.array([float(v.detach().cpu().numpy()[0]) for v in (p, q)])
            np.testing.assert_array_equal(after, before)
        assert losses[-1] < losses[0]
        assert run.stats['captured'] == 1
        assert run.stats['replayed'] > 0
        assert run.stats['graph'] > 0
        assert float(unrelated.grad) == 2.

    def test_forward_gradients_belong_to_original_callers(self, device):
        for kind in ('bound', 'closure', 'functional'):
            for requires_grad in (False, True):
                model = _Model(device)
                run = _compiled(model, kind)
                for scale in (1., .5, 2., .25):
                    model.weight.grad = None
                    data = np.array([1., 4.], dtype='float32') * scale
                    x = torch.tensor(data, device=device, requires_grad=requires_grad)
                    out = run(x, model.weight) if kind == 'functional' else run(x)
                    out.backward()
                    np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), data)
                    if requires_grad:
                        np.testing.assert_array_equal(x.grad.cpu().numpy(), [2., 3.])
                    else:
                        assert x.grad is None
                assert run.refused is not None
                assert run.stats['captured'] == run.stats['replayed'] == 0

    def test_queued_forwards_keep_independent_graphs(self, device):
        model = _Model(device)
        run = _compiled(model, 'closure')
        data = [np.array([scale, scale * 4], dtype='float32') for scale in (1., .5, 2.)]
        outputs = [run(torch.tensor(value, device=device)) for value in data]
        for output, expected in zip(outputs, data):
            model.weight.grad = None
            output.backward()
            np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), expected)

    def test_retained_and_higher_order_outputs_keep_their_graphs(self, device):
        for higher_order in (False, True):
            model = _Model(device)
            def fn(x):
                loss = (model.weight.square() * x).sum()
                if higher_order:
                    return torch.autograd.grad(loss, model.weight, create_graph=True)[0]
                loss.backward(retain_graph=True)
                return loss
            run = torch.compile(fn, mode='reduce-overhead')
            for scale in (1., .5, 2.):
                model.weight.grad = None
                data = np.array([1., 4.], dtype='float32') * scale
                result = run(torch.tensor(data, device=device))
                result.sum().backward()
                expected = 2 * data if higher_order else 4 * np.array([2., 3.]) * data
                np.testing.assert_allclose(model.weight.grad.cpu().numpy(), expected,
                                           rtol=1e-5, atol=1e-6)
            assert run.refused is not None

    def test_training_step_preserves_differentiable_inputs(self, device):
        model = _Model(device)
        opt = torch.optim.SGD(model.parameters(), lr=.01, foreach=False)
        def step(x):
            opt.zero_grad(set_to_none=True)
            loss = (model.weight * x).square().sum()
            loss.backward()
            opt.step()
            return loss
        run = torch.compile(step, mode='reduce-overhead')
        expected = np.array([2., 3.], dtype='float32')
        data = np.array([.5, .25], dtype='float32')
        losses = []
        for _ in range(4):
            x = torch.tensor(data, device=device, requires_grad=True)
            before = expected.copy()
            losses.append(float(run(x)))
            np.testing.assert_allclose(x.grad.cpu().numpy(), 2 * before * before * data,
                                       rtol=1e-5, atol=1e-6)
            expected = before - .01 * 2 * before * data * data
            np.testing.assert_allclose(model.weight.detach().cpu().numpy(), expected,
                                       rtol=1e-5, atol=1e-6)
        assert losses[-1] < losses[0]
        assert run.refused is not None

    def test_no_grad_recording_does_not_survive_enabling_gradients(self, device):
        model = _Model(device)
        run = _compiled(model, 'bound')
        x = torch.tensor([1., 4.], device=device)
        with torch.no_grad():
            for _ in range(6):
                assert float(run(x)) == 14.
        assert run.stats['graph'] > 0
        for _ in range(2):
            model.weight.grad = None
            run(x).backward()
            np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), [1., 4.])
        assert run.refused is not None

    def test_frozen_closure_parameters_can_become_trainable(self, device):
        model = _Model(device)
        model.weight.requires_grad_(False)
        run = _compiled(model, 'closure')
        x = torch.tensor([1., 4.], device=device)
        for _ in range(2):
            assert float(run(x)) == 14.
        model.weight.requires_grad_(True)
        for _ in range(2):
            model.weight.grad = None
            run(x).backward()
            np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), [1., 4.])
        assert run.refused is not None


    def test_backward_probe_rejects_invalid_order_without_losing_scope(self, device):
        outer = jt.core._begin_step_backward_probe()
        inner = jt.core._begin_step_backward_probe()
        try:
            with pytest.raises(RuntimeError, match='creating thread in LIFO order'):
                jt.core._end_step_backward_probe(outer)
            model = _Model(device)
            model(torch.tensor([1., 4.], device=device)).backward()
        finally:
            inner_ids = jt.core._end_step_backward_probe(inner)
            outer_ids = jt.core._end_step_backward_probe(outer)
        assert inner_ids and inner_ids == outer_ids
        np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), [1., 4.])

    def test_failed_warmup_does_not_admit_an_unchecked_capture(self, device):
        model = _Model(device)
        calls = []
        def fn(x):
            calls.append(1)
            result = model(x)
            if len(calls) == 1:
                result.backward()
                raise RuntimeError('warmup marker')
            return result
        run = torch.compile(fn, mode='reduce-overhead')
        x = torch.tensor([1., 4.], device=device)
        with pytest.raises(RuntimeError, match='warmup marker'):
            run(x)
        for _ in range(2):
            model.weight.grad = None
            run(x).backward()
            np.testing.assert_array_equal(model.weight.grad.cpu().numpy(), [1., 4.])
        assert len(calls) == 3
        assert run.refused is not None


instantiate_device_type_tests(TestCompileExternalAutograd, globals(), only_for=('cuda',))
