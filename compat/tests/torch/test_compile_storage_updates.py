"""Completed compiled steps preserve state storage and take each update once."""
import numpy as np
import jittor as jt
import torch
from _helpers.device_types import instantiate_device_type_tests


def _run(device, optimizer):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
    alias = parameter.detach()
    view = alias.view(1, 2)
    pointer = parameter.data_ptr()
    if optimizer == 'sgd':
        opt = torch.optim.SGD([parameter], lr=.125, momentum=.5,
                              weight_decay=.25, foreach=False)
    else:
        opt = torch.optim.RMSprop([parameter], lr=.125, alpha=.5, eps=0.)

    def step(gradient):
        opt.zero_grad(set_to_none=True)
        loss = (parameter * gradient).sum()
        loss.backward()
        opt.step()
        return loss

    run = torch.compile(step, mode='reduce-overhead')
    expected = np.array([1., 2.], dtype=np.float32)
    state = np.zeros(2, dtype=np.float32)
    for index, scale in enumerate((1., .5, 2., .25, 1., .5)):
        grad = np.array([.25, .5], dtype=np.float32) * scale
        want_loss = np.sum(expected * grad)
        loss = run(torch.tensor(grad, device=device))
        np.testing.assert_allclose(float(loss), want_loss, rtol=1e-5, atol=1e-6)
        if optimizer == 'sgd':
            update = grad + .25 * expected
            state = update if index == 0 else .5 * state + update
            expected = expected - .125 * state
        else:
            state = .5 * state + .5 * grad * grad
            expected = expected - .125 * grad / np.sqrt(state)
        for value in (parameter.detach(), alias, view):
            np.testing.assert_allclose(value.cpu().numpy().reshape(-1), expected,
                                       rtol=1e-5, atol=1e-6)
            assert value.data_ptr() == pointer
        assert parameter.data_ptr() == pointer
    if optimizer == 'sgd':
        assert run.refused is None
        assert run.stats['replayed'] > 0
    else:
        assert run.refused is not None
        assert run.stats['replayed'] == 0


class TestCompileStorageUpdates:
    def test_replayed_sgd_keeps_storage_and_preupdate_loss(self, device):
        _run(device, 'sgd')

    def test_refused_rmsprop_commits_each_step_once(self, device):
        _run(device, 'rmsprop')

    def test_warmed_replay_preserves_a_pending_outside_forward(self, device):
        # An unrelated forward built between two replayed steps leaves its
        # backward pending; the next replay writes the parameter underneath
        # it. The backward must see the value the parameter held when the
        # forward was built, as eager does.
        parameter = torch.nn.Parameter(torch.zeros((2, 2), device=device))
        opt = torch.optim.SGD([parameter], lr=.125, foreach=False, fused=False)
        gradient = torch.full_like(parameter, .25)

        def step(gradient):
            opt.zero_grad(set_to_none=True)
            loss = (parameter * gradient).sum()
            loss.backward()
            opt.step()
            return loss

        run = torch.compile(step, mode='reduce-overhead')
        for _ in range(7):
            run(gradient).sync()
        assert run.refused is None
        assert run.stats['replayed'] > 0
        # Read the value back now: a device clone is lazy, and reading it at
        # assert time would see the parameter as the last replay left it.
        forward_time = parameter.detach().cpu().numpy().copy()
        outside = (parameter + 1).exp().sum()
        run(gradient).sync()
        opt.zero_grad(set_to_none=True)
        outside.backward()
        np.testing.assert_allclose(
            parameter.grad.cpu().numpy(), np.exp(forward_time + 1),
            rtol=1e-5, atol=1e-6)


instantiate_device_type_tests(TestCompileStorageUpdates, globals(), only_for=('cuda',))
