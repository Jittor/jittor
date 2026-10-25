"""Optimizer writes keep retained Tensor storage and pending readers valid."""
import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestOptimizerStorageUpdates:
    def test_portable_sgd_keeps_aliases_and_published_gradients(self, device):
        for momentum, nesterov, dampening in ((0., False, 0.),
                                               (.9, False, .25), (.9, True, 0.)):
            p = torch.tensor([1., 2.], device=device, requires_grad=True)
            alias = p.detach()
            view = alias.view(1, 2)
            pointer = p.data_ptr()
            optimizer = torch.optim.SGD([p], lr=.125, momentum=momentum,
                                        nesterov=nesterov, dampening=dampening,
                                        weight_decay=.25, fused=False)
            expected = np.array([1., 2.], dtype="float64")
            velocity = None
            retained_momentum = None
            for step in range(2):
                optimizer.zero_grad()
                # Do not materialize this clone before the write: it is a
                # pending reader of the old storage in a lazy implementation.
                snapshot = p.detach().clone()
                before = expected.copy()
                gradient = np.array([.25, .5], dtype="float32") * (step + 1)
                (p * torch.tensor(gradient, device=device)).sum().backward()
                published = p.grad
                optimizer.step()
                update = gradient + .25 * before
                if momentum:
                    velocity = (update if velocity is None else
                                momentum * velocity + (1 - dampening) * update)
                    direction = update + momentum * velocity if nesterov else velocity
                    buffer = optimizer.state[p]["momentum_buffer"]
                    if retained_momentum is None:
                        retained_momentum = buffer.detach()
                    np.testing.assert_allclose(retained_momentum.cpu().numpy(), velocity,
                                               rtol=1e-6, atol=1e-7)
                else:
                    direction = update
                expected -= .125 * direction
                assert p.grad is published
                assert p.is_leaf and p.requires_grad
                assert p.data_ptr() == pointer == alias.data_ptr() == view.data_ptr()
                for tensor in (p, alias, view):
                    np.testing.assert_allclose(tensor.detach().cpu().numpy().ravel(),
                                               expected, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(snapshot.cpu().numpy(), before, rtol=1e-6)
                np.testing.assert_array_equal(published.cpu().numpy(), gradient)

    def test_strided_parameter_updates_its_retained_storage(self, device):
        p = torch.tensor([[1., 2.], [3., 4.]], device=device).t().detach().requires_grad_(True)
        alias = p.detach()
        pointer = p.data_ptr()
        # Isolate the storage-write contract from gradient initialization.
        p.grad = torch.full_like(p, 2.)
        torch.optim.SGD([p], lr=.125, fused=False).step()
        assert p.data_ptr() == pointer == alias.data_ptr()
        assert p.stride() == alias.stride() == (1, 2)
        expected = [[.75, 2.75], [1.75, 3.75]]
        for tensor in (p, alias):
            np.testing.assert_array_equal(tensor.detach().cpu().numpy(), expected)

    def test_other_optimizer_writes_use_the_same_storage_contract(self, device):
        for optimizer_type in (torch.optim.AdamW, torch.optim.RMSprop):
            p = torch.tensor([1., 2.], device=device, requires_grad=True)
            alias = p.detach()
            pointer = p.data_ptr()
            options = {"foreach": False} if optimizer_type is torch.optim.AdamW else {}
            optimizer = optimizer_type([p], lr=.125, weight_decay=0., **options)
            for _ in range(2):
                optimizer.zero_grad()
                p.square().sum().backward()
                optimizer.step()
                assert p.data_ptr() == pointer == alias.data_ptr()
                np.testing.assert_array_equal(p.detach().cpu().numpy(), alias.cpu().numpy())
                assert np.isfinite(p.detach().cpu().numpy()).all()
                assert p.is_leaf and p.requires_grad


instantiate_device_type_tests(TestOptimizerStorageUpdates, globals(), only_for=("cpu", "cuda"))
