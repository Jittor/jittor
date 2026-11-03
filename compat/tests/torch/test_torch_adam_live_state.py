"""Adam counters remain mutable tensors across updates and checkpoint views.

The same cases are runnable in an independent genuine Torch interpreter by
calling the methods with ``cpu`` or ``cuda``. Recorded second-step values
come from Torch 2.8 after externally changing the counter from one to seven.
"""
import copy
import weakref

import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


def _cases():
    return ((torch.optim.Adam, {}),
            (torch.optim.AdamW, {"fused": False}),
            (torch.optim.AdamW, {"fused": True}))


def _step(parameter, optimizer, gradient, device):
    parameter.grad = torch.tensor(gradient, device=device)
    optimizer.step()


class TestAdamLiveState:
    def test_counter_identity_alias_and_mutation_affect_update(self, device):
        expected = ([.9837156534194946, -1.9850393533706665],
                    [.981896698474884, -1.982884407043457])
        for algorithm, options in _cases():
            p = torch.tensor([1., -2.], device=device, requires_grad=True)
            opt = algorithm([p], lr=.01, weight_decay=.1, **options)
            assert p not in opt.state
            _step(p, opt, [.25, -.5], device)
            counter = opt.state[p]["step"]
            assert torch.is_tensor(counter)
            assert counter.dtype == torch.float32 and counter.ndim == 0
            assert counter.device.type == (device if options.get("fused") else "cpu")
            assert opt.state[p]["step"] is counter
            assert opt.state_dict()["state"][0]["step"] is counter
            # .data has an independent autograd version. A host mirror based
            # only on the original tensor's version would silently miss this.
            counter.data.fill_(7)
            _step(p, opt, [.75, .125], device)
            assert opt.state[p]["step"] is counter
            assert counter.item() == 8
            np.testing.assert_allclose(p.detach().cpu().numpy(),
                                       expected[int(algorithm is torch.optim.AdamW)],
                                       atol=1e-6, rtol=1e-6)
            opt.zero_grad(set_to_none=True)
            opt.step()
            assert counter.item() == 8

    def test_zero_replacement_and_deletion_keep_counter_semantics(self, device):
        for algorithm, options in _cases():
            p = torch.tensor([1., -2.], device=device, requires_grad=True)
            opt = algorithm([p], lr=.01, **options)
            _step(p, opt, [.25, -.5], device)
            old = opt.state[p]["step"]
            old.zero_()
            assert p in opt.state and opt.state[p]["step"] is old
            assert opt.state_dict()["state"][0]["step"] is old
            replacement = torch.tensor(4., device=old.device)
            opt.state[p]["step"] = replacement
            assert opt.state[p]["step"] is replacement
            opt.step()
            assert replacement.item() == 5 and old.item() == 0
            del opt.state[p]
            assert p not in opt.state and replacement.item() == 5
            opt.step()
            assert opt.state[p]["step"].item() == 1
            assert opt.state[p]["step"] is not replacement

    def test_checkpoint_counter_is_live_after_load_and_parameter_reorder(self, device):
        for algorithm, options in _cases():
            p = torch.tensor([1.], device=device, requires_grad=True)
            q = torch.tensor([2.], device=device, requires_grad=True)
            opt = algorithm([p, q], lr=.01, **options)
            _step(p, opt, [.25], device)
            assert q not in opt.state
            snapshot = copy.deepcopy(opt.state_dict())
            _step(q, opt, [.5], device)
            opt.load_state_dict(snapshot)
            assert q not in opt.state
            counter = opt.state[p]["step"]
            assert counter is snapshot["state"][0]["step"]
            assert opt.state_dict()["state"][0]["step"] is counter
            counter.fill_(6)
            opt.param_groups[0]["params"] = [q, p]
            assert opt.state[p]["step"] is counter
            assert q not in opt.state
            opt.step()
            assert opt.state[p]["step"] is counter and counter.item() == 7
            assert opt.state[q]["step"].item() == 1
            assert snapshot["state"][0]["step"].item() == 7

    def test_new_group_inherits_counter_device_policy(self, device):
        for algorithm, options in _cases():
            first = torch.tensor([1.], device=device, requires_grad=True)
            late = torch.tensor([2.], device=device, requires_grad=True)
            optimizer = algorithm([first], lr=.01, **options)
            optimizer.add_param_group({"params": [late]})
            _step(late, optimizer, [.25], device)
            assert first not in optimizer.state
            counter = optimizer.state[late]["step"]
            assert counter.device.type == (device if options.get("fused") else "cpu")
            assert counter.item() == 1

    def test_retaining_counter_does_not_retain_optimizer_parameters(self, device):
        for algorithm, options in _cases():
            p = torch.tensor([1.], device=device, requires_grad=True)
            optimizer = algorithm([p], lr=.01, **options)
            _step(p, optimizer, [.25], device)
            counter = optimizer.state[p]["step"]
            parameter_ref = weakref.ref(p)
            del p, optimizer
            assert parameter_ref() is None
            assert counter.item() == 1


instantiate_device_type_tests(TestAdamLiveState, globals())
