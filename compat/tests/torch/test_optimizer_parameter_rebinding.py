"""Public optimizer groups may replace or reorder their parameter objects."""
import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestOptimizerParameterRebinding:
    def test_replace_uninitialized_group_with_flat_parameter(self, device):
        old = torch.tensor([[1., 2.], [3., 4.]], device=device, requires_grad=True)
        optimizer = torch.optim.AdamW([old], lr=.1, weight_decay=0, foreach=False)
        flat = torch.tensor([1., 2., 3., 4., 5., 6., 7.], device=device, requires_grad=True)
        optimizer.param_groups[0]["params"] = [flat]
        flat.grad = torch.tensor([1., -1., 2., -2., 3., -3., 4.], device=device)
        optimizer.step()
        np.testing.assert_allclose(flat.detach().cpu().numpy(), [0.9, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9], rtol=1e-6)
        assert tuple(optimizer.state[flat]["exp_avg"].shape) == (7,)
        np.testing.assert_allclose(optimizer.state[flat]["exp_avg"].cpu().numpy(), [.1, -.1, .2, -.2, .3, -.3, .4], rtol=1e-6)
        np.testing.assert_array_equal(old.detach().cpu().numpy(), [[1., 2.], [3., 4.]])

    def test_reordered_parameters_preserve_their_own_moments(self, device):
        first = torch.tensor([1., 2.], device=device, requires_grad=True)
        second = torch.tensor([3., 4., 5.], device=device, requires_grad=True)
        optimizer = torch.optim.Adam([first, second], lr=.1, foreach=False)
        first.grad = torch.tensor([1., 2.], device=device)
        second.grad = torch.tensor([3., 4., 5.], device=device)
        optimizer.step()
        first_moment = optimizer.state[first]["exp_avg"]
        second_moment = optimizer.state[second]["exp_avg"]
        optimizer.param_groups[0]["params"] = [second, first]
        first.grad = torch.tensor([2., 4.], device=device)
        second.grad = torch.tensor([6., 8., 10.], device=device)
        optimizer.step()
        assert optimizer.state[first]["exp_avg"] is first_moment
        assert optimizer.state[second]["exp_avg"] is second_moment
        np.testing.assert_allclose(first_moment.cpu().numpy(), [.29, .58], rtol=1e-6)
        np.testing.assert_allclose(second_moment.cpu().numpy(), [.87, 1.16, 1.45], rtol=1e-6)
        assert float(optimizer.state[first]["step"]) == 2
        assert float(optimizer.state[second]["step"]) == 2


instantiate_device_type_tests(TestOptimizerParameterRebinding, globals(), only_for=("cpu", "cuda"))
