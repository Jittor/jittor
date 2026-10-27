"""Custom Module state conversion and differentiable scalar accumulation."""
import operator

import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class _Stateful(torch.nn.Module):
    """Unregistered state is owned by the subclass's _apply protocol."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.register_buffer("count", torch.tensor([3], dtype=torch.int64))
        self.state = torch.tensor([4.0])
        self.history = [torch.tensor([5.0])]
        self.apply_count = 0

    def _apply(self, fn):
        self.apply_count += 1
        result = super()._apply(fn)
        self.state = fn(self.state)
        self.history = [fn(value) for value in self.history]
        return result


class TestCustomModuleState:
    def test_to_dispatches_to_root_and_child_override(self, device):
        root = _Stateful()
        root.child = _Stateful()
        root.child_alias = root.child
        parameters = list(root.parameters())
        assert root.to(device=device, dtype=torch.float64) is root
        for module in (root, root.child):
            assert module.apply_count == 1
            assert module.state.dtype == torch.float64
            assert module.history[0].dtype == torch.float64
            assert module.count.dtype == torch.int64
            for value in (module.state, module.history[0], module.count, module.weight):
                assert value.device.type == device
        assert all(old is new for old, new in zip(parameters, root.parameters()))
        assert all(value.requires_grad for value in parameters)
        result = root.weight * root.state + root.child.weight * root.child.state
        result.sum().backward()
        for value in parameters:
            np.testing.assert_array_equal(value.grad.cpu().numpy(), [4.0])

    def test_float_conversion_dispatches_override(self, device):
        root = _Stateful().to(device=device, dtype=torch.float64)
        root.child = _Stateful().to(device=device, dtype=torch.float64)
        root.apply_count = root.child.apply_count = 0
        assert root.float() is root
        for module in (root, root.child):
            assert module.apply_count == 1
            assert module.state.dtype == torch.float32
            assert module.history[0].dtype == torch.float32
            assert module.weight.dtype == torch.float32

    def test_apply_recurse_false_converts_only_immediate_tensors(self, device):
        root = torch.nn.Module()
        root.weight = torch.nn.Parameter(torch.tensor([2.0], device=device))
        root.child = _Stateful().to(device)
        root._apply(lambda value: value.to(dtype=torch.float64), recurse=False)
        assert root.weight.dtype == torch.float64
        assert root.child.weight.dtype == torch.float32
        assert root.child.state.dtype == torch.float32
        assert root.child.apply_count == 1

    def test_inplace_augmented_arithmetic_keeps_input_gradient(self, device):
        cases = ((operator.iadd, [2.0, 4.0]),
                 (operator.isub, [-2.0, -4.0]),
                 (operator.imul, [8.0, 16.0]),
                 (operator.itruediv, [-0.32, -0.64]))
        for operation, expected in cases:
            value = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
            state = torch.tensor(4.0, device=device)
            original = state
            state = operation(state, value.square().sum())
            assert state is original
            assert state.requires_grad
            state.backward()
            assert value.grad is not None
            np.testing.assert_allclose(value.grad.cpu().numpy(), expected, rtol=1e-6)

    def test_state_reduction_does_not_detach_cached_batch_result(self, device):
        prediction = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        target = torch.tensor([0.0, 0.0], device=device, requires_grad=True)
        state = torch.tensor(0.0, device=device)
        with torch.set_grad_enabled(True):
            state += (prediction - target).square().sum()
        batch = (state / 2).clone()
        with torch.no_grad():
            state = state + torch.tensor(3.0, device=device)
        assert not state.requires_grad
        batch.backward()
        assert prediction.grad is not None and target.grad is not None
        np.testing.assert_array_equal(prediction.grad.cpu().numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(target.grad.cpu().numpy(), [-1.0, -2.0])

    def test_detached_cloned_state_gains_input_gradient(self, device):
        value = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        state = torch.zeros(1, device=device).detach().clone()
        state += value.square().sum()
        assert state.requires_grad
        state.sum().backward()
        assert value.grad is not None
        np.testing.assert_array_equal(value.grad.cpu().numpy(), [2.0, 4.0])

    def test_augmented_arithmetic_under_no_grad_stays_frozen(self, device):
        value = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
        state = torch.tensor(0.0, device=device)
        with torch.no_grad():
            state += value.square().sum()
        assert not state.requires_grad
        leaf = torch.tensor(1.0, device=device, requires_grad=True)
        (state + leaf).backward()
        assert value.grad is None
        np.testing.assert_array_equal(leaf.grad.cpu().numpy(), 1.0)


instantiate_device_type_tests(TestCustomModuleState, globals(), only_for=("cpu", "cuda"))
