import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class _TaggedOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {"lr": 0.25})

    @torch.no_grad()
    def step(self):
        state = self.state["block.lora"]
        state["count"] = state.get("count", 0) + 1
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.add_(param.grad, alpha=-group["lr"])


class TestOptimizerNamedState:
    def test_named_entries_and_parameter_entries_roundtrip(self, device):
        parameter = torch.nn.Parameter(torch.tensor([2., 3.], device=device))
        optimizer = _TaggedOptimizer([parameter])
        assert optimizer.state.get("block.lora") is None
        assert "block.lora" not in optimizer.state
        param_state = optimizer.state[parameter]
        param_state["step"] = 7
        param_state["buffers"] = [torch.ones(2, dtype=torch.float64)]
        optimizer.state["0"] = "named key"
        optimizer.state["history"] = torch.tensor([1.])
        parameter.square().sum().backward()
        optimizer.step()
        assert parameter.tolist() == [1., 1.5]
        doc = optimizer.state_dict()
        assert doc["state"]["block.lora"] == {"count": 1}
        assert doc["state"][0]["step"] == 7 and doc["state"]["0"] == "named key"
        assert not torch.is_tensor(next(iter(doc["state"])))

        other = torch.nn.Parameter(torch.tensor([1., 1.5], device=device))
        restored = _TaggedOptimizer([other])
        restored.state["obsolete"]
        restored.load_state_dict(doc)
        assert "obsolete" not in restored.state and restored.state["0"] == "named key"
        assert restored.state[other]["step"] == 7
        loaded_buffer = restored.state[other]["buffers"][0]
        assert loaded_buffer.device.type == device and loaded_buffer.dtype == other.dtype
        assert restored.state["history"].device.type == "cpu"
        other.square().sum().backward()
        restored.step()
        assert other.tolist() == [0.5, 0.75]
        assert restored.state["block.lora"]["count"] == 2
        del restored.state["history"]
        assert set(restored.state.keys()) == {other, "0", "block.lora"}

    def test_native_moments_and_auxiliary_state_are_distinct(self, device):
        parameter = torch.nn.Parameter(torch.tensor([2.0], device=device))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        optimizer.state["run"] = {"epoch": 3}
        optimizer.state["0"] = 17
        assert optimizer.state.setdefault("optional") is None and "optional" in optimizer.state
        parameter.square().sum().backward()
        optimizer.step()
        saved = optimizer.state_dict()
        assert "exp_avg" in saved["state"][0]
        assert saved["state"]["0"] == 17 and saved["state"]["run"]["epoch"] == 3
        restored = torch.optim.Adam([parameter], lr=0.01)
        with pytest.raises(ValueError, match="different number"):
            restored.load_state_dict({"state": {"run": 0}, "param_groups": []})
        assert "run" not in restored.state
        restored.load_state_dict(saved)
        assert torch.equal(restored.state[parameter]["exp_avg"], optimizer.state[parameter]["exp_avg"])
        assert restored.state["run"]["epoch"] == 3 and restored.state["0"] == 17


instantiate_device_type_tests(TestOptimizerNamedState, globals(), only_for=("cpu", "cuda"))
