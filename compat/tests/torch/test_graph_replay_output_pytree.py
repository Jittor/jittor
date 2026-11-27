import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.graph_replay import graph_replay
from _helpers.capability import require_accelerator


class State:
    def __init__(self, tensor):
        self.entries = {"cache": [tensor]}
        self.dtype = tensor.dtype
        self.device = tensor.device


class Registered:
    __slots__ = ("entries", "metadata")

    def __init__(self, tensor, metadata):
        self.entries = {"cache": [tensor]}
        self.metadata = metadata


torch.utils._pytree.register_pytree_node(
    Registered,
    lambda result: ([result.entries], result.metadata),
    lambda children, metadata: Registered(children[0]["cache"][0], metadata),
)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("entry", ["inference", "step"])
def test_replay_returns_fresh_custom_outputs_and_preserves_tensor_aliases(device, entry):
    if device == "cuda":
        require_accelerator("cuda")

    class Module(torch.nn.Module):
        def forward(self, source):
            tensor = source.square() + 2
            return {"registered": Registered(tensor, ("tag", 4)),
                    "state": State(tensor), "tensor": tensor}

    with torch.no_grad(), jt.flag_scope(auto_graph_replay=0, backend_fallback="error"):
        module = Module().eval()
        call = (graph_replay(module) if entry == "inference"
                else torch.compile(module.forward, mode="reduce-overhead"))
        history = []
        for number in range(7):
            source = torch.full((2, 3), float(number), device=device)
            result = call(source)
            history.append(result)
            jt.sync([result["tensor"]], device_sync=(device == "cuda"))

        assert call.refused is None
        assert call.stats["replayed"] > 0
        if device == "cuda" and entry == "step":
            assert call.stats["graph"] > 0, call.graph_refused
        assert len({id(result["state"]) for result in history}) == len(history)
        assert len({id(result["registered"]) for result in history}) == len(history)

        for number, result in enumerate(history):
            tensor = result["tensor"]
            assert isinstance(result["registered"], Registered)
            assert result["registered"].entries["cache"][0] is tensor
            assert result["registered"].metadata == ("tag", 4)
            assert isinstance(result["state"], State)
            assert result["state"].entries["cache"][0] is tensor
            assert result["state"].dtype is torch.float32
            assert result["state"].device == tensor.device
            np.testing.assert_array_equal(tensor.cpu().numpy(), number**2 + 2)


def test_cyclic_output_refuses_capture():
    def forward(source):
        value = source + 1
        result = State(value)
        result.entries["self"] = result
        return result

    with torch.no_grad():
        call = torch.compile(forward, mode="reduce-overhead")
        result = call(torch.ones(2, device="cpu"))
        assert result.entries["self"] is result
        assert "cyclic result" in call.refused
