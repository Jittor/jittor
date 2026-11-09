"""A hooked root must retain Python output recording in its child calls."""
import numpy as np
import pytest
import torch
import jittor as jt
from jittor._core.module import Module

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_parent_output_recording_runs_once_per_forward(device):
    if device == 'cuda' and not jt.has_cuda:
        pytest.skip('CUDA unavailable')
    class Leaf(torch.nn.Module):
        def forward(self, x):
            return x * 2
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Leaf()
        def forward(self, x):
            seen = []
            original = self.layer.forward
            def recording(x):
                result = original(x)
                seen.append(result)
                return result
            self.layer.forward = recording
            try:
                result = self.layer(x)
            finally:
                self.layer.forward = original
            return result, len(seen)
    model = Parent()
    observations = []
    handle = model.register_forward_hook(lambda m, args, output: observations.append(output[1]))
    with jt.flag_scope(auto_graph_replay=1), torch.no_grad():
        for i in range(5):
            output, count = model(torch.tensor([float(i)], device=device))
            assert count == 1
            np.testing.assert_array_equal(output.cpu().numpy(), [2*i])
    handle.remove()
    assert observations == [1]*5
    assert Module._call_depth == 0

def test_exception_in_hooked_root_restores_call_depth():
    class Failing(torch.nn.Module):
        def forward(self, x):
            raise ValueError('expected forward failure')
    module = Failing()
    handle = module.register_forward_hook(lambda *args: None, always_call=True)
    with pytest.raises(ValueError, match='expected forward failure'):
        module(torch.ones(1, device='cpu'))
    handle.remove()
    assert Module._call_depth == 0
