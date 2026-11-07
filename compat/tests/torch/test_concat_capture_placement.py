"""Graph-capture concatenation allocates helpers on the inputs' device."""
from unittest.mock import patch
import numpy as np
import pytest
import torch
import jittor as jt
from jittor.ops import concatenation

@pytest.mark.parametrize('input_device,ambient_device', [('cpu','cuda'), ('cuda','cpu')])
def test_fused_concat_helpers_ignore_conflicting_module_device(input_device, ambient_device):
    if not jt.has_cuda:
        pytest.skip('CUDA unavailable')
    class Join(torch.nn.Module):
        def forward(self, ambient):
            # Views select the fused capture path. The module has a
            # deliberately different device from these explicitly placed inputs.
            left = torch.arange(3, device=input_device).view(1,3)
            right = torch.arange(3, device=input_device).view(1,3)
            with patch.object(concatenation, '_tracing', return_value=True):
                return torch.cat([left,right], dim=0)
    output = Join()(torch.ones(1, device=ambient_device))
    output.sync()
    assert output.device.type == input_device
    assert output.location() == ('device' if input_device=='cuda' else 'cpu')
    np.testing.assert_array_equal(output.cpu().numpy(), [[0,1,2],[0,1,2]])
