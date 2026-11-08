"""Torch dtype spellings must obey the 310P FP32 cube opt-in guard."""
import pytest
import torch
import jittor as jt
from jittor.backends.acl.kernels import neural


@pytest.mark.parametrize('dtype,allowed,expected', [
    (torch.float32, False, False), (torch.float32, True, True),
    (torch.float16, False, True), (torch.float16, True, True),
])
def test_frontend_dtype_obeys_transpose_convolution_precision_guard(monkeypatch, dtype, allowed, expected):
    monkeypatch.setattr(neural, 'soc_name', lambda: 'Ascend310P3')
    monkeypatch.setattr(neural, 'allow_reduced_precision', lambda: allowed)
    with jt.runtime.scope(use_cuda=0, use_acl=0), torch.no_grad():
        source = torch.ones((1, 2, 3, 4), dtype=dtype)
        weight = torch.ones((2, 3, 2, 2), dtype=dtype)
        assert str(source.dtype).startswith('torch.')
        assert neural.conv_transpose2d_supported(
            source, weight, None, (2, 2), (0, 0), (0, 0), (1, 1), 1) is expected
