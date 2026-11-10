"""CPU concatenation inside a CUDA module's construction scope."""
from contextlib import contextmanager

import numpy as np
import pytest
from _helpers import capability as _test_capability


@contextmanager
def _cuda_placement(jt):
    before = jt.core._current_tensor_placement()
    token = jt.core._set_tensor_placement(1, 0)
    construction = jt.core._set_construction_placement(1, 0)
    try:
        yield
    finally:
        jt.core._restore_construction_placement(construction)
        jt.core._reset_tensor_placement(token)
    assert jt.core._current_tensor_placement() == before


@pytest.mark.parametrize('op,expected', [
    ('cat', [1., 2., 3., 4.]),
    ('stack', [[1., 2.], [3., 4.]]),
])
def test_cpu_inputs_inside_cuda_context_keep_device_and_gradients(op, expected):
    import jittor as jt
    import torch
    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('CUDA runtime unavailable')
    with jt.flag_scope(use_cuda=1, auto_flush_ops=0):
        a = torch.tensor([1., 2.], device='cpu', requires_grad=True)
        b = torch.tensor([3., 4.], device='cpu', requires_grad=True)
        with _cuda_placement(jt):
            result = getattr(torch, op)([a, b])
        da, db = torch.autograd.grad(result.sum(), (a, b))
        jt.sync_all(True)
        for value in (a, b, result, da, db):
            assert value.device.type == 'cpu'
            assert value.location() == 'cpu'
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(da.numpy(), [1., 1.])
        np.testing.assert_array_equal(db.numpy(), [1., 1.])
        device_tensor = torch.tensor([5., 6.], device='cuda')
        with _cuda_placement(jt), pytest.raises(RuntimeError, match='same backend and device'):
            getattr(torch, op)([a, device_tensor])
