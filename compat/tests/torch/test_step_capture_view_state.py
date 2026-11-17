"""State written through physical views belongs to the captured root."""
import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('record', [False, True])
@torch.no_grad()
def test_replayed_indexed_writes_preserve_all_shared_storage_views(device, record):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')

    # A byte allocation can back several strided, reinterpreted cache views.
    shape = (28, 4, 16, 2, 16)
    storage = torch.zeros((int(np.prod(shape)) * 4,), dtype=torch.uint8, device=device)
    state = storage.view(torch.float32).reshape(shape)
    views = [state[layer].transpose(1, 2) for layer in range(shape[0])]
    jt.sync_all()
    expected = np.zeros(shape, dtype=np.float32)

    def write(values, slots):
        blocks, offsets = slots // 16, slots % 16
        for view in views:
            view[blocks, :, offsets, :] = values
            jt.sync([view])
        # State must be replayed even when the result does not read it.
        return values.sum()

    step = jt.capture_step(write, record=record)
    try:
        for index in range(12):
            slots = np.array([index % 16, 32 + index * 3 % 16,
                              48 + index * 7 % 16], dtype=np.int64)
            value = torch.full((3, 2, 16), float(index), device=device)
            actual = step(value, torch.tensor(slots, device=device))
            np.testing.assert_array_equal(actual.detach().cpu().numpy(), 96 * index)
            for layer in range(shape[0]):
                expected[layer, slots // 16, slots % 16, :, :] = index
            np.testing.assert_array_equal(state.detach().clone().cpu().numpy(), expected)
        assert step.refused is None
        assert step.stats['replayed'] >= 5
        if device == 'cuda' and record:
            assert step.stats['graph'] >= 2
    finally:
        step.invalidate()
