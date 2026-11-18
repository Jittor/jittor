"""Captured indexed stores update storage and preserve earlier readouts."""
import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA storage writes are required')
@pytest.mark.parametrize('layout', ['contiguous', 'transposed'])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bfloat16'])
@torch.no_grad()
def test_captured_stores_copy_all_rows_and_preserve_pending_snapshots(layout, index_dtype, dtype):
    rng = np.random.default_rng(119)
    expected = rng.integers(-128, 128, size=(4, 2, 16, 16)).astype('float32') / 64
    if layout == 'contiguous':
        target = torch.tensor(expected.copy(), dtype=dtype, device='cuda')
    else:
        target = torch.tensor(expected.transpose(0, 2, 1, 3).copy(),
                              dtype=dtype, device='cuda').transpose(1, 2)
    jt.sync([target])
    accepted = []

    def write(values, blocks, offsets):
        direct = target._try_indexed_storage_write(values, [blocks, offsets], [0, 2])
        accepted.append(direct)
        if not direct:
            target[blocks, :, offsets, :] = values
        return values.sum()

    step = jt.capture_step(write)
    snapshots = []
    try:
        for index in range(9):
            snapshots.append((target.clone(), expected.copy()))
            blocks = np.array([0, -2, 3], dtype=np.int64)
            offsets = np.array([index % 16, (3 * index) % 16, -1], dtype=np.int64)
            values = rng.integers(-128, 128, size=(3, 2, 16)).astype('float32') / 64
            step(torch.tensor(values, dtype=dtype, device='cuda'),
                 torch.tensor(blocks, dtype=index_dtype, device='cuda'),
                 torch.tensor(offsets, dtype=index_dtype, device='cuda'))
            expected[blocks, :, offsets, :] = values
            np.testing.assert_array_equal(target.detach().clone().float().cpu().numpy(), expected)
        for snapshot, previous in snapshots:
            np.testing.assert_array_equal(snapshot.float().cpu().numpy(), previous)
        assert any(accepted), 'the CUDA storage primitive was not exercised'
        assert step.refused is None
        assert step.stats['graph'] >= 2
    finally:
        step.invalidate()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@torch.no_grad()
def test_host_assignment_uses_ordinary_capture_and_preserves_aliases(dtype):
    target = torch.zeros((4, 2, 16, 16), dtype=dtype, device='cpu')
    expected = np.zeros((4, 2, 16, 16), dtype=np.float64)
    accepted = []

    def write(value, slots):
        blocks, offsets = slots // 16, slots % 16
        accepted.append(target._try_indexed_storage_write(value, [blocks, offsets], [0, 2]))
        target[blocks, :, offsets, :] = value
        return value.sum()

    step = jt.capture_step(write, record=False)
    try:
        for index in range(5):
            slots = np.array([index, 32 + index], dtype=np.int64)
            value = torch.full((2, 2, 16), float(index + 1), dtype=dtype, device='cpu')
            step(value, torch.tensor(slots, device='cpu'))
            expected[slots // 16, :, slots % 16, :] = index + 1
            np.testing.assert_array_equal(target.detach().clone().numpy(), expected)
        assert not any(accepted)
        assert step.stats['replayed'] > 0
    finally:
        step.invalidate()
