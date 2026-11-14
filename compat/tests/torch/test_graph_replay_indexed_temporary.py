"""Indexed window masks remain correct across automatic and explicit replay."""

import numpy as np
import pytest
import torch
import jittor as jt

from jittor._runtime.graph_replay import release_auto_replay


class WindowMask(torch.nn.Module):
    def forward(self, values):
        mask = torch.zeros((1, 4, 4, 1), device=values.device)
        intervals = (slice(0, -2), slice(-2, -1), slice(-1, None))
        count = 0
        for row in intervals:
            for column in intervals:
                mask[:, row, column, :] = count
                count += 1
        windows = mask.view(1, 2, 2, 2, 2, 1).permute(0, 1, 3, 2, 4, 5)
        windows = windows.contiguous().view(-1, 4)
        bias = windows.unsqueeze(1) - windows.unsqueeze(2)
        bias = bias.masked_fill(bias != 0, -100.).masked_fill(bias == 0, 0.)
        return values + bias


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.cuda)])
@pytest.mark.parametrize('mode', ['automatic', 'module', 'step'])
@jt.flag_scope(auto_graph_replay=1, backend_fallback='error')
def test_indexed_temporary_preserves_every_window_and_new_inputs(device, mode):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires actual CUDA')
    labels = np.array([[0, 0, 1, 2], [0, 0, 1, 2],
                       [3, 3, 4, 5], [6, 6, 7, 8]], dtype='float32')
    windows = labels.reshape(1, 2, 2, 2, 2, 1).transpose(0, 1, 3, 2, 4, 5).reshape(4, 4)
    bias = np.where(windows[:, None, :] == windows[:, :, None], 0., -100.)
    model = WindowMask().eval()
    runner = (model if mode == 'automatic' else torch.compile(
        model.forward if mode == 'step' else model, mode='reduce-overhead'))
    try:
        with torch.no_grad():
            for index in range(8):
                data = np.arange(64, dtype='float32').reshape(4, 4, 4) + index
                output = runner(torch.tensor(data, device=device))
                assert output.device.type == device
                np.testing.assert_array_equal(output.cpu().numpy(), data + bias)
        if mode == 'step':
            assert runner.refused is None
            assert runner.stats['replayed'] >= 4
            if device == 'cuda':
                assert runner.stats['graph'] > 0, runner._graph_refused
    finally:
        if mode == 'step':
            runner.invalidate()
        elif mode == 'module':
            runner.__dict__['_replay'].invalidate()
        release_auto_replay(model)
