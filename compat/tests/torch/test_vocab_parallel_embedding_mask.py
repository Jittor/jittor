"""vLLM TP=2 vocabulary masking with changing token IDs and batch shapes.

The expression matches vLLM 0.24 get_masked_input_and_mask, including the
empty added-vocabulary range used by Qwen3. No vLLM installation is needed.
Set JITTOR_VOCAB_TEST_DEVICE=cuda for real CUDA execution.
"""
import os

import numpy as np
import pytest
import torch


def get_masked_input_and_mask(input_, org_vocab_start_index, org_vocab_end_index,
                              num_org_vocab_padding, added_vocab_start_index,
                              added_vocab_end_index):
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
    added_offset = (added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index)
                    - num_org_vocab_padding)
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


class ShardedEmbedding(torch.nn.Module):
    def __init__(self, rank, device):
        super().__init__()
        self.rank = rank
        self.local_size = 75968
        self.weight = torch.nn.Parameter(torch.arange(
            self.local_size * 4, dtype=torch.float32, device=device).reshape(self.local_size, 4))

    def forward(self, tokens):
        masked_input, input_mask = get_masked_input_and_mask(
            tokens, self.rank * self.local_size, (self.rank + 1) * self.local_size,
            0, 151936, 151936)
        output = torch.nn.functional.embedding(masked_input.long(), self.weight)
        output.masked_fill_(input_mask.unsqueeze(-1), 0)
        return output, masked_input, input_mask


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('reuse_input', [False, True])
def test_vocab_mask_embedding_tracks_input_changes(rank, replay, dtype, reuse_input):
    import jittor as jt

    device = os.environ.get('JITTOR_VOCAB_TEST_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('JITTOR_VOCAB_TEST_DEVICE must be cpu or cuda')
    if device == 'cuda' and not jt.has_cuda:
        pytest.skip('CUDA unavailable')
    boundary = np.asarray([0, 75967, 75968, 92807, 92999, 151935], dtype=np.int64)
    # Repeated fixed shapes reach capture/replay, then shrinking/growing
    # shapes exercise invalidation and return to a previously seen shape.
    sizes = [6] * 6 + [4, 4, 4, 3, 3, 3, 1, 1, 1, 6, 6, 6]
    with jt.flag_scope(use_cuda=int(device == 'cuda'), auto_graph_replay=int(replay)):
        module = ShardedEmbedding(rank, device).eval()
        storage = torch.empty(6, dtype=dtype, device=device)
        with torch.no_grad():
            for step, size in enumerate(sizes):
                values = np.roll(boundary, step % len(boundary))[:size].copy()
                value = torch.tensor(values, dtype=dtype, device=device)
                if reuse_input:
                    storage[:size].copy_(value)
                    value = storage[:size]
                output, masked, mask = module(value)
                # Materialize embedding first: reading the mask first could
                # hide a fusion issue by splitting the graph before indexing.
                actual = output.numpy()
                expected_mask = (values < rank * 75968) | (values >= (rank + 1) * 75968)
                local = np.where(expected_mask, 0, values - rank * 75968)
                expected = local[:, None] * 4 + np.arange(4, dtype=np.int64)
                expected[expected_mask] = 0
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(masked.numpy(), local)
                np.testing.assert_array_equal(mask.numpy(), expected_mask)
                assert np.all((masked.numpy() >= 0) & (masked.numpy() < 75968))
        # Automatic mode may conservatively refuse capture for this module;
        # correctness must hold both when replay runs and when it falls back.


@pytest.mark.parametrize('size', [1, 4])
@pytest.mark.parametrize('scalar', [75968, 151936, 256, -1, .5])
@pytest.mark.parametrize('reflected', [False, True])
def test_bool_scalar_multiplication_promotes_before_computing(size, scalar, reflected):
    import jittor as jt

    device = os.environ.get('JITTOR_VOCAB_TEST_DEVICE', 'cpu')
    if device == 'cuda' and not jt.has_cuda:
        pytest.skip('CUDA unavailable')
    with jt.flag_scope(use_cuda=int(device == 'cuda')):
        values = [True] if size == 1 else [True, False, True, False]
        mask = torch.tensor(values, dtype=torch.bool, device=device)
        output = scalar * mask if reflected else mask * scalar
        expected_dtype = torch.get_default_dtype() if isinstance(scalar, float) else torch.int64
        assert output.dtype == expected_dtype
        np.testing.assert_array_equal(output.numpy(), np.asarray(values) * scalar)
