"""Coordinate counts used by Python must be guarded before replay."""
import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator


@pytest.mark.cuda
@pytest.mark.parametrize("record", [False, True])
def test_unique_count_and_masked_pooling_guard_optimizer_replay(record):
    require_accelerator("cuda")

    def make_step(weight, optimizer):
        def step(tokens, values):
            eos = tokens == 1
            if len(torch.unique_consecutive(eos.sum(1))) > 1:
                raise ValueError("Unequal EOS counts")
            pooled = (values * weight)[eos, :].view(tokens.shape[0], -1, 2)[:, -1, :]
            loss = pooled.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.detach()
        return step

    reference_weight = torch.nn.Parameter(torch.tensor([1., 2.], device="cuda"))
    candidate_weight = torch.nn.Parameter(reference_weight.detach().clone())
    reference_opt = torch.optim.AdamW([reference_weight], lr=.001)
    candidate_opt = torch.optim.AdamW([candidate_weight], lr=.001)
    eager = make_step(reference_weight, reference_opt)
    captured = jt.capture_step(make_step(candidate_weight, candidate_opt), record=record)
    values = torch.arange(48, dtype=torch.float32, device="cuda").reshape(3, 8, 2) / 48

    def feed(rows):
        return torch.tensor(rows, dtype=torch.int64, device="cuda")

    def compare(tokens):
        expected, actual = eager(tokens, values), captured(tokens, values)
        np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-6)
        np.testing.assert_allclose(candidate_weight.detach().cpu().numpy(),
                                   reference_weight.detach().cpu().numpy(), rtol=2e-6)

    with jt.flag_scope(backend_fallback="error"):
        for i in range(8):
            compare(feed([[int(j == (i + row) % 8) for j in range(8)] for row in range(3)]))
        assert captured.refused is None
        assert captured.stats["replayed"]
        if record:
            assert captured.stats["graph"], captured.graph_refused

        # The total stays at three so only the count-of-runs guard, rather
        # than the compaction size for masked indexing, sees this change.
        invalid = feed([[1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
        before = candidate_weight.detach().cpu().numpy().copy()
        with pytest.raises(ValueError, match="Unequal EOS counts"):
            captured(invalid, values)
        np.testing.assert_array_equal(candidate_weight.detach().cpu().numpy(), before)

        # The step can continue with a new valid number of positions.
        for i in range(8):
            compare(feed([[int((j + i + row) % 8 < 2) for j in range(8)] for row in range(3)]))
        opt_state = candidate_opt.state[candidate_weight]
        assert opt_state["step"].item() == 16
