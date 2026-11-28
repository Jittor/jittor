"""Projection grouping preserves Linear storage and reads live parameters."""

import jittor as jt
import numpy as np
import pytest
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_shared_input_linears_replay_and_recheck_weights(device, has_bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(52)
    with torch.no_grad(), forbid_backend_fallbacks(), jt.flag_scope(auto_graph_replay=0):
        first = torch.nn.Linear(256, 256, bias=has_bias).to(device)
        second = torch.nn.Linear(256, 320, bias=has_bias).to(device)

        def forward(value):
            return first(value), second(value)

        run = torch.compile(forward, mode="reduce-overhead")

        def check(value):
            expected = forward(value)
            actual = run(value)
            for output, reference in zip(actual, expected):
                assert output.device.type == device
                assert output.is_contiguous()
                np.testing.assert_allclose(output.cpu().numpy(), reference.cpu().numpy(),
                                           rtol=1e-4, atol=2e-5)

        for _ in range(7):
            value = torch.tensor(rng.normal(size=(2, 17, 256)), dtype=torch.float32, device=device)
            check(value)
        if device == "cuda":
            assert run.stats["graph"] > 0
            assert any(holder.var_ptr == first.weight.var_ptr
                       for holder, _, _ in (run._capture.folded_leaves or ()))
        first.weight.mul_(2)
        check(value)
        if device == "cuda":
            assert run.last_invalidated == "a folded constant leaf was replaced or mutated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_weight_tensors_passed_to_the_step_remain_live():
    from torch.nn import functional as F

    with torch.no_grad(), forbid_backend_fallbacks():
        x = torch.randn(35, 256, device="cuda")
        weight = torch.randn(256, 256, device="cuda")
        other = torch.randn(256, 256, device="cuda")

        @torch.compile(mode="reduce-overhead")
        def forward(value, left, right):
            return F.linear(value, left) + F.linear(value, right)

        for i in range(7):
            expected = F.linear(x, weight + i) + F.linear(x, other)
            actual = forward(x, weight + i, other)
            np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(),
                                       rtol=1e-4, atol=2e-4)
        assert forward.stats["graph"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_linear_discovery_limits_dense_output_copies():
    from jittor.nn.functional._parallel_linear import LinearDiscovery

    with torch.no_grad(), forbid_backend_fallbacks():
        first = torch.ones(1024, 256, dtype=torch.float32, device="cuda")
        second = torch.zeros_like(first)
        for rows, should_group in ((512, True), (513, False)):
            value = torch.ones(rows, 256, dtype=torch.float32, device="cuda")
            plan = LinearDiscovery([value])
            plan(value, first, None)
            plan(value, second, None)
            assert bool(plan.groups()) == should_group
