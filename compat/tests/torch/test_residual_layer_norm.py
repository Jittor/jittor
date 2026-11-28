import jittor as jt
import numpy as np
import pytest
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("clipped", [False, True])
@pytest.mark.parametrize("device, dtype", [
    ("cpu", torch.float32),
    ("cuda", torch.float32),
    ("cuda", torch.float16),
    ("cuda", torch.bfloat16),
])
def test_layer_norm_preserves_the_residual(clipped, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(618)
    shape = (2, 65, 384)
    left, right = (torch.tensor(rng.normal(size=shape), device=device, dtype=dtype)
                   for _ in range(2))
    norm = torch.nn.LayerNorm(shape[-1]).to(device=device, dtype=dtype).eval()

    def residual(a, b):
        value = a + b
        return value.clamp(-1.25, 0.875) if clipped else value

    def forward(a, b):
        value = residual(a, b)
        normalized = norm(value)
        if device == "cuda" and jt.flags.keep_graph == 2:
            assert value._producer_op() == "code"
        return normalized, value

    with torch.no_grad(), forbid_backend_fallbacks():
        original = residual(left, right)
        exact = original.float().cpu().numpy()
        expected = norm(original).float().cpu().numpy()
        with jt.flag_scope(auto_graph_replay=0):
            run = torch.compile(forward, mode="reduce-overhead")
            for _ in range(6):
                normalized, value = run(left, right)
                jt.sync([normalized, value])
            if device == "cuda":
                assert run.stats["graph"] > 0
            np.testing.assert_array_equal(value.float().cpu().numpy(), exact)
            np.testing.assert_allclose(normalized.float().cpu().numpy(), expected,
                                       atol=0.015 if dtype == torch.bfloat16 else 0.002,
                                       rtol=0.005)

        # A pre-existing view must continue sharing the original residual.
        with jt.flag_scope(keep_graph=2, auto_graph_replay=0):
            value = residual(left, right)
            view = value.view(-1, shape[-1])
            normalized = norm(value)
            value.add_(0.5)
            np.testing.assert_array_equal(view.float().cpu().numpy().reshape(shape),
                                          value.float().cpu().numpy())
            np.testing.assert_allclose(normalized.float().cpu().numpy(), expected,
                                       atol=0.015 if dtype == torch.bfloat16 else 0.002,
                                       rtol=0.005)
