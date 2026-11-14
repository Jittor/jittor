"""Medium CUDA products keep ordinary Linear values and every derivative."""
import importlib
from unittest import mock

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("features", [(768, 768), (768, 3072)])
def test_medium_product_avoids_the_tuner_context_and_keeps_all_gradients(bias, features):
    if not torch.cuda.is_available() or not getattr(jt.compiler, "is_cuda", False):
        pytest.skip("CUDA kernel-selection policy requires an allocated CUDA build")
    rng = np.random.default_rng(73)
    cin, cout = features
    # A medium product above the former floor, below the tuned-route floor.
    # Binary fractions keep the independent FP64 oracle well-conditioned.
    values = rng.integers(-2, 3, (1024, cin)).astype("float32") / 32
    weights = rng.integers(-2, 3, (cout, cin)).astype("float32") / 32
    cotangent = rng.integers(-2, 3, (1024, cout)).astype("float32") / 32
    offset = rng.integers(-2, 3, (cout,)).astype("float32") / 32
    lt = importlib.import_module("jittor.backends.cuda.kernels.cublas.lt_linear_cuda")
    with jt.flag_scope(auto_graph_replay=0, backend_fallback="error",
                       float32_matmul_precision="highest"):
        layer = torch.nn.Linear(cin, cout, bias=bias).to("cuda")
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weights, device="cuda"))
            if bias:
                layer.bias.copy_(torch.tensor(offset, device="cuda"))
        x = torch.tensor(values, device="cuda", requires_grad=True)
        outer = torch.nn.Sequential(layer)
        with mock.patch.object(lt._LtLinearProduct, "apply",
                side_effect=AssertionError("Medium product entered the costly tuner context")):
            output = outer(x)
            target = (output * torch.tensor(cotangent, device="cuda")).sum()
            targets = [x, layer.weight] + ([layer.bias] if bias else [])
            gradients = torch.autograd.grad(target, targets)
            jt.sync_all(True)
        v, w, g = (a.astype("float64") for a in (values, weights, cotangent))
        expected = [v @ w.T + (offset if bias else 0), g @ w, g.T @ v]
        if bias:
            expected.append(g.sum(axis=0))
        for actual, reference in zip([output, *gradients], expected):
            actual.sync()
            assert actual.device.type == "cuda" and actual.location() == "device"
            np.testing.assert_allclose(actual.detach().cpu().numpy(), reference,
                                       rtol=1e-5, atol=1e-5)
