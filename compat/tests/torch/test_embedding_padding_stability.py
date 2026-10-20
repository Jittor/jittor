"""Public Torch embedding padding preserves forward data and masked gradients."""
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability


DEVICES = ["cpu"] + (["cuda"] if capability.check_accelerator("cuda", backend=jt).enabled else [])


def array(value):
    return value.detach().float().cpu().numpy().copy()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("padding", [0, -1])
def test_module_padding_nonzero_forward_and_first_second_grad(device, padding):
    initial = np.arange(35, dtype=np.float32).reshape(7, 5) / 32 + .125
    raw = np.resize(np.array([0, 2, 2, 6, 6, 0, 1], dtype=np.int64), 385).reshape(1, -1)
    normalized_padding = padding % 7
    active = raw != normalized_padding
    counts = np.bincount(raw[active], minlength=7)[:, None]
    with jt.runtime.scope(use_cuda=int(device == "cuda"), backend_fallback="error"):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        with forbid_backend_fallbacks():
            # Load nonzero padding data through the ordinary public copy path.
            # Padding masks its derivative; forward must preserve the data.
            module = torch.nn.Embedding(7, 5, padding_idx=padding, device=device)
            with torch.no_grad():
                module.weight.copy_(torch.tensor(initial, dtype=torch.float32, device=device))
            assert module.padding_idx == normalized_padding
            indices = torch.tensor(raw, dtype=torch.int64, device=device)
            contexts = (False, True) if device == "cuda" else (False,)
            for use_amp in contexts:
                with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
                    output = module(indices)
                    first, = torch.autograd.grad(output.square().sum(), module.weight, create_graph=True)
                second, = torch.autograd.grad(first.sum(), module.weight)
                assert output.dtype == torch.float32
                assert first.dtype == torch.float32 and second.dtype == torch.float32
                assert output.device.type == device
                assert jt.core.dispatch_context([output, first, second])[0] == device
                np.testing.assert_array_equal(array(output), initial[raw])
                np.testing.assert_array_equal(array(first), 2 * counts * initial)
                np.testing.assert_array_equal(array(second), np.broadcast_to(2 * counts, initial.shape))


@pytest.mark.parametrize("invalid", [7, -8])
def test_module_padding_invalid_boundaries_still_raise(invalid):
    with pytest.raises(AssertionError, match="within num_embeddings"):
        torch.nn.Embedding(7, 5, padding_idx=invalid)
