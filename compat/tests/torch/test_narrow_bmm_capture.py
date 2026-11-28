import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator
from jittor._runtime.step_capture import tracing


def test_narrow_bmm_cpu():
    a = np.arange(2 * 17 * 65, dtype="float32").reshape(2, 17, 65) / 1000
    b = np.arange(2 * 65 * 19, dtype="float32").reshape(2, 65, 19) / 1000
    result = torch.bmm(torch.tensor(a, device="cpu"), torch.tensor(b, device="cpu"))
    assert result.device.type == "cpu"
    np.testing.assert_allclose(result.numpy(), a @ b, rtol=2e-6, atol=2e-6)


@pytest.mark.cuda
def test_narrow_bmm_cuda_graph_handles_partial_tiles():
    require_accelerator("cuda")
    rng = np.random.default_rng(249)
    a = rng.normal(size=(3, 23, 197)).astype("float32")
    b = rng.normal(size=(3, 197, 21)).astype("float32")
    expected = a.astype("float64") @ b.astype("float64")
    producers = []

    def product(a, b):
        result = torch.bmm(a, b)
        if tracing():
            producers.append(result._producer_name())
        return result

    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad(), jt.flag_scope(backend_fallback="error", auto_graph_replay=0):
            step = torch.compile(product, mode="reduce-overhead", dynamic=False)
            for scale in (1, 2):
                inputs = torch.tensor(a * scale, device="cuda"), torch.tensor(b, device="cuda")
                jt.sync(inputs, device_sync=True)
                for _ in range(6):
                    result = step(*inputs)
                    result.sync()
                assert result.is_cuda
                np.testing.assert_allclose(result.cpu().numpy(), expected * scale,
                                           rtol=1e-5, atol=2e-5)
            assert producers and set(producers) == {"code"}
            assert step.stats["graph"] > 0
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.cuda
@pytest.mark.parametrize("transposed,strided", [(False, False), (False, True), (True, True)])
def test_short_batched_matmul_with_partial_tiles(transposed, strided):
    require_accelerator("cuda")
    rng = np.random.default_rng(38)
    a = rng.standard_normal((3, 11, 37, 35)).astype(np.float32)
    b = rng.standard_normal((3, 11, 35, 53)).astype(np.float32)
    stored = b.swapaxes(-2, -1).copy() if transposed else b
    producers = []

    def product(a, rhs):
        if strided:
            a, rhs = a.transpose(1, 2), rhs.transpose(1, 2)
        result = torch.matmul(a, rhs.transpose(-2, -1) if transposed else rhs)
        if tracing():
            producers.append(result._producer_name())
        return result

    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad(), jt.flag_scope(backend_fallback="error", auto_graph_replay=0):
            step = torch.compile(product, mode="reduce-overhead", dynamic=False)
            arrays = a, stored
            if strided:
                arrays = tuple(value.swapaxes(1, 2).copy() for value in arrays)
            inputs = tuple(torch.tensor(value, device="cuda") for value in arrays)
            for _ in range(6):
                result = step(*inputs)
                result.sync()
            assert result.device.type == "cuda"
            assert step.stats["graph"] > 0
            assert producers and set(producers) == {"code"}
            np.testing.assert_allclose(result.cpu().numpy(), a.astype(np.float64) @ b,
                                       rtol=1e-5, atol=1e-5)
    finally:
        torch.set_float32_matmul_precision(precision)
