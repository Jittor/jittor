"""Native CPU randn selection, replay and reference sequence boundaries."""
import numpy as np
import pytest
import torch
import jittor as jt
from jittor._core.var import ops as native_ops


@pytest.fixture(autouse=True)
def preserve_random_state():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize("explicit", [False, True])
def test_cpu_randn_scalar_reference_and_frontend(explicit):
    torch.manual_seed(42)
    kwargs = {"generator": torch.Generator().manual_seed(42)} if explicit else {}
    value = torch.randn(6, dtype=torch.float32, device="cpu", requires_grad=True, **kwargs)
    assert isinstance(value, torch.Tensor)
    assert value.device.type == "cpu" and value.is_leaf and value.requires_grad
    expected = np.array(
        [0.3366903663, 0.1288094074, 0.2344623655, 0.2303330302,
         -1.1228563786, -0.1863282919], dtype=np.float32)
    np.testing.assert_array_equal(value.detach().numpy(), expected)


@pytest.mark.parametrize("length, expected", [
    (16, [1.9269153, 1.4872841, 0.9007172, -2.105521, 0.67841846, -1.2345449]),
    (17, [1.9269153, -0.15959747, -0.49739754, 0.43958926, -0.75813115, 1.07831764]),
])
def test_cpu_randn_changes_to_block_generation_at_sixteen(length, expected):
    generator = torch.Generator().manual_seed(42)
    value = torch.randn(size=(length,), device="cpu", dtype=torch.float32, generator=generator)
    np.testing.assert_allclose(value.numpy()[:6], expected, rtol=0, atol=5e-7)


def test_snapshot_preserves_cached_normal_and_native_substream():
    torch.manual_seed(129)
    torch.randn(3, dtype=torch.float64, device="cpu").sync()
    state = torch.get_rng_state()

    def draw():
        return [value.numpy().copy() for value in (
            torch.randn(17, dtype=torch.float64, device="cpu"),
            torch.rand(13, device="cpu"),
            torch.randn_like(torch.zeros(3, dtype=torch.float64, device="cpu")),
        )]

    expected = draw()
    torch.set_rng_state(state)
    for actual, reference in zip(draw(), expected):
        np.testing.assert_array_equal(actual, reference)
    before = torch.get_rng_state().numpy().copy()
    with pytest.raises(RuntimeError):
        torch.set_rng_state(torch.tensor(before[:-20], dtype=torch.uint8, device="cpu"))
    np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)


def test_fork_rng_continues_cpu_normal_sequence():
    torch.manual_seed(71)
    torch.randn(3, device="cpu").sync()
    with torch.random.fork_rng(devices=[]):
        first = torch.randn(13, device="cpu").numpy().copy()
    np.testing.assert_array_equal(torch.randn(13, device="cpu").numpy(), first)


def test_native_compat_normal_refuses_an_unsupported_output_dtype():
    with jt.runtime.scope(use_cuda=0):
        with pytest.raises(RuntimeError, match="requires a CPU float32 or float64 output"):
            native_ops.random((3,), "float16", "normal", None, True)
