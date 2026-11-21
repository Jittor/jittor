"""Canonical CPU grid sampling and public kernel-selection seam."""
import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from _helpers.grid_sample_reference import bilinear_grid_sample


@pytest.mark.parametrize("padding", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align", [False, True])
def test_cpu_grid_sample_keeps_geometry_after_dispatch(padding, align):
    values = np.arange(40, dtype=np.float32).reshape(1, 2, 4, 5) / 32
    grid = np.asarray([[[[-1., -1.], [1., 1.], [.25, -.5]],
                        [[0., 0.], [-1.25, 1.25], [3.5, -3.5]]]], dtype=np.float32)
    with jt.runtime.scope(use_cuda=0), jt.no_grad():
        output = jt.nn.grid_sample(jt.array(values), jt.array(grid),
                                   padding_mode=padding, align_corners=align)
        np.testing.assert_allclose(output.numpy(), bilinear_grid_sample(values, grid, padding, align),
                                   rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_nearest_grid_sample_rounds_half_coordinates_to_even(dimensions, backend):
    if backend == "cuda" and not _test_capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires CUDA")
    values = np.array([10, 20, 30, 40], dtype=np.float32).reshape((1,) * (dimensions + 1) + (4,))
    grid = np.zeros((1,) * dimensions + (5, dimensions), dtype=np.float32)
    grid[..., 0] = [-1., -0.5, 0., 0.5, 1.]
    with jt.runtime.scope(use_cuda=int(backend == "cuda")), jt.no_grad():
        result = jt.nn.grid_sample(jt.array(values), jt.array(grid), mode="nearest", align_corners=False)
        np.testing.assert_array_equal(result.numpy().reshape(-1), [10, 10, 30, 30, 0])


def test_grid_sample_forwards_arguments_once_to_selected_kernel(monkeypatch):
    from jittor.nn.functional import grid as module
    selected, calls = [], []
    sentinel = object()
    def kernel(*args):
        calls.append(args)
        return sentinel
    def select(name, *args):
        selected.append((name, args))
        return kernel
    monkeypatch.setattr(module, "select_kernel", select)
    with jt.runtime.scope(use_cuda=0):
        x, positions = jt.ones((1, 2, 3, 4)), jt.zeros((1, 2, 2, 2))
        assert module.grid_sample(x, positions, "bilinear", "border", True) is sentinel
    expected = (x, positions, "bilinear", "border", True)
    assert selected == [("nn.grid_sample", expected)] and calls == [expected]
