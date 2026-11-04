"""Actual 310P grid-sampling inference and admission boundaries."""
import numpy as np
import pytest
import jittor as jt

from _helpers import capability
from _helpers.grid_sample_reference import bilinear_grid_sample
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels import neural
from jittor.backends.acl.kernels._soc import soc_name


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    if not soc_name().startswith("Ascend310P"):
        pytest.skip("grid-sampling inference admission is validated on Ascend310P only")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("shape,grid_shape", [
    ((2, 3, 5, 7), (2, 4, 6, 2)),
    ((8, 32, 80, 80), (8, 300, 4, 2)),
    ((8, 32, 40, 40), (8, 300, 4, 2)),
    ((8, 32, 20, 20), (8, 300, 4, 2)),
    ((1, 2, 1, 5), (1, 3, 4, 2)),
    ((1, 2, 5, 1), (1, 3, 4, 2)),
])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("strided", [False, True])
def test_public_grid_sample_matches_independent_geometry(shape, grid_shape, dtype, strided):
    rng = np.random.RandomState(20261029)
    values = rng.uniform(-.5, .5, shape).astype(dtype)
    grid = rng.uniform(-1.3, 1.3, grid_shape).astype(dtype)
    grid.reshape(-1, 2)[:4] = [[-1., -1.], [1., 1.], [0., 0.], [-1.25, 1.25]]
    expected = bilinear_grid_sample(values, grid)
    with jt.no_grad():
        x, positions = jt.array(values), jt.array(grid)
        if strided:
            x = jt.core.ops.transpose(jt.array(values.swapaxes(2, 3).copy()), (0, 1, 3, 2))
            positions = jt.core.ops.transpose(jt.array(grid.swapaxes(1, 2).copy()), (0, 2, 1, 3))
        output = jt.nn.grid_sample(x, positions, mode="bilinear", padding_mode="zeros", align_corners=False)
        output.sync(True)
        assert output.location() == "device" and str(output.dtype) == dtype
        actual = output.numpy()
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=3e-3 if dtype == "float16" else 3e-5,
                                   atol=1e-3 if dtype == "float16" else 3e-5)


def test_grid_sample_declines_other_soc_grad_and_unverified_semantics(monkeypatch):
    x, grid = jt.ones((1, 2, 3, 4)).float16(), jt.ones((1, 2, 2, 2)).float16()
    supports = neural.grid_sample_supported
    with jt.no_grad():
        assert supports(x, grid)
        for name in ("Ascend910B3", "Ascend910", ""):
            monkeypatch.setattr(neural, "soc_name", lambda: name)
            assert not supports(x, grid)
        monkeypatch.setattr(neural, "soc_name", lambda: "Ascend310P3")
        for mode, padding, align in (("nearest", "zeros", False), ("bilinear", "border", False),
                                      ("bilinear", "reflection", False), ("bilinear", "zeros", True),
                                      ("bilinear", "zeros", None)):
            assert not supports(x, grid, mode, padding, align)
        assert not supports(x, grid.float32())
        assert not supports(x.int32(), grid.int32())
        assert not supports(x, jt.ones((2, 2, 2, 2)).float16())
        assert not supports(x, jt.ones((1, 2, 2, 3)).float16())
    with jt.enable_grad():
        assert not supports(x, grid)


def test_direct_inference_owner_rejects_training():
    from jittor.backends.acl.kernels.ops.grid_sample_op import GridSampler2DACL
    with jt.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        GridSampler2DACL()(jt.ones((1, 2, 3, 4)), jt.ones((1, 2, 2, 2)))
