"""310P inference bilinear resize against independent coordinate equations."""

import numpy as np
import pytest

import jittor as jt

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels import neural
from jittor.backends.acl.kernels._soc import soc_name


@pytest.fixture(autouse=True)
def acl_device():
    if not capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    if not soc_name().startswith("Ascend310P"):
        pytest.skip("bilinear inference admission is validated on Ascend310P only")
    before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, use_acl=1, transpose_storage_view=1), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


def reference(values, size, align_corners):
    def coordinates(source, target):
        if align_corners:
            point = np.arange(target, dtype=np.float64) * ((source - 1) / (target - 1) if target > 1 else 0.)
        else:
            point = np.maximum((np.arange(target, dtype=np.float64) + .5) * source / target - .5, 0.)
        left = np.floor(point).astype(np.int64)
        return left.clip(0, source - 1), (left + 1).clip(0, source - 1), point - left
    y0, y1, fy = coordinates(values.shape[2], size[0])
    x0, x1, fx = coordinates(values.shape[3], size[1])
    values = values.astype(np.float64)
    rows = values[:, :, y0, :] * (1 - fy)[None, None, :, None] + values[:, :, y1, :] * fy[None, None, :, None]
    return rows[:, :, :, x0] * (1 - fx)[None, None, None, :] + rows[:, :, :, x1] * fx[None, None, None, :]


@pytest.mark.parametrize("shape,size", [
    ((2, 3, 4, 5), (9, 7)), ((1, 64, 19, 19), (37, 37)),
    ((1, 64, 37, 37), (74, 74)), ((2, 3, 9, 7), (3, 4)),
    ((1, 2, 4, 5), (4, 5)), ((1, 2, 1, 3), (7, 1)),
])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_public_bilinear_resize(shape, size, dtype, align_corners, strided):
    values = np.random.RandomState(20261029).uniform(-1, 1, shape).astype(dtype)
    expected = reference(values, size, align_corners)
    with jt.no_grad():
        x = jt.array(values)
        if strided:
            x = jt.core.ops.transpose(jt.array(values.swapaxes(2, 3).copy()), (0, 1, 3, 2))
        actual = jt.nn.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)
        actual.sync(True)
        assert actual.location() == "device" and str(actual.dtype) == dtype
        np.testing.assert_allclose(actual.numpy(), expected,
                                   rtol=2e-3 if dtype == "float16" else 3e-5,
                                   atol=1e-3 if dtype == "float16" else 3e-5)


def test_bilinear_declines_other_soc_grad_and_tf_coordinates(monkeypatch):
    x = jt.ones((1, 2, 3, 4)).float16()
    with jt.no_grad():
        assert neural.resize_acl(x, (5, 7), "bilinear", True, True) is None
        for name in ("Ascend910B3", "Ascend910", ""):
            monkeypatch.setattr(neural, "soc_name", lambda: name)
            assert neural.resize_acl(x, (5, 7), "bilinear", True, False) is None
        monkeypatch.setattr(neural, "soc_name", lambda: "Ascend310P3")
        assert neural.resize_acl(jt.ones((1, 2, 3, 4), dtype="int32"), (5, 7), "bilinear") is None
    with jt.enable_grad():
        assert neural.resize_acl(x, (5, 7), "bilinear", True, False) is None


def test_nearest_and_bicubic_paths_remain_available_after_bilinear():
    values = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) / 16
    with jt.no_grad():
        x = jt.array(values)
        bilinear = jt.nn.interpolate(x, size=(6, 8), mode="bilinear", align_corners=True)
        nearest = jt.nn.interpolate(x, size=(6, 8), mode="nearest")
        bicubic = jt.nn.interpolate(x, size=(3, 4), mode="bicubic", align_corners=False)
        jt.sync([bilinear, nearest, bicubic], True)
        for actual in (bilinear, nearest, bicubic):
            assert actual.location() == "device" and str(actual.dtype) == "float32"
        np.testing.assert_array_equal(nearest.numpy(), values.repeat(2, 2).repeat(2, 3))
        np.testing.assert_allclose(bicubic.numpy(), values, rtol=1e-6, atol=1e-6)
