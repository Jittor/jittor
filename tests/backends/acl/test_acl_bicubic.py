"""ACL inference bicubic resize against independent half-pixel cubic math."""
import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def _reference(array, size):
    def coordinates(source, target):
        position = (np.arange(target, dtype=np.float64) + .5) * source / target - .5
        index = np.floor(position).astype(np.int64)[:, None] + np.arange(-1, 3)[None]
        d = np.abs(position[:, None]-index)
        a = -.75
        weight = np.where(d <= 1, ((a+2)*d-(a+3))*d*d+1,
                          np.where(d < 2, ((a*d-5*a)*d+8*a)*d-4*a, 0))
        return index.clip(0, source-1), weight
    array = array.astype(np.float64)
    iy, wy = coordinates(array.shape[2], size[0])
    ix, wx = coordinates(array.shape[3], size[1])
    vertical = np.zeros(array.shape[:2]+(size[0], array.shape[3]))
    for tap in range(4):
        vertical += np.take(array, iy[:, tap], axis=2)*wy[:, tap].reshape(1, 1, -1, 1)
    output = np.zeros(array.shape[:2]+tuple(size))
    for tap in range(4):
        output += np.take(vertical, ix[:, tap], axis=3)*wx[:, tap].reshape(1, 1, 1, -1)
    return output


@pytest.mark.parametrize("shape,size", [
    ((2, 3, 4, 5), (9, 7)),
    ((1, 2, 9, 7), (3, 4)),
    ((1, 96, 7, 7), (256, 256)),
])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_bicubic_inference_matches_independent_reference(shape, size, dtype):
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    values = np.random.default_rng(20261020).uniform(-1, 1, shape).astype(dtype)
    expected = _reference(values, size)
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.no_grad(), forbid_backend_fallbacks():
        result = jt.nn.interpolate(jt.array(values), size=size, mode="bicubic", align_corners=False)
        result.sync(True)
        assert result.location() == "device"
        assert str(result.dtype).replace("torch.", "") == dtype
        actual = result.numpy()
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected,
                                   rtol=2e-3 if dtype == "float16" else 3e-5,
                                   atol=2e-3 if dtype == "float16" else 3e-5)
    assert jt.core.backend_fallback_count() == before


def test_bicubic_inference_kernel_declines_training_and_other_sampling_modes():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    from jittor.backends.acl.kernels.neural import resize_acl
    from jittor.backends.acl.kernels.ops.upsample_op import UpsampleBicubic2dACL
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.enable_grad():
        x = jt.ones((1, 2, 3, 4))
        assert resize_acl(x, (5, 6), mode="bicubic") is None
        with pytest.raises(RuntimeError, match="requires no_grad"):
            UpsampleBicubic2dACL()(x, (5, 6))
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.no_grad():
        assert resize_acl(x, (5, 6), mode="bicubic", align_corners=True) is None
        assert resize_acl(x, (5, 6), mode="bicubic", tf_mode=True) is None
