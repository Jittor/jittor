import numpy as np
import pytest

import jittor as jt
from _helpers import capability


@pytest.mark.parametrize("align_corners", [False, True])
def test_bilinear_zero_padding_with_strided_inputs(align_corners):
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("requires a CUDA runtime")

    image = np.arange(60, dtype=np.float32).reshape(1, 3, 5, 4) / 7
    points = np.array([[[[-1.3, -0.9], [0.0, 0.25], [1.0, 1.0]],
                        [[1.7, 0.0], [-1.0, -1.0], [np.nan, 0.0]]]], dtype=np.float32)
    with jt.flag_scope(use_cuda=1), jt.no_grad():
        source = jt.array(image)._storage_permute((0, 3, 1, 2))
        grid = jt.array(points.transpose(0, 2, 1, 3).copy())._storage_permute((0, 2, 1, 3))
        assert not source._storage_is_contiguous()
        assert not grid._storage_is_contiguous()
        expected = jt.nn.grid_sampler(source, grid, "bilinear", "zeros", align_corners)
        actual = jt.nn.grid_sample(source, grid, align_corners=align_corners)
        assert actual._producer_name() == "code"
        jt.sync([actual, expected], True, False)
        assert actual.location() == expected.location() == "device"
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
