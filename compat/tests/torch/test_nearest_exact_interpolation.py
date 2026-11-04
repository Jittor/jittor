"""Half-pixel nearest sampling, integer scale semantics and ACL reuse."""
import numpy as np
import pytest
import torch
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


def _indices(source, target):
    return np.floor((np.arange(target, dtype=np.float64)+.5)*source/target).astype(np.int64)


def _reference(values, size):
    return values[:, :, _indices(values.shape[2], size[0])[:, None], _indices(values.shape[3], size[1])[None, :]]


@pytest.mark.parametrize("size", [(5, 7), (2, 3), (6, 10)])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_nearest_exact_size_forward_and_weighted_gradient(size, dtype):
    values = np.arange(30, dtype=dtype).reshape(1, 2, 3, 5)/7
    expected = _reference(values, size)
    cotangent = np.random.default_rng(20261020).uniform(-.4, .4, expected.shape).astype(dtype)
    gradient = np.zeros(values.shape, dtype=np.float64)
    for row, iy in enumerate(_indices(values.shape[2], size[0])):
        for col, ix in enumerate(_indices(values.shape[3], size[1])):
            gradient[:, :, iy, ix] += cotangent[:, :, row, col]
    with jt.flag_scope(use_cuda=0, use_acl=0):
        x = torch.tensor(values, dtype=getattr(torch, dtype), device="cpu", requires_grad=True)
        result = torch.nn.functional.interpolate(x, size=size, mode="nearest-exact")
        actual_grad, = torch.autograd.grad((result*torch.tensor(cotangent, device="cpu")).sum(), (x,))
        assert result.dtype == x.dtype == actual_grad.dtype
        np.testing.assert_array_equal(result.detach().numpy(), expected)
        np.testing.assert_allclose(actual_grad.numpy(), gradient,
                                   rtol=3e-3 if dtype == "float16" else 1e-6,
                                   atol=1e-3 if dtype == "float16" else 1e-6)


def test_nearest_exact_differs_from_legacy_nearest_on_an_odd_output_size():
    values = np.arange(3, dtype=np.float32).reshape(1, 1, 3, 1)
    with jt.flag_scope(use_cuda=0, use_acl=0), torch.no_grad():
        x = torch.tensor(values, device="cpu")
        exact = torch.nn.functional.interpolate(x, size=(5, 1), mode="nearest-exact")
        legacy = torch.nn.functional.interpolate(x, size=(5, 1), mode="nearest")
        np.testing.assert_array_equal(exact.numpy().reshape(-1), [0, 0, 1, 2, 2])
        np.testing.assert_array_equal(legacy.numpy().reshape(-1), [0, 0, 1, 1, 2])


def test_half_pixel_index_product_does_not_overflow_for_a_long_skinny_image():
    values = np.arange(65536, dtype=np.float32).reshape(1, 1, 65536, 1)
    with jt.flag_scope(use_cuda=0, use_acl=0), torch.no_grad():
        output = torch.nn.functional.interpolate(torch.tensor(values, device="cpu"),
                                                 size=(65537, 1), mode="nearest-exact")
        np.testing.assert_array_equal(output.numpy(), _reference(values, (65537, 1)))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("factor", [2, (2, 3)])
def test_integer_scale_functional_and_upsample_module(dtype, factor):
    values = np.arange(24, dtype=dtype).reshape(1, 2, 3, 4)
    factors = (factor, factor) if isinstance(factor, int) else factor
    size = tuple(value*scale for value, scale in zip(values.shape[-2:], factors))
    with jt.flag_scope(use_cuda=0, use_acl=0), torch.no_grad():
        x = torch.tensor(values, dtype=getattr(torch, dtype), device="cpu")
        functional = torch.nn.functional.interpolate(x, scale_factor=factor, mode="nearest-exact")
        module = torch.nn.Upsample(scale_factor=factor, mode="nearest-exact")(x)
        np.testing.assert_array_equal(functional.numpy(), _reference(values, size))
        np.testing.assert_array_equal(module.numpy(), _reference(values, size))


@pytest.mark.parametrize("factor", [1.6, .5, (2, 1.5)])
def test_fractional_explicit_scale_is_not_silently_replaced_by_output_ratio(factor):
    with jt.flag_scope(use_cuda=0, use_acl=0):
        x = torch.ones((1, 1, 3, 4), device="cpu")
        with pytest.raises(NotImplementedError, match="integer scale_factor only"):
            torch.nn.functional.interpolate(x, scale_factor=factor, mode="nearest-exact")
        with pytest.raises(NotImplementedError, match="integer scale_factor only"):
            torch.nn.Upsample(scale_factor=factor, mode="nearest-exact")(x)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("strided", [False, True])
def test_acl_integer_two_scale_matches_reference_without_fallback(dtype, strided):
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required")
    values = np.arange(72, dtype=dtype).reshape(3, 2, 3, 4)
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1, use_acl=1, transpose_storage_view=1), torch.no_grad(), forbid_backend_fallbacks():
        if strided:
            x = torch.tensor(values.transpose(1, 0, 2, 3).copy(), dtype=getattr(torch, dtype), device="cuda").permute(1, 0, 2, 3)
            assert not x._storage_is_contiguous()
        else:
            x = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
        outputs = [torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest-exact"),
                   torch.nn.Upsample(scale_factor=2, mode="nearest-exact")(x)]
        jt.sync_all(True)
        for result in outputs:
            assert result.location() == "device" and result.dtype == x.dtype
            np.testing.assert_array_equal(result.cpu().numpy(), _reference(values, (6, 8)))
    assert jt.core.backend_fallback_count() == before
