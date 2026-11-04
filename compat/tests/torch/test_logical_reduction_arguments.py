"""Public all/any retain Torch shape, truth, and byte-mask contracts.

RT-DETRv2 uses ``mask.all(-1, keepdim=True)`` before broadcasting its anchor
mask. Keep both public spellings aligned without adapting the model.
"""

import os

import numpy as np
import pytest
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.fixture
def device():
    with forbid_backend_fallbacks():
        yield os.environ.get("JITTOR_TEST_DEVICE", "cpu")


VALUES = np.array([[[0, 2, 4, 6], [1, 2, 3, 4], [0, 0, 0, 0]],
                   [[1, 0, 2, 3], [1, 1, 1, 1], [4, 3, 2, 0]]])


def _call(value, name, owner, *args, **kwargs):
    if owner == "method":
        return getattr(value, name)(*args, **kwargs)
    return getattr(torch, name)(value, *args, **kwargs)


def _check(result, expected, device, dtype):
    assert result.device.type == device.split(":", 1)[0]
    assert tuple(result.shape) == expected.shape
    assert result.dtype == (torch.uint8 if dtype == "uint8" else torch.bool)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), expected)


@pytest.mark.parametrize("name", ("all", "any"))
@pytest.mark.parametrize("owner", ("method", "module"))
@pytest.mark.parametrize("dtype", ("bool", "float32", "int64", "uint8"))
def test_logical_reduction_dimension_and_keepdim_spellings(device, name, owner, dtype):
    value = torch.tensor(VALUES, dtype=getattr(torch, dtype), device=device)
    reduce = getattr(np, name)
    for dim in (-1, (0, -1)):
        expected = reduce(VALUES, axis=dim, keepdims=True)
        for args, kwargs in (((dim,), {"keepdim": True}),
                             ((dim, True), {}),
                             ((), {"dim": dim, "keepdim": True}),
                             ((), {"axis": dim, "keepdims": True})):
            _check(_call(value, name, owner, *args, **kwargs), expected, device, dtype)


@pytest.mark.parametrize("name", ("all", "any"))
@pytest.mark.parametrize("owner", ("method", "module"))
@pytest.mark.parametrize("dtype", ("bool", "float32", "int64", "uint8"))
def test_logical_reduction_all_axes_and_empty_axes_differ(device, name, owner, dtype):
    value = torch.tensor(VALUES, dtype=getattr(torch, dtype), device=device)
    expected = getattr(np, name)(VALUES, keepdims=True)
    for kwargs in ({"keepdim": True}, {"dim": None, "keepdim": True}):
        _check(_call(value, name, owner, **kwargs), expected, device, dtype)
    for keepdim in (False, True):
        _check(_call(value, name, owner, (), keepdim), VALUES != 0, device, dtype)


@pytest.mark.parametrize("name", ("all", "any"))
@pytest.mark.parametrize("owner", ("method", "module"))
def test_logical_reduction_rejects_invalid_and_duplicate_axes(device, name, owner):
    value = torch.tensor(VALUES, device=device)
    for dim in (-4, 3, (0, 3)):
        with pytest.raises(IndexError, match="Dimension out of range"):
            _call(value, name, owner, dim, True)
    for dim in ((1, 1), (1, -2)):
        with pytest.raises(RuntimeError, match="appears multiple times"):
            _call(value, name, owner, dim)


@pytest.mark.parametrize("name", ("all", "any"))
@pytest.mark.parametrize("owner", ("method", "module"))
def test_logical_reduction_scalar_and_nonfinite_truth(device, name, owner):
    for number in (0., -0., -2., np.nan, np.inf):
        value = torch.tensor(number, device=device)
        for dim in (None, 0, -1, ()):
            for keepdim in (False, True):
                _check(_call(value, name, owner, dim, keepdim),
                       np.asarray(number != 0), device, "float32")


def test_reduced_anchor_mask_broadcasts_into_where(device):
    anchors = torch.tensor([[[.5, .25, .05, .05], [.5, .99, .05, .05]]], device=device)
    mask = ((anchors > .01) * (anchors < .99)).all(-1, keepdim=True)
    assert tuple(mask.shape) == (1, 2, 1)
    result = torch.where(mask, anchors, torch.tensor(torch.finfo(anchors.dtype).max, device=device))
    assert tuple(result.shape) == (1, 2, 4)
    assert result.device.type == device.split(":", 1)[0]
    expected = np.array([[[.5, .25, .05, .05], [np.finfo(np.float32).max] * 4]], np.float32)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), expected)
