"""Generic sampler typing and dense tensor capability queries."""
import typing
import pytest
import torch


def test_sampler_generic_can_be_subclassed():
    alias = torch.utils.data.Sampler[int]
    assert typing.get_origin(alias) is torch.utils.data.Sampler
    assert typing.get_args(alias) == (int,)

    class Sampler(alias):
        def __iter__(self):
            return iter([2, 0])

    assert list(Sampler()) == [2, 0]


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
def test_ordinary_tensor_is_not_quantized(dtype):
    value = torch.zeros((2,), dtype=dtype)
    assert value.is_quantized is False
