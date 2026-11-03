"""Python comparison protocol remains distinct from explicit Tensor APIs."""
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64', 'bool'])
@pytest.mark.parametrize('values', [[1], [0, 1, 2]])
@pytest.mark.parametrize('kind', ['str', 'bytes', 'opaque', 'none', 'list', 'dtype'])
def test_comparison_dunders_decline_nonnumeric_sentinels(dtype, values, kind):
    tensor = torch.tensor(values, dtype=getattr(torch, dtype))
    other = {'str': 'Rep', 'bytes': b'Rep', 'opaque': object(),
             'none': None, 'list': [1], 'dtype': torch.float32}[kind]
    assert tensor.__eq__(other) is NotImplemented
    assert tensor.__ne__(other) is NotImplemented
    assert (tensor == other) is False
    assert (tensor != other) is True
    assert (other == tensor) is False
    assert (other != tensor) is True
    for operation in (lambda: tensor.eq(other), lambda: tensor.ne(other),
                      lambda: torch.eq(tensor, other), lambda: torch.ne(tensor, other)):
        with pytest.raises(TypeError):
            operation()


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'int64', 'bool'])
def test_numeric_comparisons_remain_elementwise_tensors(dtype):
    selected = getattr(torch, dtype)
    values = np.array([0, 1, 2], dtype=dtype)
    tensor = torch.tensor(values, dtype=selected)
    for other in (1, 1.0, True, np.int32(1), np.float32(1), torch.tensor([1, 1, 1], dtype=selected)):
        expected = np.asarray([1, 1, 1], dtype=dtype) if torch.is_tensor(other) else other
        for operation, reference in (
            (lambda: tensor == other, values == expected),
            (lambda: tensor != other, values != expected),
            (lambda: tensor.eq(other), values == expected),
            (lambda: tensor.ne(other), values != expected),
            (lambda: torch.eq(tensor, other), values == expected),
            (lambda: torch.ne(tensor, other), values != expected),
        ):
            result = operation()
            assert torch.is_tensor(result)
            assert result.dtype == torch.bool
            np.testing.assert_array_equal(result.cpu().numpy(), reference)


def test_comparison_dunders_allow_reflected_sentinel_implementation():
    class Sentinel:
        def __eq__(self, other):
            return 'reflected equality'
        def __ne__(self, other):
            return 'reflected inequality'
    tensor = torch.tensor([1., 2.])
    sentinel = Sentinel()
    assert tensor.__eq__(sentinel) is NotImplemented
    assert tensor.__ne__(sentinel) is NotImplemented
    assert (tensor == sentinel) == 'reflected equality'
    assert (tensor != sentinel) == 'reflected inequality'
