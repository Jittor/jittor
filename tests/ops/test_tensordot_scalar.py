"""Full contraction produces a scalar without squeezing free singleton axes."""

import jittor as jt
import numpy as np
import pytest

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        capability.require_accelerator("cuda")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
@pytest.mark.parametrize(
    "shapes,dims,equation,output_shape",
    [
        pytest.param(((2, 3), (2, 3)), 2, "ij,ij->", (), id="full-integer"),
        pytest.param(
            ((2, 3), (3, 2)),
            ([0, 1], [1, 0]),
            "ij,ji->",
            (),
            id="full-explicit-permuted",
        ),
        pytest.param(
            ((1, 3), (3, 1)),
            1,
            "ik,kj->ij",
            (1, 1),
            id="free-singletons-integer",
        ),
        pytest.param(
            ((1, 2, 3), (2, 3, 1)),
            ([1, 2], [0, 1]),
            "ijk,jkl->il",
            (1, 1),
            id="free-singletons-explicit",
        ),
    ],
)
def test_tensordot_scalar_and_free_singletons(
    device, dtype, shapes, dims, equation, output_shape
):
    arrays = []
    for index, shape in enumerate(shapes):
        data = (np.arange(np.prod(shape)).reshape(shape) / 8 - 0.4 + index / 5).astype(
            dtype
        )
        if np.issubdtype(data.dtype, np.complexfloating):
            data.imag = 0.3 - data.real / 4 + index / 7
        arrays.append(data)

    a, b = (jt.array(data) for data in arrays)
    value = jt.nn.tensordot(a, b, dims=dims)
    loss = value.real.sum() if dtype.startswith("complex") else value.sum()
    gradient_a, gradient_b = jt.grad(loss, [a, b])

    expected_value = np.asarray(np.tensordot(*arrays, axes=dims))
    assert expected_value.shape == output_shape
    inputs, output = equation.split("->")
    subs_a, subs_b = inputs.split(",")
    cotangent = np.ones(output_shape, dtype=dtype)
    # For real(sum(tensordot(a, b))), complex gradients conjugate the other
    # operand. Contract the unit output cotangent without numerical differencing.
    expected_a = np.einsum(f"{output},{subs_b}->{subs_a}", cotangent, arrays[1].conj())
    expected_b = np.einsum(f"{subs_a},{output}->{subs_b}", arrays[0].conj(), cotangent)
    pairs = (
        (value, expected_value),
        (gradient_a, expected_a),
        (gradient_b, expected_b),
    )
    # Check residency before any host fetch, including other tensors in this graph.
    for actual, expected in pairs:
        assert tuple(actual.shape) == expected.shape
        assert str(actual.dtype) == dtype
        actual.sync()
        assert actual.location() == ("device" if device == "cuda" else "cpu")

    tolerance = 2e-12 if dtype == "complex128" else 2e-6
    for actual, expected in pairs:
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=tolerance, atol=tolerance
        )
