"""complex128 is a supported Torch frontend dtype end to end (CPU).

The core computes complex128 natively, but the frontend dtype registry used to
list it as a metadata-only placeholder, so every constructor and cast refused
it at the dtype gate. These tests pin the frontend surface the registry gates:
the dtype object and its reporting, every factory spelling, casts through
``.to``/``.type``, and a small compute path. complex32 stays metadata-only; the
placeholder boundary contract itself is covered by
``test_dtype_object_boundaries.py``.
"""
import numpy as np
import pytest
import torch

import jittor as jt
from jittor.compat.torch.types import dtype

C128 = torch.complex128
C64 = torch.complex64


def _dtype_name(value):
    return jt.core._checked_dtype_name(value.dtype)


# ------------------------------------------------------------------ factories

_FACTORIES = {
    "empty": lambda: torch.empty((2,), dtype=C128),
    "empty_like": lambda: torch.empty_like(torch.tensor([1.0]), dtype=C128),
    "zeros": lambda: torch.zeros((2,), dtype=C128),
    "zeros_like": lambda: torch.zeros_like(torch.tensor([1.0]), dtype=C128),
    "ones": lambda: torch.ones((2,), dtype=C128),
    "ones_like": lambda: torch.ones_like(torch.tensor([1.0]), dtype=C128),
    "full": lambda: torch.full((2,), 1 + 2j, dtype=C128),
    "full_like": lambda: torch.full_like(torch.tensor([1.0]), 1 + 2j, dtype=C128),
    "eye": lambda: torch.eye(2, dtype=C128),
    "arange": lambda: torch.arange(2, dtype=C128),
    "linspace": lambda: torch.linspace(0.0, 1.0, 2, dtype=C128),
    "rand": lambda: torch.rand((2,), dtype=C128),
    "randn": lambda: torch.randn((2,), dtype=C128),
    "rand_like": lambda: torch.rand_like(torch.tensor([1.0, 2.0]), dtype=C128),
    "randn_like": lambda: torch.randn_like(torch.tensor([1.0, 2.0]), dtype=C128),
    "hann_window": lambda: torch.hann_window(2, dtype=C128),
    "kaiser_window": lambda: torch.kaiser_window(2, dtype=C128),
    "sparse_coo_tensor": lambda: torch.sparse_coo_tensor(
        [[0]], [1.0], (2,), dtype=C128),
    "tensor": lambda: torch.tensor([1.0], dtype=C128),
    "as_tensor": lambda: torch.as_tensor([1.0], dtype=C128),
    "from_numpy": lambda: torch.from_numpy(np.array([1.0], dtype=np.complex128)),
    "new_empty": lambda: torch.tensor([1.0]).new_empty((2,), dtype=C128),
    "new_zeros": lambda: torch.tensor([1.0]).new_zeros((2,), dtype=C128),
    "new_ones": lambda: torch.tensor([1.0]).new_ones((2,), dtype=C128),
    "new_full": lambda: torch.tensor([1.0]).new_full((2,), 1 + 2j, dtype=C128),
    "new_tensor": lambda: torch.tensor([1.0]).new_tensor([1 + 2j], dtype=C128),
}


@pytest.mark.parametrize("factory", sorted(_FACTORIES))
def test_each_factory_constructs_complex128(factory):
    result = _FACTORIES[factory]()
    if factory == "sparse_coo_tensor":
        # The dense-backed COO wrapper reports its dense Var's native dtype.
        assert _dtype_name(result) == "complex128"
    else:
        assert result.dtype is C128
        assert _dtype_name(result) == "complex128"


def test_factory_values_are_complex():
    np.testing.assert_array_equal(torch.zeros(2, dtype=C128).numpy(), [0j, 0j])
    np.testing.assert_array_equal(torch.ones(2, dtype=C128).numpy(), [1 + 0j] * 2)
    np.testing.assert_array_equal(torch.full((2,), 1 + 2j, dtype=C128).numpy(),
                                  [1 + 2j] * 2)
    np.testing.assert_array_equal(torch.arange(2, dtype=C128).numpy(), [0j, 1 + 0j])
    np.testing.assert_array_equal(torch.linspace(0.0, 1.0, 3, dtype=C128).numpy(),
                                  [0j, 0.5 + 0j, 1 + 0j])
    np.testing.assert_array_equal(torch.eye(2, dtype=C128).numpy(),
                                  np.eye(2, dtype=np.complex128))
    np.testing.assert_array_equal(torch.hann_window(2, dtype=C128).numpy(),
                                  [0j, 1 + 0j])
    np.testing.assert_array_equal(
        torch.sparse_coo_tensor([[0]], [1 + 2j], (2,), dtype=C128).to_dense().numpy(),
        [1 + 2j, 0j])
    # A real fill converts to a complex value, not by reinterpreting storage.
    np.testing.assert_array_equal(torch.tensor([1.0, 2.5], dtype=C128).numpy(),
                                  [1 + 0j, 2.5 + 0j])
    np.testing.assert_array_equal(
        torch.tensor([1.0]).new_full((2,), 3 + 4j, dtype=C128).numpy(),
        [3 + 4j] * 2)


def test_complex_random_factories_draw_two_real_parts():
    # torch defines complex rand/randn as independent real draws for the real
    # and imaginary parts over the requested shape.
    uniform = torch.rand((256,), dtype=C128).numpy()
    assert uniform.dtype == np.complex128
    assert np.all((uniform.real >= 0) & (uniform.real < 1))
    assert np.all((uniform.imag >= 0) & (uniform.imag < 1))
    normal = torch.randn((256,), dtype=C128).numpy()
    assert np.all(np.isfinite(normal))
    assert np.any(normal.imag != 0)
    like = torch.rand_like(torch.ones(8, dtype=C128)).numpy()
    assert like.dtype == np.complex128 and like.shape == (8,)
    # A seeded generator is honored through the composed draw.
    left, right = torch.Generator(), torch.Generator()
    left.manual_seed(2026)
    right.manual_seed(2026)
    np.testing.assert_array_equal(
        torch.rand(8, dtype=C128, generator=left).numpy(),
        torch.rand(8, dtype=C128, generator=right).numpy())


# --------------------------------------------------------------- casts + type

def test_to_and_type_cast_to_complex128():
    base = torch.tensor([1.0, 2.5], dtype=torch.float64)
    expected = np.array([1 + 0j, 2.5 + 0j])
    for got in (
        base.to(C128),
        base.to(dtype="complex128"),
        base.cast(C128),
        base.cast("complex128"),
        base.type(C128),
        base.type(torch.ComplexDoubleTensor),
        base.type("torch.ComplexDoubleTensor"),
    ):
        assert got.dtype is C128
        np.testing.assert_array_equal(got.numpy(), expected)
    narrow = torch.tensor([1 + 2j, 3 - 4j], dtype=C64)
    widened = narrow.to(C128)
    assert widened.dtype is C128
    np.testing.assert_array_equal(widened.numpy(),
                                  [1 + 2j, 3 - 4j])


def test_type_reporting_names_the_complex_typed_tensor():
    wide = torch.tensor([1 + 2j], dtype=C128)
    narrow = torch.tensor([1 + 2j], dtype=C64)
    assert wide.type() == "torch.ComplexDoubleTensor"
    assert narrow.type() == "torch.ComplexFloatTensor"
    assert isinstance(wide, torch.ComplexDoubleTensor)
    assert isinstance(narrow, torch.ComplexFloatTensor)
    assert not isinstance(wide, torch.ComplexFloatTensor)
    assert not isinstance(narrow, torch.ComplexDoubleTensor)
    built = torch.ComplexDoubleTensor([1 + 2j])
    assert built.dtype is C128
    np.testing.assert_array_equal(built.numpy(), [1 + 2j])


# ------------------------------------------------------------- dtype identity

def test_dtype_reporting_round_trip():
    assert "complex128" in dtype._supported
    assert "complex32" not in dtype._supported
    value = torch.tensor([1 + 2j], dtype=C128)
    assert value.dtype is C128
    assert isinstance(value.dtype, dtype)
    assert str(value.dtype) == "torch.complex128"
    assert repr(value.dtype) == "torch.complex128"
    assert bool(C128.is_complex) is True
    assert C128.is_complex() is True
    assert bool(C128.is_floating_point) is False
    assert C128.itemsize == 16 == C128.element_size
    assert value.is_complex() is True
    assert value.is_floating_point() is False
    assert jt.core._checked_dtype_name(C128) == "complex128"
    assert jt.core._checked_dtype_name(str(C128)) == "complex128"
    rebuilt = jt.array(np.array([1 + 2j], dtype=np.complex128), dtype=C128)
    # jt.array returns the native Var, whose dtype is the core dtype object;
    # the canonical name is what both spellings agree on.
    assert _dtype_name(rebuilt) == "complex128"
    np.testing.assert_array_equal(rebuilt.numpy(), [1 + 2j])


def test_complex32_stays_metadata_only():
    with pytest.raises(NotImplementedError, match="complex32"):
        torch.empty((2,), dtype=torch.complex32)


# -------------------------------------------------------------- compute path

def test_arithmetic_and_real_imag_path():
    a_np = np.array([1 + 2j, 3 - 4j])
    b_np = np.array([2 + 1j, -1 + 1j])
    a = torch.tensor(a_np, dtype=C128)
    b = torch.tensor(b_np, dtype=C128)
    assert (a + b).dtype is C128
    np.testing.assert_allclose((a + b).numpy(), a_np + b_np)
    assert (a * b).dtype is C128
    np.testing.assert_allclose((a * b).numpy(), a_np * b_np)
    assert (a * 2).dtype is C128
    np.testing.assert_allclose((a * 2).numpy(), a_np * 2)
    magnitude = torch.abs(a)
    assert magnitude.dtype is torch.float64
    np.testing.assert_allclose(magnitude.numpy(), np.abs(a_np))
    for part, expected in ((torch.real(a), a_np.real), (torch.imag(a), a_np.imag),
                           (a.real, a_np.real), (a.imag, a_np.imag)):
        assert part.dtype is torch.float64
        np.testing.assert_allclose(part.numpy(), expected)
    assert torch.conj(a).dtype is C128
    np.testing.assert_allclose(torch.conj(a).numpy(), np.conj(a_np))
    assert torch.exp(a).dtype is C128
    np.testing.assert_allclose(torch.exp(a).numpy(), np.exp(a_np), rtol=1e-12)
