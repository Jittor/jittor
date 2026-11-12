"""Contracts for test assertion helpers."""

import pytest

from _helpers.assertions import expect_error


def test_expect_error_checks_type_and_message_and_returns_the_error():
    def fail():
        raise ValueError("bad dimension 7")

    error = expect_error(fail, exc_type=ValueError, match=r"dimension \d+")
    assert isinstance(error, ValueError)


def test_expect_error_rejects_the_wrong_exception_type():
    with pytest.raises(AssertionError, match="expected ValueError"):
        expect_error(lambda: (_ for _ in ()).throw(TypeError("wrong")),
                     exc_type=ValueError, match="wrong")


def test_expect_error_rejects_the_wrong_message():
    with pytest.raises(AssertionError, match="did not match"):
        expect_error(lambda: (_ for _ in ()).throw(ValueError("actual")),
                     exc_type=ValueError, match="expected")


def test_expect_error_rejects_a_successful_call():
    with pytest.raises(AssertionError, match="did not raise"):
        expect_error(lambda: None, exc_type=ValueError, match="anything")


@pytest.mark.parametrize('name', [
    'float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128',
])
def test_tolerances_keep_the_declared_dtype_budget_for_frontend_names(name):
    import torch
    from _helpers.common import default_tolerances

    expected = default_tolerances(name)
    assert expected != (0., 0.)
    assert default_tolerances('torch.' + name) == expected
    assert default_tolerances(getattr(torch, name)) == expected


def test_complex_reference_precision_does_not_tighten_the_result_budget():
    import numpy as np
    from _helpers.common import default_tolerances

    assert default_tolerances('torch.complex64', np.dtype('complex128')) == (
        1e-5, 1.3e-6)
