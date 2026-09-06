"""A transposed view must see a later in-place assign to its source.

`5.03`'s acceptance is spelled out as one line: ``at = a.transpose();
a.assign(0); matmul(at, b)`` must give 0. Today it does not -- ``matmul``
consumes the hidden transpose flag that ``transpose()`` leaves on the returned
Var and computes against the pre-assign contents, so the caller silently gets
a stale answer rather than an error.

The marker is ``strict``: when the storage model of `5.02` lands and this
starts passing, the suite fails until someone removes the marker. An ordinary
xfail would let a fixed defect keep reporting as "expected failure" forever,
which is the same shape as the gates that stayed green while observing nothing
(see the handoff, "门禁绿着，但不是因为它通过了").
"""

import numpy as np
import pytest

import jittor as jt


def _stale_transpose_case():
    a = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    b = jt.array(np.ones((3, 5), dtype="float32"))
    at = a.transpose()
    a.assign(jt.zeros((3, 4), "float32"))
    return jt.matmul(at, b).numpy()


@pytest.mark.xfail(strict=True, reason="5.03: transpose keeps a hidden flag, "
                                      "so matmul reads the pre-assign source")
def test_a_transposed_view_sees_a_later_assign_to_its_source():
    # Reference semantics, verified against real torch 2.12.1:
    #   a = torch.arange(12.).reshape(3, 4); at = a.t(); a.zero_(); at @ b
    # gives all zeros. Jittor currently returns the pre-assign product, whose
    # first row is 0 + 4 + 8 = 12 for every column of a ones matrix.
    np.testing.assert_array_equal(_stale_transpose_case(), np.zeros((4, 5),
                                                                    "float32"))


def test_the_stale_result_is_the_pre_assign_product():
    """Pin what is actually returned today, so the defect cannot drift quietly.

    Without this, a change that made the result stale in some *other* way --
    uninitialised memory, a partially applied assign -- would still leave the
    xfail above red and look like no change at all.
    """
    got = _stale_transpose_case()
    source = np.arange(12, dtype="float32").reshape(3, 4)
    expected_stale = source.T @ np.ones((3, 5), dtype="float32")
    np.testing.assert_allclose(got, expected_stale, rtol=0, atol=0)


def test_transpose_then_matmul_without_an_assign_is_correct():
    """The path itself is fine; only the invalidation is missing."""
    a = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    b = jt.array(np.ones((3, 5), dtype="float32"))
    got = jt.matmul(a.transpose(), b).numpy()
    np.testing.assert_allclose(got, a.numpy().T @ b.numpy(), rtol=1e-6,
                               atol=1e-6)
