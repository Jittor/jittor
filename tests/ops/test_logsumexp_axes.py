"""``nn.logsumexp`` accepts several axes, the way torch does.

It reduced over them correctly and then handed the whole tuple to
``Var.squeeze``, which takes one axis, so the call died with
``TypeError: '<' not supported between instances of 'tuple' and 'int'`` --
found through ``einops.reduce``. The reference column is torch 2.13 on this
machine.

``Var.squeeze`` itself still takes one axis. Widening it to torch's tuple form
is a separate change and not a safe drop-in: torch's ``squeeze`` leaves rank 0
when every axis goes, while Jittor's falls back to ``[1]``, and callers that
then permute the result behave differently. See KI-SHAPE-001.
"""
import numpy as np
import pytest

import jittor as jt


def test_logsumexp_over_several_axes():
    x = jt.random([2, 3, 4])
    got = jt.nn.logsumexp(x, (0, 1, 2))
    ref = np.log(np.exp(x.numpy().astype("float64")).sum())
    assert got.numel() == 1, tuple(got.shape)
    np.testing.assert_allclose(float(got.item()), ref, rtol=1e-5, atol=1e-5)


def test_logsumexp_over_some_axes_drops_exactly_those():
    x = jt.random([2, 3, 4])
    got = jt.nn.logsumexp(x, (0, 1))
    assert tuple(got.shape) == (4,), tuple(got.shape)
    ref = np.log(np.exp(x.numpy().astype("float64")).sum(axis=(0, 1)))
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-5, atol=1e-5)


def test_logsumexp_negative_axes():
    x = jt.random([2, 3, 4])
    got = jt.nn.logsumexp(x, (-1, -3))
    assert tuple(got.shape) == (3,), tuple(got.shape)
    ref = np.log(np.exp(x.numpy().astype("float64")).sum(axis=(2, 0)))
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-5, atol=1e-5)


def test_logsumexp_keepdims_is_unchanged():
    x = jt.random([2, 3, 4])
    kept = jt.nn.logsumexp(x, (0, 1), keepdims=True)
    assert tuple(kept.shape) == (1, 1, 4), tuple(kept.shape)


def test_logsumexp_single_axis_still_takes_an_int():
    x = jt.random([2, 3, 4])
    got = jt.nn.logsumexp(x, 1)
    assert tuple(got.shape) == (2, 4), tuple(got.shape)
