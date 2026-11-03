# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Gradients with respect to a transposed intermediate.
#
# `matmul` hands a GEMM the untransposed base of a transposed operand and a
# transpose flag, and `transpose` of a transpose is built from the source;
# both skip the transposed Var itself. The product then did not depend on it,
# so `jt.grad(loss, base.T)` came back zero with a missing-gradient warning
# while the base's gradient was right -- every dtype, either side, CPU and
# CUDA. The gradient must reach both, at first and second order.
import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


_TOL = {"float32": 3e-5, "float64": 1e-10, "complex64": 3e-5, "complex128": 1e-10}


def _operands(dtype, batch=()):
    a = (np.arange(np.prod(batch, dtype=int) * 6).reshape(batch + (2, 3)) * 0.07 - 0.23)
    b = (np.arange(np.prod(batch, dtype=int) * 12).reshape(batch + (3, 4)) * 0.07 - 0.12)
    if dtype.startswith("complex"):
        a = a + 0.3j * a[..., ::-1, :]
        b = b - 0.2j * b[..., :, ::-1]
    return a.astype(dtype), b.astype(dtype)


def _swap(x):
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return axes


def _operand(value, transposed, dtype):
    """``value`` as a leaf, or as the transpose of a leaf base."""
    if not transposed:
        leaf = jt.array(value, dtype=dtype)
        return leaf, leaf
    base = jt.array(np.ascontiguousarray(np.swapaxes(value, -1, -2)), dtype=dtype)
    return base.transpose(_swap(base)), base


def _loss(c):
    return (c * c.conj()).real.sum() if "complex" in str(c.dtype) else (c * c).sum()


@pytest.mark.parametrize("materialized", [False, True], ids=["lazy", "materialized"])
@pytest.mark.parametrize("storage_view", [0, 1])
@pytest.mark.parametrize("side", ["left", "right", "both"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("batch", [(), (3,)], ids=["2d", "batched"])
def test_gradient_reaches_the_transposed_operand_and_its_base(
        device, batch, dtype, side, storage_view, materialized):
    a_np, b_np = _operands(dtype, batch)
    with jt.flag_scope(transpose_storage_view=storage_view):
        a, a_base = _operand(a_np, side in ("left", "both"), dtype)
        b, b_base = _operand(b_np, side in ("right", "both"), dtype)
        if materialized:
            jt.sync([a, b])
        c = jt.matmul(a, b)
        bases = [v for v, w in ((a_base, a), (b_base, b)) if v is not w]
        with jt.flag_scope(missing_grad_error=1):
            grads = jt.grad(_loss(c), [a, b] + bases)
    cc = a_np @ b_np
    expected_a = 2 * cc @ np.conj(np.swapaxes(b_np, -1, -2))
    expected_b = 2 * np.conj(np.swapaxes(a_np, -1, -2)) @ cc
    expected_bases = [np.swapaxes(e, -1, -2)
                      for e, v, w in ((expected_a, a_base, a), (expected_b, b_base, b)) if v is not w]
    tol = _TOL[dtype]
    np.testing.assert_allclose(c.numpy(), cc, rtol=tol, atol=tol)
    for got, expected in zip(grads, [expected_a, expected_b] + expected_bases):
        np.testing.assert_allclose(got.numpy(), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_second_derivative_through_the_transposed_operand(device, dtype):
    # L = sum |A B|^2 with A = base^T, so dL/dA = 2 A M with M = B B^H.
    # The gradient of Re sum(dL/dA * V) for a fixed V is 2 conj(V) M, at A,
    # and its transpose at the base.
    a_np, b_np = _operands(dtype)
    v_np = np.linspace(-1, 1, a_np.size).reshape(a_np.shape)
    if dtype.startswith("complex"):
        v_np = v_np + 0.5j * v_np[::-1]
    v_np = v_np.astype(dtype)
    a, base = _operand(a_np, True, dtype)
    b = jt.array(b_np, dtype=dtype)
    ga = jt.grad(_loss(jt.matmul(a, b)), a)
    inner = (ga * jt.array(v_np, dtype=dtype)).real.sum() if dtype.startswith("complex") \
        else (ga * jt.array(v_np, dtype=dtype)).sum()
    with jt.flag_scope(missing_grad_error=1):
        ha, hbase = jt.grad(inner, [a, base])
    expected = 2 * np.conj(v_np) @ b_np @ np.conj(b_np.T)
    np.testing.assert_allclose(ha.numpy(), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(hbase.numpy(), expected.T, rtol=1e-10, atol=1e-10)


def test_forward_still_skips_the_transpose_copy(device):
    # The base and a transpose flag are what make the product cheap; keeping
    # the transposed Var in the graph must not run its copy.
    base = jt.array(np.random.rand(64, 32).astype("float32"))
    b = jt.array(np.random.rand(64, 16).astype("float32"))
    jt.sync_all()
    with jt.profile_scope(warmup=0, rerun=0) as report:
        jt.matmul(base.transpose((1, 0)), b).sync()
    names = [str(row[0]) for row in report[1:]]
    assert not any("transpose" in name for name in names), names


@pytest.mark.parametrize("case", ["cancel", "compose"])
def test_transpose_of_a_transpose_keeps_the_intermediate(device, case):
    if case == "cancel":
        base = jt.array(np.arange(6, dtype="float32").reshape(2, 3))
        x = base.transpose((1, 0))
        y = x.transpose((1, 0))
    else:
        base = jt.array(np.arange(24, dtype="float32").reshape(2, 3, 4))
        x = base.transpose((1, 0, 2))
        y = x.transpose((0, 2, 1))
    w_np = np.random.default_rng(0).normal(size=y.shape).astype("float32")
    with jt.flag_scope(missing_grad_error=1):
        gx, gbase = jt.grad((y * jt.array(w_np)).sum(), [x, base])
    perm = (1, 0) if case == "cancel" else (0, 2, 1)
    expected_x = np.transpose(w_np, np.argsort(perm))
    expected_base = np.transpose(expected_x, (1, 0) if case == "cancel" else (1, 0, 2))
    np.testing.assert_allclose(gx.numpy(), expected_x)
    np.testing.assert_allclose(gbase.numpy(), expected_base)
    np.testing.assert_allclose(y.numpy(), np.transpose(np.transpose(base.numpy(),
        (1, 0) if case == "cancel" else (1, 0, 2)), perm))
