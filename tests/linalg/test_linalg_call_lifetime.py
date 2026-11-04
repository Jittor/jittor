# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# A chain of differentiable linalg calls is freed by reference counting alone.
#
# Each of these ops used to define its `jt.Function` subclass inside the call,
# with a `grad` that closed over the input `x`. A class is a reference cycle,
# so only the cyclic GC frees it, and through `x` (C++ graph edges the GC
# cannot see) it kept the previous call's context alive. A chain of n calls
# then took n full collections to free: a 40-site MPS simulation piled up
# about 19,000 Vars per iteration and slowed down as it ran. With the GC off
# nothing at all came back.
import gc

import numpy as np
import pytest
import jittor as jt
from jittor import linalg

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


_N = 6


def _qr(y):
    return linalg.qr(y)[0]


def _svd(y):
    u, s, vh = linalg.svd(y)
    return jt.matmul(u * s.unsqueeze(-2), vh)


def _eigh(y):
    return linalg.eigh(y + y.transpose(-1, -2).conj())[1]


def _cholesky(y):
    return linalg.cholesky(jt.matmul(y, y.transpose(-1, -2).conj()) + jt.init.eye(_N)) * 0.3


def _inv(y):
    return linalg.inv(y + 4 * jt.init.eye(_N))


def _det(y):
    return y * linalg.det(y * 0.1 + jt.init.eye(_N))


def _slogdet(y):
    return y * linalg.slogdet(y * 0.1 + jt.init.eye(_N))[1]


def _solve(y):
    return linalg.solve(y + 4 * jt.init.eye(_N), y)


_OPS = {
    "qr": _qr, "svd": _svd, "eigh": _eigh, "cholesky": _cholesky,
    "inv": _inv, "det": _det, "slogdet": _slogdet, "solve": _solve,
}


def _input(rng, dtype):
    a = rng.normal(size=(_N, _N))
    if dtype == "complex64":
        a = a + 1j * rng.normal(size=(_N, _N))
    return jt.array(a.astype(dtype))


def _chain(op, x, grad):
    y = x
    for _ in range(8):
        y = op(y * 0.5)
    loss = y.abs().sum()
    if grad:
        jt.grad(loss, x).sync()
    loss.sync()


@pytest.mark.parametrize("grad", [False, True], ids=["forward", "grad"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("name", sorted(_OPS))
def test_chain_is_freed_without_the_cyclic_gc(device, name, dtype, grad):
    op = _OPS[name]
    rng = np.random.default_rng(0)
    # Warm up: kernels, caches and lazily built constants.
    _chain(op, _input(rng, dtype), grad)
    gc.collect()
    x = _input(rng, dtype)
    gc.disable()
    try:
        before = jt.number_of_lived_vars()
        _chain(op, x, grad)
        after = jt.number_of_lived_vars()
    finally:
        gc.enable()
    assert after == before
