# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Second derivatives through the legacy ComplexNumber decompositions.
#
# complex_inv/qr/svd/eigh computed their gradient in a numpy callback, so a
# second derivative raised "numpy_code has no backward callback". They now
# run the native complex implementations through the differentiable
# ComplexNumber <-> native bridge. Each case checks the Hessian-vector product
# (the gradient of <grad, v>) against central finite differences of the
# gradient along v, in float64.
import numpy as np
import pytest
import jittor as jt
from jittor.nn import ComplexNumber
from jittor.linalg import complex as legacy

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


def _sq(cn):
    return (cn.real * cn.real + cn.imag * cn.imag).sum()


def _quartic(cn):
    # sum |z|^4: invariant to a phase on any column or row.
    modulus2 = cn.real * cn.real + cn.imag * cn.imag
    return (modulus2 * modulus2).sum()


def _hermitian(re, im):
    # sym(X): the lower triangle eigh reads, mirrored, plus a shift.
    return ComplexNumber(re + re.transpose() + 6 * jt.init.eye(re.shape[0], dtype="float64"),
                         im - im.transpose())


# Losses invariant to each decomposition's per-column phase.
def _inv_loss(re, im):
    return _sq(legacy.complex_inv(ComplexNumber(re + 3 * jt.init.eye(3, dtype="float64"), im)))


def _qr_loss(re, im):
    q, r = legacy.complex_qr(ComplexNumber(re, im))
    return _quartic(q) + _quartic(r)


def _svd_loss(re, im):
    u, s, _ = legacy.complex_svd(ComplexNumber(re, im))
    return _quartic(u) + (s.real * s.real * s.real).sum()


def _eigh_loss(re, im):
    w, v = legacy.complex_eigh(_hermitian(re, im))
    return (w.real * w.real).sum() + _quartic(v)


_CASES = {"inv": _inv_loss, "qr": _qr_loss, "svd": _svd_loss, "eigh": _eigh_loss}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_hessian_vector_product_matches_finite_differences(device, name):
    f = _CASES[name]
    rng = np.random.default_rng(7)
    re0, im0 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    vr, vi = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))

    def grads(re_np, im_np):
        re = jt.array(re_np, dtype="float64")
        im = jt.array(im_np, dtype="float64")
        return re, im, jt.grad(f(re, im), [re, im])

    re, im, (gr, gi) = grads(re0, im0)
    inner = (gr * jt.array(vr, dtype="float64")).sum() + (gi * jt.array(vi, dtype="float64")).sum()
    with jt.flag_scope(missing_grad_error=1):
        hr, hi = jt.grad(inner, [re, im])

    eps = 1e-5
    _, _, (gr_p, gi_p) = grads(re0 + eps * vr, im0 + eps * vi)
    _, _, (gr_m, gi_m) = grads(re0 - eps * vr, im0 - eps * vi)
    # The Hessian is symmetric, so H v is also the directional derivative of
    # the gradient along v.
    np.testing.assert_allclose(hr.numpy(), (gr_p.numpy() - gr_m.numpy()) / (2 * eps), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hi.numpy(), (gi_p.numpy() - gi_m.numpy()) / (2 * eps), rtol=1e-5, atol=1e-6)
