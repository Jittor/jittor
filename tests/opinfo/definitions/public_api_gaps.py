# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Small OpInfos for high-frequency public APIs missing from the registry."""

from ._refs import *  # noqa: F401,F403
from ..core import OpInfo


def nonzero_ref(x):
    return np.argwhere(x != 0)


def sample_nonzero(op_info, device, dtype, requires_grad):
    value = np.array([[0.0, 2.0, 0.0], [-3.0, 0.0, 4.0]], dtype="float32")
    return [SampleInput(jt.array(value, dtype=dtype))]


def unique_ref(x):
    return np.unique(x)


def sample_unique(op_info, device, dtype, requires_grad):
    value = np.array([1.5, -2.0, 1.5, 0.25, -2.0, 3.0], dtype="float32")
    return [SampleInput(jt.array(value, dtype=dtype))]


def einsum_ref(a, b, equation):
    return np.einsum(equation, a, b)


def _einsum(a, b, equation):
    return jt.einsum(equation, a, b)


def sample_einsum(op_info, device, dtype, requires_grad):
    a = make_tensor(2, 3, dtype=dtype, requires_grad=requires_grad, seed=2100)
    b = make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=2101)
    return [SampleInput(a, b, equation="ij,jk->ik")]


def rms_norm_ref(x, weight, eps=1e-6):
    value = x.astype("float64")
    scale = 1.0 / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)
    return value * scale * weight


def sample_rms_norm(op_info, device, dtype, requires_grad):
    x = make_tensor(2, 4, dtype=dtype, requires_grad=requires_grad, seed=2110)
    weight = make_tensor(
        4, dtype=dtype, low=0.5, high=1.5,
        requires_grad=requires_grad, seed=2111)
    return [SampleInput(x, weight, eps=1e-6)]


_FLOAT32 = (cu.float32,)

op_db = [
    OpInfo("nonzero", op=jt.nonzero, ref=nonzero_ref,
           sample_inputs_func=sample_nonzero, dtypes=_FLOAT32,
           supports_autograd=False),
    OpInfo("unique", op=jt.unique, ref=unique_ref,
           sample_inputs_func=sample_unique, dtypes=_FLOAT32,
           supports_autograd=False),
    OpInfo("einsum", op=_einsum, ref=einsum_ref,
           sample_inputs_func=sample_einsum, dtypes=_FLOAT32),
    OpInfo("rms_norm", op=nn.rms_norm, ref=rms_norm_ref,
           sample_inputs_func=sample_rms_norm, dtypes=_FLOAT32,
           reference_tol=(1e-5, 1e-5)),
]
