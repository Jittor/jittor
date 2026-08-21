# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Floating-point predicate OpInfos -- ``isnan`` / ``isinf`` / ``isfinite`` (core,
``jittor.misc``). Bool output, no gradient (``supports_autograd=False``).

Their entire purpose is to classify the SPECIAL values (NaN, +/-Inf), so -- unlike every
other op_db entry -- the samples must deliberately CONTAIN those values; ``make_tensor``
never produces them. The forward is compared exactly against ``np.isnan`` / ``np.isinf`` /
``np.isfinite``, and (via ``test_device_parity``) the CUDA classification is pinned against
the CPU one -- NaN/Inf handling is a notorious accelerator/fusion divergence point (the
fusion suite separately checks direct comparisons against NumPy's IEEE behavior).
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo

_INF = float("inf")
# fixed pattern exercising every class: finite +/-, zero, subnormal-ish, NaN, +/-Inf.
_PATTERN = np.array([1.0, -2.5, 0.0, 1e-7, np.nan, _INF, -_INF, 3.3], dtype="float64")


def isnan_ref(x):      return np.isnan(x)
def isinf_ref(x):      return np.isinf(x)
def isfinite_ref(x):   return np.isfinite(x)


def sample_predicate(op_info, device, dtype, requires_grad):
    # build Vars that actually contain NaN/+-Inf, at the requested float dtype; a couple
    # of shapes (1-D and reshaped 2-D) so the kernel is exercised over >1 layout.
    base = _PATTERN.astype(dtype)
    out = [SampleInput(jt.array(base, dtype=dtype))]
    tiled = np.concatenate([base, base]).reshape(4, 4).astype(dtype)
    out.append(SampleInput(jt.array(tiled, dtype=dtype)))
    return out


_FLOAT = cu.floating_types()

op_db = [
    OpInfo("isnan", op=jt.isnan, ref=isnan_ref, sample_inputs_func=sample_predicate,
           dtypes=_FLOAT, supports_autograd=False),
    OpInfo("isinf", op=jt.isinf, ref=isinf_ref, sample_inputs_func=sample_predicate,
           dtypes=_FLOAT, supports_autograd=False),
    OpInfo("isfinite", op=jt.isfinite, ref=isfinite_ref, sample_inputs_func=sample_predicate,
           dtypes=_FLOAT, supports_autograd=False),
]
