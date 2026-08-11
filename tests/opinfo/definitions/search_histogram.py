# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Search / histogram OpInfos -- ``searchsorted`` (binary-search kernel) and ``histc``
(histogram kernel), both jittor-core (``jittor.misc``). Integer/count output, no
gradient (``supports_autograd=False``).

These are deterministic data-dependent kernels that had no op_db entry. The binary search
(boundary handling, the ``right`` side flag) and the histogram bin assignment (the value
that lands exactly on a bin edge) are exactly the off-by-one-prone logic a forward-vs-numpy
oracle pins precisely -- and ``test_device_parity`` then pins the CUDA kernel against the
CPU one. Forward is compared against ``np.searchsorted`` / ``np.histogram``.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F, cu)
from ..core import OpInfo


# ----------------------------------------------------------------- numpy refs
def searchsorted_ref(sorted, values, right=False):
    return np.searchsorted(sorted, values, side="right" if right else "left")


def histc_ref(input, bins, min=0.0, max=0.0):
    return np.histogram(input, bins=bins, range=(min, max))[0].astype("float32")


# --------------------------------------------------------------- sample builders
def sample_searchsorted(op_info, device, dtype, requires_grad):
    # the haystack must be SORTED; sweep both side flags. requires_grad ignored.
    out = []
    rng = np.random.RandomState(1600)
    for i, right in enumerate([False, True]):
        haystack = np.sort(rng.uniform(-3, 3, size=8).astype(dtype))
        needles = rng.uniform(-4, 4, size=(2, 5)).astype(dtype)   # incl. out-of-range ends
        out.append(SampleInput(jt.array(haystack, dtype=dtype),
                               jt.array(needles, dtype=dtype), right=right))
    return out


def sample_histc(op_info, device, dtype, requires_grad):
    # values kept strictly inside [min, max] so every element is counted (np.histogram
    # drops out-of-range values; matching jittor's in-range counting exactly).
    out = []
    for i, (n, bins, lo, hi) in enumerate([(40, 5, 0.0, 1.0), (60, 8, -1.0, 1.0)]):
        x = make_tensor(n, dtype=dtype, low=lo + 0.02, high=hi - 0.02, seed=1610 + i)
        out.append(SampleInput(x, bins, min=lo, max=hi))
    return out


_FLOAT = cu.floating_types()

op_db = [
    OpInfo("searchsorted", op=jt.searchsorted, ref=searchsorted_ref,
           sample_inputs_func=sample_searchsorted, dtypes=_FLOAT, supports_autograd=False),
    OpInfo("histc", op=jt.histc, ref=histc_ref,
           sample_inputs_func=sample_histc, dtypes=_FLOAT, supports_autograd=False),
]
