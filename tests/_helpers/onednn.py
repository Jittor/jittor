# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Get oneDNN's operators, loading the library rather than asking whether it is.

``jt.mkl_ops`` is not an accessor. It resolves through
``jittor._runtime.backend_libraries.library_attribute``, which calls
``get_library_ops("mkl")`` *without* ``load=True`` -- so it answers "has MKL
already been loaded in this process", and the answer at import time is no,
because the loader is lazy by design (it fires from
``jittor.nn.functional.matrix``). Every MKL test read it as an accessor, and the
consequences were measured rather than guessed:

* ``tests/backends/cpu`` -- 5 cases, all 5 failing with
  ``AttributeError: 'NoneType' object has no attribute 'mkl_conv'``.
* ``tests/ops/test_mkl_batched_matmul.py`` -- 8 cases, all 8 skipped, and
  skipped for a reason that names no missing hardware, so
  ``JITTOR_TEST_REQUIRE_EXECUTION=1`` cannot even explain it.

That is the whole of Jittor's MKL coverage: 13 cases, none of which ran. The
audit's claim that MKL matmul is fp32-only had no runtime evidence behind it
for the same reason.

So a test asks for the library and gets it, or gets a skip whose reason is the
real cause. ``load=True`` is the entire fix; the lazy loader is doing what it
was built to do.
"""

import pytest

from jittor._runtime.backend_libraries import get_library_ops


#: Set once per process. The load compiles the wrapper sources on a cold cache,
#: and a failure there is worth reporting once with its real message instead of
#: once per test case.
_FAILURE = None


def onednn_ops(required=False):
    """oneDNN's operator module, or ``None``.

    ``required=True`` skips the calling test with the load failure as the
    reason. A build error therefore surfaces as its own message -- "the wrapper
    did not compile" -- and not as the far more comfortable "this machine does
    not have oneDNN", which is what a bare availability check reports and what
    kept the cuTT suite green while running nothing (4.13).
    """
    global _FAILURE
    ops = None
    if _FAILURE is None:
        try:
            ops = get_library_ops("mkl", load=True)
        except BaseException as error:            # noqa: BLE001 - becomes a skip
            _FAILURE = "%s: %s" % (type(error).__name__, str(error)[:400])
        else:
            if ops is None:
                _FAILURE = ("the oneDNN loader produced no module; oneDNN is "
                            "disabled in this build (use_mkl=0)")
    if ops is None and required:
        pytest.skip("oneDNN operators unavailable -- " + _FAILURE)
    return ops


def requires_onednn():
    """``onednn_ops(required=True)``, for a ``setUp``/fixture one-liner."""
    return onednn_ops(required=True)
