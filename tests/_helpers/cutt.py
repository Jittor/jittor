"""Getting hold of the cuTT ops, which are built on first use."""

import unittest

import jittor as jt

from jittor._runtime.backend_libraries import get_library_ops


def require_cutt_ops():
    """Return the cuTT ops module, or skip if this machine cannot build it.

    Reading ``compile_extern.cutt_ops`` reports ``None`` on a machine that has
    cuTT: the library is compiled the first time a CUDA transpose asks for it,
    not during import, so it has to be requested with ``load=True``. Every cuTT
    test used to read the plain attribute at module scope and skip with "Not
    use cutt" -- a reason that was never true and that no run could disprove.

    Call this from ``setUpClass``, not at module scope: collecting a test file
    must not compile a backend.
    """
    if not jt.has_cuda:
        raise unittest.SkipTest("no CUDA on this machine, cuTT cannot be built")
    ops = get_library_ops("cutt", load=True)
    if ops is None:
        raise unittest.SkipTest("cuTT is not available in this configuration")
    return ops
