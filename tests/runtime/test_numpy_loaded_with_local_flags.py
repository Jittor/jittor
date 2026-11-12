"""Importing jittor first must not make numpy's BLAS global.

``jittor_core`` was imported inside an ``RTLD_GLOBAL`` scope and its module
init was numpy's first import, so numpy's ILP64 OpenBLAS entered the global
symbol namespace. SciPy's LP64 OpenBLAS, loaded afterwards, bound its internal
symbols to it and every LAPACK workspace query returned garbage
("Too large work array required").
"""
import importlib.util
import unittest

from _helpers.child_process import run_python_child

SCRIPT = r"""
import jittor
import scipy.linalg.lapack as lapack
print("lwork", lapack.dgesdd_lwork(16, 3, compute_uv=0))
"""


@unittest.skipUnless(importlib.util.find_spec("scipy"), "needs scipy")
class TestNumpyLoadedWithLocalFlags(unittest.TestCase):
    def test_scipy_lapack_after_jittor(self):
        result = run_python_child(["-c", SCRIPT], env={"CUDA_VISIBLE_DEVICES": ""}, merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("lwork (201.0, 0)", result.stdout)


if __name__ == "__main__":
    unittest.main()
