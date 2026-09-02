"""Rebuilding jit_utils means the command did not run, and must say so.

``import jittor`` recompiles ``jit_utils_core`` when its sources change and
then cannot load the new file into this interpreter, so it asks to be run
again. It used to report that with ``sys.exit(0)``: a training script or a CI
job that touched ``src/utils/*.cc`` "succeeded" without executing a line of the
user's program, and the only trace was one line of log among hundreds.
"""

import unittest

import jittor as jt
from jittor import compiler


class TestJitUtilsRerunExitCode(unittest.TestCase):

    def test_the_exit_code_is_not_success(self):
        self.assertNotEqual(compiler.JIT_UTILS_UPDATED_EXIT_CODE, 0)

    def test_asking_for_a_rerun_exits_with_that_code(self):
        """Nothing is built: ``compile`` is stubbed to claim it rebuilt."""
        self.assertTrue(compiler.jit_utils.cc,
                        "jit_utils_core is not loaded in this process")
        saved = compiler.compile
        compiler.compile = lambda *args, **kw: True
        try:
            with self.assertRaises(SystemExit) as caught:
                compiler.check_cache_compile()
        finally:
            compiler.compile = saved
        self.assertEqual(caught.exception.code,
                         compiler.JIT_UTILS_UPDATED_EXIT_CODE)
        self.assertNotEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
