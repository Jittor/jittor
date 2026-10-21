"""A failed compile in the pool reaches the caller once, as a plain RuntimeError.

The pool pickles a worker's exception back to the parent. Pickling
``jittor.JittorError`` looks the class up as ``jittor.JittorError``, which a
worker forked half way through ``import jittor`` does not have yet: the first
cold build that failed ended in ``MaybeEncodingError: Can't pickle <class
'jittor.JittorError'>`` instead of the compiler's diagnostic. And the pool
chains the worker's traceback as the cause, which ends in the same message, so
the diagnostic was printed twice.
"""
import os
import tempfile
import unittest

import jittor as jt
import jittor_utils


class TestCompilePoolFailure(unittest.TestCase):

    def test_the_diagnostic_arrives_once_and_unchained(self):
        directory = tempfile.mkdtemp()
        source = os.path.join(directory, "broken.cc")
        with open(source, "w") as f:
            f.write("int f() { return pool_probe_missing_symbol; }\n")
        commands = ['"%s" "%s" -c -o "%s"' % (jt.compiler.cc_path, source,
                                               os.path.join(directory, name))
                    for name in ("a.o", "b.o")]
        with self.assertRaises(RuntimeError) as caught:
            jittor_utils.run_cmds(commands, jt.compiler.cache_path,
                                  jt.compiler.jittor_path, "pool error probe")
        error = caught.exception
        self.assertIs(type(error), RuntimeError)
        self.assertIn("error:", str(error))
        self.assertIn("pool_probe_missing_symbol", str(error))
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
