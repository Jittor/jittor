"""The CUDA cache directory must outrank the plain one on the import path.

Jittor keeps two build directories under one cache: ``<cache>/2.0`` for a
CPU-only core and ``<cache>/2.0/<cuda key>`` for a CUDA one. Both go on
``sys.path``. Any run that imports Jittor without nvcc -- a test that spawns a
subprocess with a trimmed environment, a CPU-only job sharing the cache --
builds the CPU-only core into the parent directory, where it stays.

If the parent is searched first, that leftover wins every later import, the
process silently runs without the accelerator, and each CUDA operator fails
with "Op ... doesn't have cuda version". The ordering is the whole defence.
"""

import os
import sys
import unittest

import jittor as jt


class TestCachePathPrecedence(unittest.TestCase):
    @unittest.skipUnless(jt.has_cuda, "CUDA is unavailable")
    def test_cuda_cache_precedes_its_parent(self):
        cache_path = jt.flags.cache_path
        parent = os.path.dirname(cache_path)
        self.assertIn(cache_path, sys.path, "the CUDA cache is not on the path")
        if parent not in sys.path:
            self.skipTest("the plain cache directory is not on the path")
        self.assertLess(
            sys.path.index(cache_path),
            sys.path.index(parent),
            "a CPU-only jittor_core in {} would shadow the CUDA build".format(parent),
        )

    @unittest.skipUnless(jt.has_cuda, "CUDA is unavailable")
    def test_the_imported_core_is_the_cuda_one(self):
        import jittor_core

        self.assertTrue(
            hasattr(jt.flags, "cuda_archs"),
            "jittor_core at {} has no CUDA support".format(jittor_core.__file__),
        )
        self.assertTrue(
            os.path.dirname(os.path.abspath(jittor_core.__file__))
            == os.path.abspath(jt.flags.cache_path),
            "jittor_core came from {}, not the CUDA cache {}".format(
                jittor_core.__file__, jt.flags.cache_path
            ),
        )


if __name__ == "__main__":
    unittest.main()
