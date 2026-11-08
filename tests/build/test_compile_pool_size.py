"""The compile pool is sized by the CPUs this process may run on.

It was sized by memory alone (one process per 3 GiB, at most 16): a worker
pinned to one CPU started 16 compilers that all shared that CPU, each holding a
compiler's memory and none finishing sooner. The native parallel compiler
already took the affinity into account (`parallel_compile_worker_count`).
"""
import os
import textwrap
import unittest

import jittor_utils
from _helpers.child_process import run_child_script

GIB = 1024 ** 3


class TestCompilePoolSize(unittest.TestCase):

    def test_memory_and_cpus_both_bound_the_pool(self):
        size = jittor_utils.compile_pool_size
        self.assertEqual(size(96 * GIB, 64), 16)     # the 16 cap
        self.assertEqual(size(12 * GIB, 64), 4)      # 3 GiB per compiler
        self.assertEqual(size(96 * GIB, 1), 1)       # one CPU, one compiler
        self.assertEqual(size(96 * GIB, 6), 6)
        self.assertEqual(size(1 * GIB, 8), 1)        # never below one
        self.assertEqual(size(96 * GIB, 0), 16)      # affinity unknown

    @unittest.skipUnless(hasattr(os, "sched_setaffinity"), "needs CPU affinity")
    def test_a_process_pinned_to_one_cpu_starts_one_compiler(self):
        cpu = sorted(os.sched_getaffinity(0))[0]
        script = textwrap.dedent(f"""
            import os
            os.sched_setaffinity(0, {{{cpu}}})
            import jittor_utils
            jittor_utils.pool_size = 0
            jittor_utils.run_cmds([], "", "", "pool size probe")
            print("POOL", jittor_utils.pool_size)
        """)
        result = run_child_script(script, text=True, timeout=600, merge_stderr=True,
                                  name="compile_pool_size")
        self.assertIn("POOL 1", result.stdout, result.stdout[-2000:])


if __name__ == "__main__":
    unittest.main()
