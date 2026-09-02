"""Whose kernel failed to compile: the operator that queued it, or the next one?

``tests/backends/parity/test_device_parity.py`` used to switch the parallel op
compiler off for its whole class, with the reason "otherwise an asynchronous
compile error can surface in the following test and be attributed to the wrong
operator". The observation was right and the diagnosis was not, and the cost was
the compile parallelism of a 227-operator battery -- the dominant term in the
CUDA gate.

What actually happens (each step below is one assertion here):

1. A JIT compile failure is reported with the deliberate cause in it, parallel
   compiler or not. Attribution is not lost by compiling in threads.
2. A Var whose op failed to compile **stays in the graph**, and every later
   ``sync_all(True)`` re-raises that same failure -- so an innocent later test
   reports another test's broken kernel. This is the real defect, and it has
   nothing to do with how many threads compile.
3. Dropping the reference to that Var makes the next ``sync_all(True)`` clean.

And what holds the reference after a *failed test* is the exception's traceback:
the failing frame's locals. ``traceback.clear_frames`` releases them, which is
what the parity battery now does around each operator.

Everything runs in a child process: the point of the exercise is a deliberately
poisoned runtime, and the assertions in step 2 say it does not stay local.
"""

import unittest

from _helpers.child_process import run_child_script


#: A compile flag that cannot succeed and says why in the compiler's own words.
#: A bad ``-D`` is not enough -- an unused macro compiles fine -- so this forces
#: the driver to open a file that does not exist.
_BAD_FLAG = (
    '{"FLAGS: -include '
    '/nonexistent-header-that-proves-jit-compile-attribution.h ": 1}'
)
_MARKER = "nonexistent-header-that-proves-jit-compile-attribution"

_PREAMBLE = """
import gc
import sys
import traceback
import jittor as jt

jt.flags.use_parallel_op_compiler = {threads}
BAD = {bad}

def sync(tag):
    try:
        jt.sync_all(True)
        print(tag, "CLEAN")
    except Exception as error:
        print(tag, "RAISED", "cause_named=%s" % ({marker!r} in str(error)))

def poisoned_var():
    a = jt.array([1.0, 2.0, 3.0])
    with jt.flag_scope(compile_options=BAD):
        return a.sqr() * 7.0
"""


def _script(body, threads=16):
    return _PREAMBLE.format(threads=threads, bad=_BAD_FLAG, marker=_MARKER) + body


def _run(body, threads=16):
    result = run_child_script(_script(body, threads), timeout=900,
                              name="compile_attribution")
    return result.stdout.decode() + result.stderr.decode()


class TestCompileFailureAttribution(unittest.TestCase):

    def test_the_failure_names_its_own_cause_under_the_parallel_compiler(self):
        """Threads do not lose the reason; step 1."""
        output = _run('held = poisoned_var()\nsync("first")\n')
        self.assertIn("first RAISED cause_named=True", output, output[-3000:])

    def test_a_live_failed_var_makes_the_next_sync_report_the_same_failure(self):
        """Step 2 -- the defect, with the parallel compiler *off*.

        Serialising the compiler was the old workaround. It does not help: the
        second sync belongs to an unrelated computation and still reports the
        first one's broken kernel.
        """
        output = _run(
            'held = poisoned_var()\n'
            'sync("first")\n'
            'unrelated = jt.array([4.0, 5.0]) * 2\n'
            'sync("second")\n'
            'print("unrelated_value", unrelated.sum().item())\n',
            threads=0,
        )
        self.assertIn("first RAISED cause_named=True", output, output[-3000:])
        self.assertIn("second RAISED cause_named=True", output, output[-3000:])
        # The innocent computation itself is fine -- fetching just it works. Only
        # the whole-graph sync is contaminated, which is exactly why the symptom
        # lands on whichever test syncs next rather than on the one that broke.
        self.assertIn("unrelated_value 18.0", output, output[-3000:])

    def test_dropping_the_failed_var_makes_the_next_sync_clean(self):
        """Step 3 -- the remedy the parity battery relies on."""
        output = _run(
            'held = poisoned_var()\n'
            'sync("first")\n'
            'del held\n'
            'gc.collect()\n'
            'sync("second")\n'
        )
        self.assertIn("first RAISED cause_named=True", output, output[-3000:])
        self.assertIn("second CLEAN", output, output[-3000:])

    def test_a_traceback_keeps_the_failed_var_alive_until_frames_are_cleared(self):
        """Why a *failed test* poisons the next one, and what fixes it.

        Nothing in the failing test still names the Var -- the frame that does is
        the one pytest keeps for the report. So the release has to be explicit.
        """
        output = _run(
            'def failing_test():\n'
            '    local = poisoned_var()\n'
            '    jt.sync_all(True)\n'
            'try:\n'
            '    failing_test()\n'
            'except Exception:\n'
            '    kept = sys.exc_info()[2]\n'
            'gc.collect()\n'
            'sync("while_traceback_held")\n'
            'traceback.clear_frames(kept)\n'
            'gc.collect()\n'
            'sync("after_clear_frames")\n'
        )
        self.assertIn("while_traceback_held RAISED cause_named=True",
                      output, output[-3000:])
        self.assertIn("after_clear_frames CLEAN", output, output[-3000:])


if __name__ == "__main__":
    unittest.main()
