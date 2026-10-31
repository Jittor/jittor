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
2. A Var whose op failed to compile **stays in the graph** and keeps failing
   when it is read: ``held.sync()`` and ``held.numpy()`` raise the same failure
   every time. A later ``sync_all(True)`` that meets it only as a bystander is
   clean, because implicit execution excludes failed graphs
   (``src/core/executor.cc``, "Exclude failed bystanders from implicit
   execution") -- so it no longer lands on whichever test syncs next.
3. Dropping the reference releases it, and is no longer what makes the next
   ``sync_all(True)`` clean.

A *failed test* keeps that Var alive through the exception's traceback: the
failing frame's locals. ``traceback.clear_frames`` releases them, which is what
the parity battery does around each operator; it is a memory-release measure
now, not what keeps the next operator green.

Everything runs in a child process: the point of the exercise is a deliberately
poisoned runtime, and the assertions in step 2 say it does not stay local.
"""

import os
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

from contextlib import ExitStack as _ProcessPolicyStack
import atexit as _process_policy_atexit
_process_policy_scopes = _ProcessPolicyStack()
_process_policy_atexit.register(_process_policy_scopes.close)
_process_policy_scopes.enter_context(jt.runtime.scope(use_parallel_op_compiler={threads}))
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


_FORK_SCRIPT = """
import os
import threading
import time

import jittor as jt

from contextlib import ExitStack as _ProcessPolicyStack
import atexit as _process_policy_atexit
_process_policy_scopes = _ProcessPolicyStack()
_process_policy_atexit.register(_process_policy_scopes.close)
_process_policy_scopes.enter_context(jt.runtime.scope(use_parallel_op_compiler=2))

def compile_batch(label):
    outputs = []
    for index in range(4):
        option = "FLAGS: -DPC_FORK_%s_%d=1 " % (label, index)
        with jt.flag_scope(compile_options={option: 1}):
            value = jt.array([1.0, 2.0, 3.0])
            outputs.append((value * (index + 2)).sqr())
    jt.sync_all(True)
    return outputs

# Force the parent to construct and use the parallel compiler workers. The
# child needs different keys, or the inherited JIT map would make compilation
# unnecessary and leave the post-fork worker state untested.
parent_outputs = compile_batch("PARENT_%d" % os.getpid())
child_pid = os.fork()
if child_pid == 0:
    jt.jt_init_subprocess()

    def watchdog():
        time.sleep(8)
        os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    child_outputs = compile_batch("CHILD_%d" % os.getpid())
    print("CHILD_PARALLEL_COMPILE_OK", flush=True)
    os._exit(0)

_, status = os.waitpid(child_pid, 0)
exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else 128 + os.WTERMSIG(status)
print("CHILD_STATUS", exit_code, flush=True)
"""


_PREPARE_FAILURE_SCRIPT = """
import jittor as jt

from contextlib import ExitStack as _ProcessPolicyStack
import atexit as _process_policy_atexit
_process_policy_scopes = _ProcessPolicyStack()
_process_policy_atexit.register(_process_policy_scopes.close)
_process_policy_scopes.enter_context(jt.runtime.scope(use_parallel_op_compiler=0))
prepare_once = jt.compile_custom_op(r'''
struct PrepareOnceOp : Op {
    Var* output;
    PrepareOnceOp();
    const char* name() const override { return "prepare_once"; }
    DECLARE_jit_run;
};
''', r'''
#ifndef JIT
PrepareOnceOp::PrepareOnceOp() {
    set_flag(OpFlags::_cpu);
    output = create_output({1}, ns_float32);
}
void PrepareOnceOp::jit_prepare(JK&) {
    static int calls = 0;
    ++calls;
    if (calls == 1) LOGf << "PREPARE_CALL_ONE";
    LOGf << "PREPARE_CALL_TWO";
}
#else
void PrepareOnceOp::jit_run() {}
#endif
''', "prepare_once")

try:
    prepare_once().sync()
except Exception as error:
    print(str(error), flush=True)
"""


class TestCompileFailureAttribution(unittest.TestCase):

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_parallel_compiler_has_no_ghost_workers_after_fork(self):
        result = run_child_script(_FORK_SCRIPT, timeout=180,
                                  name="parallel_compile_after_fork")
        output = result.stdout.decode() + result.stderr.decode()
        self.assertIn("CHILD_PARALLEL_COMPILE_OK", output, output[-3000:])
        self.assertIn("CHILD_STATUS 0", output, output[-3000:])

    def test_prepare_failure_is_not_repeated_by_error_reporting(self):
        result = run_child_script(_PREPARE_FAILURE_SCRIPT, timeout=180,
                                  name="prepare_failure_once")
        output = result.stdout.decode() + result.stderr.decode()
        self.assertIn("PREPARE_CALL_ONE", output, output[-3000:])
        self.assertNotIn("PREPARE_CALL_TWO", output, output[-3000:])

    def test_the_failure_names_its_own_cause_under_the_parallel_compiler(self):
        """Threads do not lose the reason; step 1."""
        output = _run('held = poisoned_var()\nsync("first")\n')
        self.assertIn("first RAISED cause_named=True", output, output[-3000:])

    def test_a_live_failed_var_is_reported_once_and_still_fails_when_read(self):
        """Step 2 -- the failure is reported once, and the failed Var keeps it.

        Serialising the compiler was the old workaround for a later, unrelated
        sync reporting another operator's broken kernel. That no longer happens:
        an implicit sync skips a failed bystander instead of retrying it, so the
        innocent computation below is clean and its value is right. What does
        stay is that reading the failed Var fails every time -- the error is not
        swallowed, only kept to the graph that owns it.
        """
        output = _run(
            'held = poisoned_var()\n'
            'sync("first")\n'
            'unrelated = jt.array([4.0, 5.0]) * 2\n'
            'sync("second")\n'
            'print("unrelated_value", unrelated.sum().item())\n'
            'for attempt in ("read", "read_again"):\n'
            '    try:\n'
            '        held.sync()\n'
            '        print(attempt, "CLEAN")\n'
            '    except Exception as error:\n'
            '        print(attempt, "RAISED", "cause_named=%s" % ("nonexistent-header-that-proves-jit-compile-attribution" in str(error)))\n',
            threads=0,
        )
        self.assertIn("first RAISED cause_named=True", output, output[-3000:])
        self.assertIn("second CLEAN", output, output[-3000:])
        self.assertIn("unrelated_value 18.0", output, output[-3000:])
        self.assertIn("read RAISED cause_named=True", output, output[-3000:])
        self.assertIn("read_again RAISED cause_named=True", output, output[-3000:])

    def test_dropping_the_failed_var_releases_it(self):
        """Step 3 -- dropping it still releases the memory, and was never the
        reason the next sync is clean."""
        output = _run(
            'held = poisoned_var()\n'
            'sync("first")\n'
            'del held\n'
            'gc.collect()\n'
            'sync("second")\n'
        )
        self.assertIn("first RAISED cause_named=True", output, output[-3000:])
        self.assertIn("second CLEAN", output, output[-3000:])

    def test_a_traceback_does_not_poison_later_syncs(self):
        """Why a *failed test* no longer poisons the next one.

        Nothing in the failing test still names the Var -- the frame that does
        is the one pytest keeps for the report -- so the traceback, not the
        test, is what holds it. Holding it is harmless to another sync now, and
        the Var itself still fails when something reads it.
        """
        output = _run(
            'kept_var = []\n'
            'def failing_test():\n'
            '    kept_var.append(poisoned_var())\n'
            '    jt.sync_all(True)\n'
            'try:\n'
            '    failing_test()\n'
            'except Exception:\n'
            '    kept = sys.exc_info()[2]\n'
            'gc.collect()\n'
            'sync("while_traceback_held")\n'
            'try:\n'
            '    kept_var[0].sync()\n'
            '    print("read CLEAN")\n'
            'except Exception as error:\n'
            '    print("read RAISED", "cause_named=%s" % ("nonexistent-header-that-proves-jit-compile-attribution" in str(error)))\n'
            'traceback.clear_frames(kept)\n'
            'gc.collect()\n'
            'sync("after_clear_frames")\n'
        )
        self.assertIn("while_traceback_held CLEAN", output, output[-3000:])
        self.assertIn("read RAISED cause_named=True", output, output[-3000:])
        self.assertIn("after_clear_frames CLEAN", output, output[-3000:])



if __name__ == "__main__":
    unittest.main()
