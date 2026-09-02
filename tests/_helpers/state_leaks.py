"""Report runtime state one test file leaves behind for the next one.

Three failures in this tree were the same failure. Each looked like a bug in the
test that reported it, and in each case the state had been left behind by a file
that had already passed:

* ``tests/compiler/test_fused_op.py::TestFusedOp::test_add`` asserts
  ``number_of_hold_vars() == 0``. It passes alone and fails after
  ``tests/core`` and ``tests/ops`` have run.
* ``test_torch_compat_fsdp2::test_single_rank_fully_shard_preserves_math_and_state``
  passes alone (20 passed) and fails in a combined run.
* ``tests/ops/test_linalg.py::TestBUG4_2Op`` sets ``use_cuda=1`` and never
  restores it, so every later case in that file runs on CUDA -- where ``eigh``'s
  gradient is wrong (6.P23). The file "passes" because the tests that would
  catch it are the ones being silently redirected.

The debugging cost is what makes this worth automating: the symptom appears in a
file that is innocent, so the first day of every such investigation is spent in
the wrong place. This module snapshots the runtime around each test *file* and
names the file that changed something.

It reports; it does not fail. The survey has to run over the whole tree before
anyone can say which of these are bugs and which are load-bearing, and a check
that goes red on its first run gets switched off rather than read.
"""

import gc
import sys


#: Flags that change what later tests measure rather than how fast they run.
#: ``use_cuda`` decides the device an op runs on, ``no_grad`` and ``amp_reg``
#: decide what a gradient means, and ``exclude_pass``/``use_parallel_op_compiler``
#: decide which compiled kernel is executed.
WATCHED_FLAGS = (
    "use_cuda",
    "no_grad",
    "amp_reg",
    "use_parallel_op_compiler",
    "exclude_pass",
    "th_mode",
)

_COUNTERS = (
    "number_of_hold_vars",
    "number_of_lived_vars",
    "number_of_lived_ops",
)


def _jittor():
    """The runtime, only if this session already imported it.

    Importing it here would pull the core into every static structure test and
    turn a 50 ms file into a compile.
    """
    return sys.modules.get("jittor")


def snapshot(collect=True):
    jittor = _jittor()
    if jittor is None:
        return None
    if collect:
        # Without this the counters measure Python's garbage collector schedule
        # rather than the test file, and every report would be noise.
        gc.collect()
    counters = {}
    for name in _COUNTERS:
        function = getattr(jittor, name, None)
        if callable(function):
            try:
                counters[name] = function()
            except Exception:
                pass
    flags = {}
    for name in WATCHED_FLAGS:
        try:
            flags[name] = getattr(jittor.flags, name)
        except Exception:
            continue
    return {
        "counters": counters,
        "flags": flags,
        "modules": {name: id(module) for name, module in list(sys.modules.items())},
    }


def differences(before, after):
    """Human-readable descriptions of what changed, or an empty list."""
    if not before or not after:
        return []
    report = []
    for name, value in sorted(after["counters"].items()):
        previous = before["counters"].get(name)
        if previous is not None and previous != value:
            report.append("%s %s -> %s" % (name, previous, value))
    for name, value in sorted(after["flags"].items()):
        previous = before["flags"].get(name)
        if previous is not None and previous != value:
            report.append("flags.%s %r -> %r (use jt.flag_scope)" % (name, previous, value))
    for name, identity in sorted(after["modules"].items()):
        previous = before["modules"].get(name)
        if previous is not None and previous != identity:
            # A different object under the same name: something replaced an
            # imported module and did not put the original back.
            report.append("sys.modules[%r] was replaced and not restored" % name)
    return report
