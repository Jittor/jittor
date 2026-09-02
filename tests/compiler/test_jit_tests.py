# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest

import pytest

import jittor as jt
from jittor import LOG

from _helpers.child_process import run_child_script


#: C++ unit tests whose assertion is a wall-clock budget.
#:
#: ``src/test/test_sfrl_allocator.cc`` asserts ``time_limit`` of 400 ms. With
#: nine agents on the box it measured 1775 ms and 10919 ms -- 4x and 27x over,
#: which is not "a bit slow", it is a different question being answered. What
#: they guard (the allocator must not degrade) is worth keeping, so they are
#: marked rather than deleted and run on an idle machine.
LOAD_SENSITIVE_TESTS = frozenset((
    "sfrl_allocator_share",
    "sfrl_allocator_time",
))

#: C++ unit tests whose whole point is that the process dies, mapped to a string
#: their crash report must contain.
#:
#: ``jit_key_guard_page`` writes past the jit key buffer into the mprotect guard
#: page on purpose. The fault is delivered to jittor's signal handler, which
#: reports and ``_exit``s -- there is no catchable path and there should not be
#: one, because throwing out of a signal handler is undefined behaviour. (It
#: used to be caught with ``expect_error()``, and "passed" for years on exactly
#: that undefined behaviour; 2.20 replaced it with defined behaviour and this
#: dependency surfaced.)
#:
#: So these are asserted on the child's exit status instead. ``crash_isolated``
#: keeps the crash from taking this pytest process down with it -- here the
#: crash is what is under test, but the runner surviving is a precondition for
#: reporting it, which is the opposite of the cases in
#: ``tests/core/test_signal_and_teardown.py`` that deliberately do not isolate.
CRASHING_TESTS = {
    "jit_key_guard_page": "Accessing protect pages",
}


def _run_test(name):
    target = getattr(jt.tests, name)
    doc = target.__doc__
    doc = doc[doc.find("From"):].strip()
    LOG.i(f"Run test {name} {doc}")
    target()


def _run_crashing_test(case, name, expected):
    child = run_child_script(
        "import jittor as jt\n"
        "jt.tests.%s()\n"
        "print('NOT-REACHED', flush=True)\n" % name,
        merge_stderr=True,
        crash_isolated=True,
    )
    output = child.stdout.decode("utf8", "replace")
    case.assertNotIn("NOT-REACHED", output)
    case.assertNotEqual(child.returncode, 0, output)
    case.assertIn(expected, output)


class TestJitTests(unittest.TestCase):
    """Bridge to the C++ unit tests registered in ``src/test/*.cc``.

    Every case in this class is generated from ``jt.tests``. When that registry is
    empty -- a wheel that strips ``src/``, or a scan that failed -- the class used
    to end up with no methods at all, which pytest collects as zero cases and
    reports exactly like a pass. ``_install_jit_tests`` therefore refuses to
    install nothing, and ``test_the_bridge_found_the_cpp_unit_tests`` keeps the
    count visible in the gate log rather than only in an exception.
    """

    installed_test_names = ()

    def test_the_bridge_found_the_cpp_unit_tests(self):
        self.assertGreater(
            len(self.installed_test_names), 0,
            "jt.tests registered no C++ unit tests; this file would have run nothing")


def _make_test(name):
    def generated_test(self):
        if name in CRASHING_TESTS:
            _run_crashing_test(self, name, CRASHING_TESTS[name])
        else:
            _run_test(name)

    generated_test.__name__ = "test_" + name
    if name in LOAD_SENSITIVE_TESTS:
        # pytest reads `pytestmark` off the function, which is the only way to
        # mark a method that is generated rather than written.
        generated_test.pytestmark = [pytest.mark.load_sensitive]
    return generated_test


def _install_jit_tests():
    names = sorted(name for name in dir(jt.tests) if not name.startswith("__"))
    if not names:
        raise RuntimeError(
            "jt.tests exposes no C++ unit tests. src/test/*.cc (expr, kernel_ir, "
            "op_compiler, op_relay, sfrl_allocator, setitem_op, jit_key, "
            "nano_vector, fast_shared_ptr) is either absent from this build or was "
            "not scanned. Installing zero generated methods would leave this file "
            "collecting zero cases, which pytest reports as a pass.")
    for name in names:
        setattr(TestJitTests, "test_" + name, _make_test(name))
    TestJitTests.installed_test_names = tuple(names)


_install_jit_tests()

if __name__ == "__main__":
    unittest.main()
