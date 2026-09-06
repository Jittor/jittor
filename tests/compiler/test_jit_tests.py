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
#: ``src/tests/test_sfrl_allocator.cc`` asserts ``time_limit`` of 400 ms. With
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
#: Asserted on the child's exit status rather than with ``expect_error()``,
#: because there is nothing to catch: the fault is delivered to jittor's signal
#: handler, which reports through ``write(2)`` and ``_exit``s -- throwing out of
#: a signal handler is undefined behaviour. ``crash_isolated`` keeps the crash
#: from taking this pytest process down with it: here the crash is what is under
#: test, but the runner surviving is a precondition for reporting it, which is
#: the opposite of the cases in ``tests/core/test_signal_and_teardown.py`` that
#: deliberately do not isolate.
#:
#: This held ``jit_key_guard_page`` until 3.02. Writing past the end of the jit
#: key buffer used to run into an mprotect'ed guard page, so an over-long key
#: was a SIGSEGV; the case asserted on "Accessing protect pages" in the child's
#: output. There is a length check now and an over-long key raises a catchable
#: ``UserError``, so its successor ``jit_key_overflow`` is an ordinary case in
#: this process -- which is the improvement, not a gap. That an over-long key is
#: still refused rather than truncated is asserted there and, from Python, in
#: ``tests/compiler/test_jit_key_structure.py``.
CRASHING_TESTS = {}

#: C++ unit tests that fail, mapped to the assertion that fails, and owned by
#: whoever wrote them rather than by this bridge.
#:
#: These are the first thing 10.18 caught by putting ``src/tests/*.cc`` into the
#: CPU gate: four cases of the native provider registry series have never been
#: executed by any gate, and four of them do not pass. They are quarantined
#: rather than deleted because each one asserts something the registry is
#: supposed to do, and ``strict=True`` means a fix turns them red here and
#: forces this entry to go away -- a skip would let a fix land unnoticed and
#: leave the quarantine forever.
#:
#: Not fixed here on purpose: ``ops/op_register.{h,cc}`` is another partition's
#: working set, and three of the four are failing assertions about the
#: registry's own behaviour, not about the test harness.
KNOWN_BROKEN_TESTS = {
    # The registry hands an observer a lifecycle event that fails its own
    # `valid()` -- `test_op_register.cc:267`, in the probe's
    # `on_provider_lifecycle_event`. Same assertion for both cases.
    "native_op_registry_lifecycle_consumer_boundary":
        "publishes a lifecycle event that fails event.valid()",
    "native_op_registry_scopes_transfer_teardown_ownership":
        "publishes a lifecycle event that fails event.valid()",
    # Registers the op into a local `NativeOpRegistry`, then looks it up with
    # the free `get_op_id()`, which reads the `op_registry()` singleton. The
    # name cannot be there, so `op_register.cc:210` fires
    # "Op definition not registered: jit_test_provider_dispatch".
    "native_op_registry_provider_dispatch_boundary":
        "asserts a local registry's op is visible through the global get_op_id()",
    # After the scope exits, the provider it replaced is still registered --
    # `test_op_register.cc:414`, `ASSERT(!registry.has_provider(...))`.
    "native_op_registry_registration_scope_is_identity_checked":
        "a registration scope leaves the replacement provider registered",
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
    """Bridge to the C++ unit tests registered in ``src/tests/*.cc``.

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

    def test_the_quarantine_names_only_tests_that_exist(self):
        """A quarantine entry for a case that is gone is worse than no entry.

        Renaming or deleting a quarantined case would otherwise leave a name
        here that marks nothing, so the list would keep claiming a defect that
        no longer has anywhere to reproduce.
        """
        installed = set(self.installed_test_names)
        for names, what in ((KNOWN_BROKEN_TESTS, "KNOWN_BROKEN_TESTS"),
                            (CRASHING_TESTS, "CRASHING_TESTS"),
                            (LOAD_SENSITIVE_TESTS, "LOAD_SENSITIVE_TESTS")):
            self.assertEqual(
                sorted(set(names) - installed), [],
                f"{what} names C++ tests that jt.tests does not register")


def _make_test(name):
    def generated_test(self):
        if name in CRASHING_TESTS:
            _run_crashing_test(self, name, CRASHING_TESTS[name])
        else:
            _run_test(name)

    generated_test.__name__ = "test_" + name
    # pytest reads `pytestmark` off the function, which is the only way to
    # mark a method that is generated rather than written.
    marks = []
    if name in LOAD_SENSITIVE_TESTS:
        marks.append(pytest.mark.load_sensitive)
    if name in KNOWN_BROKEN_TESTS:
        marks.append(pytest.mark.xfail(
            strict=True,
            reason="%s: %s" % (name, KNOWN_BROKEN_TESTS[name])))
    if marks:
        generated_test.pytestmark = marks
    return generated_test


def _install_jit_tests():
    names = sorted(name for name in dir(jt.tests) if not name.startswith("__"))
    if not names:
        raise RuntimeError(
            "jt.tests exposes no C++ unit tests. src/tests/*.cc (expr, kernel_ir, "
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
