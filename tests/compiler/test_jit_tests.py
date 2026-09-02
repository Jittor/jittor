# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import LOG


def _run_test(name):
    target = getattr(jt.tests, name)
    doc = target.__doc__
    doc = doc[doc.find("From"):].strip()
    LOG.i(f"Run test {name} {doc}")
    target()


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
        _run_test(name)

    generated_test.__name__ = "test_" + name
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
