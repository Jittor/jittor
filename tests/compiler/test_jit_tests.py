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
    pass


def _make_test(name):
    def generated_test(self):
        _run_test(name)

    generated_test.__name__ = "test_" + name
    return generated_test


def _install_jit_tests():
    for name in dir(jt.tests):
        if not name.startswith("__"):
            setattr(TestJitTests, "test_" + name, _make_test(name))


_install_jit_tests()

if __name__ == "__main__":
    unittest.main()
