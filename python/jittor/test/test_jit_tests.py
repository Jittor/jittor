# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
from jittor import LOG

def test(name):
    doc = eval(f"jt.tests.{name}.__doc__")
    doc = doc[doc.find("From"):].strip()
    LOG.i(f"Run test {name} {doc}")
    exec(f"jt.tests.{name}()")

tests = [ name for name in dir(jt.tests) if not name.startswith("__") ]
src = "class TestJitTests(unittest.TestCase):\n"
for name in tests:
    doc = eval(f"jt.tests.{name}.__doc__")
    doc = doc[doc.find("From"):].strip()
    src += f"""
    def test_{name}(self):
        test("{name}")
    """

LOG.vvv("eval src\n"+src)
exec(src)

if __name__ == "__main__":
    unittest.main()
