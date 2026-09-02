# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Every target that gets no gradient must be reported, not just the first one.

``grad()`` fills a target that collected no gradient with zeros.  The report was
a ``LOGw`` deduplicated through a **process-global map keyed on the var's
name** -- and almost every var's name is the empty string, so exactly one
missing gradient was ever announced per process and every later one was silent.
Training then converges to the wrong thing with a clean log.

What is asserted here:

  1. with ``missing_grad_error`` off, N missing gradients produce N reports (the
     dedup map made this 1);
  2. two *different* unnamed vars are both reported;
  3. with the flag on (the default), a missing gradient raises;
  4. a target that does have a gradient is never reported.

The counting cases run in a subprocess: jittor logs through its own C++ logger
straight to the process stderr, which pytest's capture does not intercept.

Run::  python -m pytest tests/core/test_grad_missing.py
"""

import os
import subprocess
import sys
import unittest

import jittor as jt


PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(jt.__file__)))

PREAMBLE = """
import numpy as np
import jittor as jt
jt.flags.missing_grad_error = {flag}

def unrelated_pair():
    # loss does not depend on target at all, so target collects no gradient.
    target = jt.array(np.array([1.0, 2.0], dtype="float32"), dtype="float32")
    source = jt.array(np.array([3.0, 4.0], dtype="float32"), dtype="float32")
    return (source * source).sum(), target
"""


def run(body, flag=0):
    source = PREAMBLE.format(flag=flag) + body
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [PYTHON_DIR] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    done = subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    return done.returncode, done.stdout.decode("utf8", "replace")


class TestMissingGradIsAlwaysReported(unittest.TestCase):
    def test_every_missing_gradient_is_reported(self):
        code, output = run("""
for _ in range(3):
    loss, target = unrelated_pair()
    g = jt.grad(loss, target)
    assert float(g.sum().numpy()[0]) == 0.0
print("DONE")
""")
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("DONE", output)
        self.assertEqual(output.count("doesn't have gradient"), 3, output[-4000:])

    def test_second_unnamed_var_is_reported(self):
        code, output = run("""
loss, first = unrelated_pair()
jt.grad(loss, first)
loss2, second = unrelated_pair()
jt.grad(loss2, second)
print("DONE")
""")
        self.assertEqual(code, 0, output[-4000:])
        self.assertEqual(output.count("doesn't have gradient"), 2, output[-4000:])

    def test_present_gradient_is_never_reported(self):
        code, output = run("""
x = jt.array(np.array([1.0, 2.0], dtype="float32"), dtype="float32")
g = jt.grad((x * x).sum(), x)
assert abs(float(g.numpy()[0]) - 2.0) < 1e-6
print("DONE")
""")
        self.assertEqual(code, 0, output[-4000:])
        self.assertNotIn("doesn't have gradient", output)

    def test_flag_on_raises(self):
        code, output = run("""
loss, target = unrelated_pair()
try:
    jt.grad(loss, target)
except Exception as error:
    print("RAISED", "doesn't have gradient" in str(error))
else:
    print("NO ERROR")
""", flag=1)
        self.assertEqual(code, 0, output[-4000:])
        self.assertIn("RAISED True", output, output[-4000:])


class TestMissingGradInProcess(unittest.TestCase):
    def test_flag_exists_and_round_trips(self):
        before = jt.flags.missing_grad_error
        try:
            jt.flags.missing_grad_error = 0
            self.assertEqual(jt.flags.missing_grad_error, 0)
            jt.flags.missing_grad_error = 1
            self.assertEqual(jt.flags.missing_grad_error, 1)
        finally:
            jt.flags.missing_grad_error = before


if __name__ == "__main__":
    unittest.main()
