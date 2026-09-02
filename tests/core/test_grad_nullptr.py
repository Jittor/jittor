# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``grad.cc`` must not dereference a null gradient.

``Op::grad`` returning ``nullptr`` is an ordinary, documented path: it is what
floor/round/ceil, mod, floor_divide and the bitwise ops all return, and what the
base ``Op::grad`` returns for any op without a backward.  Two places in
``grad.cc`` decided on one object and then dereferenced another:

  * ``make_grad`` tested ``x->loop_options`` (the *input*) and then wrote
    ``dx->loop_options`` (the *result*), so any var carrying ``compile_options``
    segfaulted as soon as one of its consumers had no gradient;
  * the ``auto_mixed_precision_level == 3`` cast tested nothing at all and read
    ``grad->ns`` for a target whose contributions were all null.

Both crash the process, so each case runs in its own subprocess: a regression
must show up as a failed assertion here, not as a dead pytest session.

Run::  python -m pytest tests/core/test_grad_nullptr.py
"""

import os
import subprocess
import sys
import unittest

import jittor as jt


#: Repo ``python/`` directory. Jittor is installed editable, so a bare
#: ``python -c "import jittor"`` in a worktree imports whatever tree the .pth
#: points at -- the child must be pinned to the tree this test is running from.
PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(jt.__file__)))

# floor has no gradient, so make_grad returns nullptr for its input.
PROGRAM = """
import numpy as np
import jittor as jt
{setup}
x = jt.array(np.array([1.5, 2.5], dtype="float32"), dtype="float32")
{tweak}
loss = jt.floor(x).sum()
g = jt.grad(loss, x)
print("GRAD", g.numpy().tolist())
"""


def _run(setup, tweak):
    source = PROGRAM.format(setup=setup, tweak=tweak)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [PYTHON_DIR] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )


class TestGradNullptr(unittest.TestCase):
    def _check(self, setup="", tweak=""):
        done = _run(setup, tweak)
        output = done.stdout.decode("utf8", "replace")
        self.assertEqual(done.returncode, 0, output[-4000:])
        self.assertIn("GRAD [0.0, 0.0]", output, output[-4000:])
        self.assertNotIn("Segfault", output, output[-4000:])

    def test_missing_grad_with_compile_options(self):
        # make_grad: `if (x->loop_options) dx->loop_options = ...`
        self._check(tweak='x.compile_options = {"compile_shapes": 1}')

    def test_missing_grad_under_amp_level_3(self):
        # grad(): `if (auto_mixed_precision_level == 3 && grad->ns != var->ns)`
        self._check(setup="jt.flags.auto_mixed_precision_level = 3")

    def test_missing_grad_plain(self):
        # The same graph without either trigger must keep working.
        self._check()


if __name__ == "__main__":
    unittest.main()
