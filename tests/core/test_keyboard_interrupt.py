# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Ctrl-C raises KeyboardInterrupt in a process that imported jittor.

jittor used to install its own SIGINT action, which printed "Caught SIGINT,
quick exit" and called quick_exit: no traceback, no ``except
KeyboardInterrupt`` to save a checkpoint, no ``finally``, no context manager
exit. Only Jupyter was exempt.

Each case runs in a child in its own session, so the signal goes to the
child's process group and nothing else. The dataset case also checks the
workers: they ignore SIGINT and leave with the parent, so one Ctrl-C does not
print a traceback per worker.
"""

import os
import signal
import subprocess
import textwrap
import time
import unittest

from _helpers.child_process import child_env, python_executable


SLEEPER = '''
import time
import jittor as jt
try:
    print("ready", flush=True)
    time.sleep(60)
    print("sleep finished", flush=True)
except KeyboardInterrupt:
    print("caught KeyboardInterrupt", flush=True)
finally:
    print("finally ran", flush=True)
'''

LOADER = '''
import time
import numpy as np
import jittor as jt
from jittor.dataset import Dataset

class Slow(Dataset):
    def __init__(self):
        super().__init__(batch_size=2, num_workers=2)
        self.set_attrs(total_len=100000)
    def __getitem__(self, i):
        time.sleep(0.01)
        return np.array([i], "float32")

try:
    for n, batch in enumerate(Slow()):
        if n == 0:
            print("ready", flush=True)
except KeyboardInterrupt:
    print("caught KeyboardInterrupt", flush=True)
finally:
    print("finally ran", flush=True)
'''


def _interrupt(source, timeout=180):
    child = subprocess.Popen(
        [python_executable(), "-c", textwrap.dedent(source)],
        env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    try:
        seen = []
        deadline = time.monotonic() + timeout
        for line in child.stdout:
            seen.append(line)
            if line.strip() == "ready" or time.monotonic() > deadline:
                break
        if not seen or seen[-1].strip() != "ready":
            child.kill()
            raise AssertionError("child never became ready:\n" + "".join(seen))
        time.sleep(1)
        os.killpg(child.pid, signal.SIGINT)
        rest, _ = child.communicate(timeout=60)
        return child.returncode, "".join(seen) + rest
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


class TestKeyboardInterrupt(unittest.TestCase):

    def _check(self, rc, out):
        self.assertIn("caught KeyboardInterrupt", out)
        self.assertIn("finally ran", out)
        self.assertNotIn("Caught SIGINT", out)
        self.assertEqual(rc, 0, out)

    def test_ctrl_c_raises_keyboard_interrupt(self):
        self._check(*_interrupt(SLEEPER))

    def test_ctrl_c_while_dataset_workers_run(self):
        rc, out = _interrupt(LOADER)
        self._check(rc, out)
        self.assertNotIn("Traceback", out,
                         "a worker reported the parent's Ctrl-C:\n" + out)


if __name__ == "__main__":
    unittest.main()
