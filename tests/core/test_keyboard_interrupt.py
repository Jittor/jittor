"""SIGINT remains catchable after importing Jittor and starting workers."""
import os
import unittest

from _helpers.child_process import run_child_script


@unittest.skipUnless(os.name == "posix", "process-group signals require POSIX")
class TestKeyboardInterrupt(unittest.TestCase):
    def _check(self, source):
        result = run_child_script(source, without_torch_mode=True, timeout=600)
        out, err = result.stdout.decode(), result.stderr.decode()
        self.assertEqual(result.returncode, 0, out + err)
        self.assertIn("caught KeyboardInterrupt", out)
        self.assertIn("finally ran", out)
        self.assertNotIn("Caught SIGINT", out + err)
        self.assertNotIn("Traceback", err)

    def test_ctrl_c_raises_keyboard_interrupt(self):
        self._check('''
import os
import signal
import jittor
try:
    os.kill(os.getpid(), signal.SIGINT)
except KeyboardInterrupt:
    print("caught KeyboardInterrupt")
finally:
    print("finally ran")
''')

    def test_ctrl_c_while_dataset_workers_run(self):
        self._check('''
import os
import signal
import time
import multiprocessing as mp
import numpy as np
from jittor.dataset import Dataset

ready = mp.Array("b", 2)

def mark_ready(worker_id):
    ready[worker_id] = 1

class Slow(Dataset):
    def __init__(self):
        super().__init__(batch_size=2, num_workers=2)
        self.set_attrs(total_len=100000)
        self.worker_init_fn = mark_ready
    def __getitem__(self, index):
        time.sleep(0.01)
        return np.array([index], dtype="float32")

dataset = Slow()
try:
    for batch in dataset:
        # The contract concerns running workers; wait for both workers to
        # finish Python's fork hooks and enter their initialized target.
        deadline = time.monotonic() + 30
        while not all(ready):
            if time.monotonic() >= deadline:
                raise AssertionError("workers did not finish initialization")
            time.sleep(0.01)
        # The test helper owns this process group, including dataset workers.
        os.killpg(os.getpgrp(), signal.SIGINT)
except KeyboardInterrupt:
    print("caught KeyboardInterrupt")
finally:
    dataset._abort_workers()
    print("finally ran")
''')
