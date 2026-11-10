"""A fetch of host data works on an accelerator build with no visible device.

``FetchOp::run`` asked for the accelerator copy stream unconditionally, so with
``CUDA_VISIBLE_DEVICES=""`` every CPU fetch (``torch.save`` of a CPU model
uses one) aborted with "Invalid stream device 0". The lazy initializer already
skipped the stream when no device is present; the run path now does too.
"""
import unittest

from _helpers.child_process import run_python_child

SCRIPT = r"""
import jittor as jt
values = []
jt.array([1.0, 2.0, 3.0]).fetch(lambda v: values.append(v.tolist()))
jt.sync_all(True)
x = jt.zeros(3) + 1
jt.sync_all(True)
print("fetched", values, x.tolist())
"""


class TestFetchWithoutVisibleDevice(unittest.TestCase):
    def test_cpu_fetch(self):
        result = run_python_child(["-c", SCRIPT], env={"CUDA_VISIBLE_DEVICES": ""}, merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        self.assertIn("fetched [[1.0, 2.0, 3.0]] [1.0, 1.0, 1.0]", result.stdout)


if __name__ == "__main__":
    unittest.main()
