import unittest

import torch


class TestThreadRuntimeKnob(unittest.TestCase):
    def test_set_num_threads_changes_native_openmp_runtime(self):
        original = torch.get_num_threads()
        try:
            torch.set_num_threads(3)
            self.assertEqual(torch.get_num_threads(), 3)
            torch.set_num_threads(1)
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(original)

    def test_set_num_threads_rejects_invalid_values(self):
        for value in (0, -1, True, 1.5, "2"):
            with self.assertRaises((RuntimeError, TypeError)):
                torch.set_num_threads(value)


if __name__ == "__main__":
    unittest.main()
