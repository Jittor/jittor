import multiprocessing as mp
import unittest

import numpy as np


def _increment_shared_tensor(tensor):
    assert tensor.is_shared()
    tensor.add_(5)


class TensorShareMemoryTest(unittest.TestCase):
    def test_spawned_process_writes_are_visible(self):
        import torch
        value = torch.tensor([1.0, 2.0, 3.0])
        returned = value.share_memory_()
        self.assertIs(returned, value)
        self.assertTrue(value.is_shared())

        process = mp.get_context("spawn").Process(
            target=_increment_shared_tensor, args=(value,))
        process.start()
        process.join(30)
        self.assertEqual(process.exitcode, 0)
        np.testing.assert_allclose(value.numpy(), [6.0, 7.0, 8.0])


if __name__ == "__main__":
    unittest.main()
