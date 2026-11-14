import pickle
import multiprocessing as mp
import unittest

import numpy as np
from _helpers.child_process import default_timeout


def _increment_shared_tensor(tensor):
    assert tensor.is_shared()
    tensor.add_(5)


class TensorShareMemoryTest(unittest.TestCase):
    def test_share_memory_returns_the_same_tensor(self):
        import torch
        tensor = torch.tensor([1, 2, 3])
        shared = tensor.share_memory_()
        self.assertIs(shared, tensor)
        self.assertEqual(shared.tolist(), [1, 2, 3])

    def test_scalar_epoch_marker_matches_datasets_usage(self):
        import torch
        epoch = torch.tensor(0).share_memory_()
        self.assertEqual(int(epoch.item()), 0)

    def test_existing_aliases_are_rejected_without_breaking_storage(self):
        import torch
        for alias_kind in (lambda x: x[1:], lambda x: x.detach()):
            value = torch.tensor([1., 2., 3.])
            alias = alias_kind(value)
            with self.assertRaisesRegex(NotImplementedError, "independent CPU"):
                value.share_memory_()
            alias.add_(5)
            self.assertEqual(value[-1].item(), 8.)

    def test_shared_views_fail_serialization_instead_of_copying(self):
        import torch
        value = torch.tensor([1., 2., 3.]).share_memory_()
        for view in (value[1:], value[1:].detach(), value[1:].data, value.reshape(3, 1).T):
            self.assertTrue(view.is_shared())
            with self.assertRaisesRegex(NotImplementedError, "shared-memory views"):
                pickle.dumps(view)
        detached = pickle.loads(pickle.dumps(value.detach()))
        detached.add_(2)
        np.testing.assert_array_equal(value.numpy(), [3., 4., 5.])

    def test_spawned_process_writes_are_visible(self):
        import torch
        value = torch.tensor([1.0, 2.0, 3.0])
        returned = value.share_memory_()
        self.assertIs(returned, value)
        self.assertTrue(value.is_shared())

        process = mp.get_context("spawn").Process(
            target=_increment_shared_tensor, args=(value,))
        process.start()
        try:
            # Spawn imports Jittor afresh. Share the suite's cold-start budget;
            # the property under test is visible writes, not import latency.
            seconds = default_timeout()
            process.join(seconds)
            self.assertFalse(process.is_alive(), 'shared-tensor child exceeded %s s' % seconds)
            self.assertEqual(process.exitcode, 0)
            np.testing.assert_allclose(value.numpy(), [6.0, 7.0, 8.0])
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
            process.close()


if __name__ == "__main__":
    unittest.main()
