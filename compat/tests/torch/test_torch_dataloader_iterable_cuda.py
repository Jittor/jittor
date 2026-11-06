"""Iterable and unbatched DataLoader values stay on the CUDA training device."""
import unittest

import numpy as np
import jittor as jt
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


@unittest.skipUnless(jt.has_cuda, "requires a real CUDA device")
class TestIterableDataLoaderCUDA(unittest.TestCase):
    def test_iterable_batches_and_unbatched_samples(self):
        class Stream(torch.utils.data.IterableDataset):
            def __iter__(self):
                for value in (1.0, 2.0, 3.0):
                    yield torch.tensor([value], device="cuda")

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
            batches = list(torch.utils.data.DataLoader(Stream(), batch_size=2))
            self.assertEqual(len(batches), 2)
            self.assertEqual([tuple(batch.shape) for batch in batches], [(2, 1), (1, 1)])
            self.assertTrue(all(batch.device.type == "cuda" for batch in batches))
            np.testing.assert_array_equal(batches[0].cpu().numpy(), [[1.0], [2.0]])
            np.testing.assert_array_equal(batches[1].cpu().numpy(), [[3.0]])

            dropped = list(torch.utils.data.DataLoader(Stream(), batch_size=2, drop_last=True))
            self.assertEqual(len(dropped), 1)
            np.testing.assert_array_equal(dropped[0].cpu().numpy(), [[1.0], [2.0]])

            unbatched = list(torch.utils.data.DataLoader(Stream(), batch_size=None))
            self.assertEqual(len(unbatched), 3)
            self.assertTrue(all(value.device.type == "cuda" for value in unbatched))
            np.testing.assert_array_equal(
                np.stack([value.cpu().numpy() for value in unbatched]),
                [[1.0], [2.0], [3.0]],
            )

        self.assertEqual(jt.core.backend_fallback_count() - before, 0)


@unittest.skipUnless(jt.has_cuda, "requires a real CUDA device")
class TestDefaultCollateCUDA(unittest.TestCase):
    def test_default_collate_and_loader_keep_the_batch_device(self):
        class Rows(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return {"value": torch.tensor([float(index)], device="cuda")}

        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
            direct = torch.utils.data.default_collate(
                [torch.tensor([float(i), float(i) + 0.5], device="cuda") for i in range(3)])
            self.assertEqual(direct.device.type, "cuda")
            np.testing.assert_array_equal(direct.cpu().numpy(), [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]])

            batches = list(torch.utils.data.DataLoader(Rows(), batch_size=2))
            self.assertEqual(len(batches), 2)
            self.assertTrue(all(batch["value"].device.type == "cuda" for batch in batches))
            np.testing.assert_array_equal(batches[0]["value"].cpu().numpy(), [[0.0], [1.0]])
            np.testing.assert_array_equal(batches[1]["value"].cpu().numpy(), [[2.0], [3.0]])

        self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()
