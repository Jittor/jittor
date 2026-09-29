"""Numeric and dispatch contracts for TensorDataset batched fetching."""

import os
import unittest
import warnings
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]


def tensor(data, **kwargs):
    return torch.tensor(data, device=_DEVICE, **kwargs)


def values(tensor):
    return tensor.detach().clone().cpu().numpy().copy()


class TestTensorDatasetBatch(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "_torch_compat_install_context"),
                         "native dispatch spy is Jittor-specific")
    def test_exact_tensor_dataset_uses_native_batch_fetch(self):
        from jittor.compat.torch.installers import data

        x = tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
        native_getitem = torch.Tensor.getitem
        calls = []

        def observed_getitem(source, index):
            calls.append(tuple(index.shape))
            return native_getitem(source, index)

        with mock.patch.object(torch.Tensor, "getitem", observed_getitem), \
                mock.patch.object(data, "_default_collate", wraps=data._default_collate) as collate:
            batch = next(iter(DataLoader(TensorDataset(x), batch_size=2)))
            np.testing.assert_array_equal(values(batch[0]), [[0.0, 1.0, 2.0],
                                                            [3.0, 4.0, 5.0]])
        self.assertEqual(calls, [(2,)])
        collate.assert_not_called()

    def test_reordered_duplicate_negative_indices_and_gradient(self):
        source = np.arange(24, dtype=np.float32).reshape(6, 4)
        x = tensor(source, requires_grad=True)
        ids = tensor(np.arange(6, dtype=np.int64))
        loader = DataLoader(TensorDataset(x, ids), batch_sampler=[[4, 1, 4, -1], [2]])
        expected_gradient = np.zeros_like(source)
        for indices, batch in zip(([4, 1, 4, -1], [2]), loader):
            self.assertIsInstance(batch, list)
            xb, yb = batch
            self.assertIs(type(xb), torch.Tensor)
            self.assertIs(type(yb), torch.Tensor)
            self.assertEqual(xb.device, x.device)
            self.assertEqual(yb.device, ids.device)
            np.testing.assert_array_equal(values(xb), source[indices])
            np.testing.assert_array_equal(values(yb), np.arange(6)[indices])
            self.assertEqual(yb.dtype, torch.int64)
            self.assertTrue(xb.requires_grad)
            (xb * xb).sum().backward()
            np.add.at(expected_gradient, indices, 2.0 * source[indices])
        np.testing.assert_allclose(values(x.grad), expected_gradient, rtol=1e-6, atol=1e-6)

    def test_batch_is_a_copy_and_tail_preserves_shape_dtype(self):
        source = np.arange(15, dtype=np.float32).reshape(5, 3)
        x = tensor(source)
        batches = list(DataLoader(TensorDataset(x), batch_size=2))
        self.assertEqual([tuple(batch[0].shape) for batch in batches], [(2, 3), (2, 3), (1, 3)])
        self.assertTrue(all(batch[0].dtype == torch.float32 for batch in batches))
        self.assertTrue(all(not batch[0].requires_grad for batch in batches))
        batches[0][0].fill_(-1.0)
        np.testing.assert_array_equal(values(x), source)

    def test_parameter_source_produces_frontend_tensor_batches(self):
        x = torch.nn.Parameter(tensor(np.arange(12, dtype=np.float32).reshape(4, 3)))
        batch = next(iter(DataLoader(TensorDataset(x), batch_size=2)))
        self.assertIs(type(batch[0]), torch.Tensor)
        self.assertEqual(batch[0].device, x.device)
        batch[0].sum().backward()
        np.testing.assert_array_equal(values(x.grad), [[1.0] * 3, [1.0] * 3,
                                                      [0.0] * 3, [0.0] * 3])

    def test_subclass_getitem_is_honored(self):
        class Shifted(TensorDataset):
            def __getitem__(self, index):
                return tuple(value + 100.0 for value in super().__getitem__(index))

        x = tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
        batch = next(iter(DataLoader(Shifted(x), batch_size=2)))
        np.testing.assert_array_equal(values(batch[0]), np.arange(6).reshape(2, 3) + 100.0)

    def test_custom_collate_receives_original_samples(self):
        x = tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
        loader = DataLoader(TensorDataset(x), batch_size=2, collate_fn=lambda samples: samples)
        batch = next(iter(loader))
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], tuple)
        np.testing.assert_array_equal(values(batch[0][0]), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(values(batch[1][0]), [3.0, 4.0, 5.0])

    def test_noncontiguous_source_and_drop_last(self):
        source = np.arange(15, dtype=np.float32).reshape(3, 5).T
        x = tensor(source.T.copy()).permute(1, 0)
        batches = list(DataLoader(TensorDataset(x), batch_size=2, drop_last=True))
        self.assertEqual(len(batches), 2)
        for batch, indices in zip(batches, ([0, 1], [2, 3])):
            np.testing.assert_array_equal(values(batch[0]), source[indices])

    def test_invalid_sampler_indices_keep_scalar_index_errors(self):
        x = tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
        for index in (4, -5):
            with self.subTest(index=index):
                loader = DataLoader(TensorDataset(x), batch_sampler=[[0, index]])
                with self.assertRaises((IndexError, RuntimeError)) as scalar_error:
                    x[index]
                with self.assertRaises(type(scalar_error.exception)):
                    next(iter(loader))

    def test_worker_fetch_preserves_order_and_tail(self):
        source = np.arange(15, dtype=np.float32).reshape(5, 3)
        x = tensor(source)
        for batch in DataLoader(TensorDataset(x), batch_size=2):
            values(batch[0])
        loader = DataLoader(TensorDataset(x), batch_size=2, num_workers=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batches = list(loader)
        self.assertEqual([tuple(batch[0].shape) for batch in batches], [(2, 3), (2, 3), (1, 3)])
        for batch, indices in zip(batches, ([0, 1], [2, 3], [4])):
            np.testing.assert_array_equal(values(batch[0]), source[indices])


if __name__ == "__main__":
    unittest.main()
