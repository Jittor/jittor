"""Formatted datasets fetch one column batch instead of rebuilding rows."""

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader


class _FormattedDataset:
    _format_type = "torch"

    def __init__(self):
        self.row_calls = 0
        self.list_calls = 0
        self.x = np.arange(16, dtype=np.float32).reshape(8, 2)
        self.y = np.arange(8, dtype=np.int64)

    def __len__(self):
        return 8

    def __getitem__(self, key):
        if isinstance(key, list):
            self.list_calls += 1
            return {"x": torch.tensor(self.x[key]), "y": torch.tensor(self.y[key])}
        self.row_calls += 1
        return {"x": torch.tensor(self.x[key]), "y": torch.tensor(self.y[key])}


class _GenericDataset:
    def __init__(self):
        self.row_calls = 0
        self.list_calls = 0
        self.x = np.arange(16, dtype=np.float32).reshape(8, 2)

    def __len__(self):
        return 8

    def __getitem__(self, key):
        if isinstance(key, list):
            self.list_calls += 1
            return [{"x": torch.tensor(row)} for row in self.x[key]]
        self.row_calls += 1
        return {"x": torch.tensor(self.x[key])}


class TestFormattedColumnBatch(unittest.TestCase):
    def test_formatted_dataset_is_fetched_as_one_column_batch(self):
        dataset = _FormattedDataset()
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        self.assertEqual(dataset.list_calls, 1)
        self.assertEqual(dataset.row_calls, 0)
        np.testing.assert_array_equal(batch["x"].numpy(), dataset.x[:4])
        np.testing.assert_array_equal(batch["y"].numpy(), dataset.y[:4])

    def test_generic_dataset_keeps_row_fetch_and_collate(self):
        dataset = _GenericDataset()
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        self.assertEqual(dataset.list_calls, 0)
        self.assertEqual(dataset.row_calls, 4)
        np.testing.assert_array_equal(batch["x"].numpy(), dataset.x[:4])


if __name__ == "__main__":
    unittest.main()
