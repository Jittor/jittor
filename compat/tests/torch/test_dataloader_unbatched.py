"""Unbatched collation and sized iterable datasets share loader semantics."""

import pytest
import torch


class SizedStream(torch.utils.data.IterableDataset):
    def __len__(self):
        return 3

    def __iter__(self):
        return iter((1, 2, 3))


@pytest.mark.parametrize("workers", (0, 2))
@pytest.mark.parametrize("dataset", ([1, 2, 3], SizedStream()))
def test_unbatched_loader_calls_collate_on_each_sample(workers, dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=workers,
        collate_fn=lambda sample: ("sample", sample * 10))
    assert list(loader) == [("sample", 10), ("sample", 20), ("sample", 30)]


@pytest.mark.parametrize("workers", (0, 2))
def test_sized_iterable_dataset_batches_without_indexing(workers):
    loader = torch.utils.data.DataLoader(
        SizedStream(), batch_size=2, num_workers=workers, collate_fn=tuple)
    assert list(loader) == [(1, 2), (3,)]


def test_unbatched_default_preserves_sample_structure():
    samples = [{"value": 1}, {"value": 2}]
    assert list(torch.utils.data.DataLoader(samples, batch_size=None)) == samples
