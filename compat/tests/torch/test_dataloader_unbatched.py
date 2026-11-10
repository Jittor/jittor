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


class BatchMap(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return ('scalar', index)

    def __getitems__(self, indices):
        return [('batch', index * 10) for index in indices]


@pytest.mark.parametrize('workers', (0, 2))
def test_map_batch_hook_preserves_sampler_order_duplicates_and_tail(workers):
    loader = torch.utils.data.DataLoader(
        BatchMap(), batch_sampler=[[3, 1, 3], [2]], num_workers=workers,
        collate_fn=tuple)
    assert list(loader) == [(('batch', 30), ('batch', 10), ('batch', 30)),
                            (('batch', 20),)]


def test_unbatched_loader_does_not_call_map_batch_hook():
    loader = torch.utils.data.DataLoader(BatchMap(), batch_size=None, collate_fn=tuple)
    assert list(loader) == [('scalar', index) for index in range(4)]


def test_batch_hook_exception_is_not_replaced_by_scalar_fetch():
    class BrokenBatch(BatchMap):
        def __getitems__(self, indices):
            raise RuntimeError('batch fetch failed')

    with pytest.raises(RuntimeError, match='batch fetch failed'):
        next(iter(torch.utils.data.DataLoader(BrokenBatch(), batch_size=2)))


def test_iterable_loader_does_not_call_map_batch_hook():
    class StreamWithHook(SizedStream):
        def __getitems__(self, indices):
            raise AssertionError('iterable streams must be consumed through their iterator')

    loader = torch.utils.data.DataLoader(StreamWithHook(), batch_size=2, collate_fn=tuple)
    assert list(loader) == [(1, 2), (3,)]
