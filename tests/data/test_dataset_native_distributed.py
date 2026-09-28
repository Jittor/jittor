import numpy as np
import pytest

from jittor import distributed as dist
from jittor.dataset import Dataset
from jittor.dataset.dataset import _dataset_worker_seed


class _IndexDataset(Dataset):
    def __init__(self, total_len=12, batch_size=6, shuffle=False, seed=1):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=0,
                         seed=seed)
        self.set_attrs(total_len=total_len)

    def __getitem__(self, index):
        return index


def test_dataset_uses_native_rank_to_partition_global_batches(monkeypatch):
    expected = {
        0: [0, 1, 2, 6, 7, 8],
        1: [3, 4, 5, 9, 10, 11],
    }
    shards = {}

    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    for rank in (0, 1):
        monkeypatch.setattr(dist, "get_rank", lambda rank=rank: rank)
        dataset = _IndexDataset()
        indices = dataset._get_index_list()
        shards[rank] = indices.tolist()
        assert dataset.real_batch_size == 3
        assert dataset.batch_len == 2
        assert shards[rank] == expected[rank]

    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(12))


def test_dataset_pads_uneven_final_batch_with_repeat_samples(monkeypatch):
    expected = {
        0: [0, 1, 2, 6, 6, 6],
        1: [3, 4, 5, 6, 6, 6],
    }
    shards = {}
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    for rank in (0, 1):
        monkeypatch.setattr(dist, "get_rank", lambda rank=rank: rank)
        dataset = _IndexDataset(total_len=7, batch_size=6)
        indices = dataset._get_index_list()
        shards[rank] = indices.tolist()
        assert dataset.batch_len == 2
        assert shards[rank] == expected[rank]

    assert sorted(set(shards[0] + shards[1])) == list(range(7))
    assert set(shards[0]) & set(shards[1]) == {6}


def test_dataset_rejects_global_batch_smaller_than_world_size(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    dataset = _IndexDataset(total_len=8, batch_size=1)

    with pytest.raises(ValueError, match=r"distributed world_size \(2\)"):
        dataset._get_index_list()


def test_dataset_without_distributed_group_keeps_full_index_list(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    dataset = _IndexDataset(total_len=8, batch_size=4)

    np.testing.assert_array_equal(dataset._get_index_list(), np.arange(8))
    assert dataset.real_batch_size == 4
    assert dataset.batch_len == 2


def test_dataset_set_epoch_repeats_common_shuffle_then_shards(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    epoch_shards = {}

    for epoch in (3, 4):
        shards = []
        for rank in (0, 1):
            monkeypatch.setattr(dist, "get_rank", lambda rank=rank: rank)
            dataset = _IndexDataset(
                total_len=24, batch_size=8, shuffle=True, seed=17)
            assert dataset.set_epoch(epoch) is dataset
            shards.append(dataset._get_index_list().tolist())
        assert set(shards[0]).isdisjoint(shards[1])
        assert sorted(shards[0] + shards[1]) == list(range(24))
        epoch_shards[epoch] = shards

    assert epoch_shards[3] != epoch_shards[4]


@pytest.mark.parametrize("epoch, error", [(-1, ValueError), (1.5, TypeError)])
def test_dataset_set_epoch_rejects_invalid_epoch(epoch, error):
    dataset = _IndexDataset()
    with pytest.raises(error):
        dataset.set_epoch(epoch)


def test_dataset_worker_seed_is_repeatable_and_rank_specific():
    seed = _dataset_worker_seed(17, 3, 0, 2)
    assert seed == _dataset_worker_seed(17, 3, 0, 2)
    assert seed != _dataset_worker_seed(17, 3, 1, 2)
    assert seed != _dataset_worker_seed(17, 4, 0, 2)
    assert seed != _dataset_worker_seed(17, 3, 0, 3)
