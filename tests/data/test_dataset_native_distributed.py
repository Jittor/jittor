import jittor as jt
import numpy as np
import pytest

from jittor import distributed as dist
from jittor.dataset import Dataset
from jittor.dataset.dataset import _dataset_worker_seed
from _helpers.child_process import run_child_script


class _IndexDataset(Dataset):
    def __init__(self, total_len=12, batch_size=6, shuffle=False, seed=1):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=0,
                         seed=seed)
        self.set_attrs(total_len=total_len)

    def __getitem__(self, index):
        return index


class _IterableOnlyDataset(Dataset):
    def __iter__(self):
        yield from range(4)


class _IterableMixin:
    def __iter__(self):
        yield from range(4)


class _MixedIterableDataset(_IterableMixin, Dataset):
    pass


def test_iterable_style_dataset_fails_closed_for_native_distributed(
        monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    dataset = _IterableOnlyDataset(batch_size=2, num_workers=0)

    with pytest.raises(NotImplementedError, match="Iterable-style Dataset"):
        list(dataset)


def test_mixin_iterable_style_dataset_fails_closed_for_native_distributed(
        monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    dataset = _MixedIterableDataset(batch_size=2, num_workers=0)

    with pytest.raises(NotImplementedError, match="Iterable-style Dataset"):
        list(dataset)


def test_iterable_style_dataset_remains_usable_without_distribution(
        monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    dataset = _IterableOnlyDataset(batch_size=2, num_workers=0)

    assert list(dataset) == [0, 1, 2, 3]


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


@pytest.mark.parametrize("rank", [1, 2])
def test_native_distributed_workers_start_with_generated_seed(monkeypatch, rank):
    # These ranks produced uint32 seeds above INT32_MAX and crashed before
    # sending a batch when the native seed setter received them unchanged.
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    with jt.flag_scope(use_cuda=0):
        dataset = _IndexDataset(total_len=16, batch_size=8, seed=20261012)
        dataset.set_attrs(num_workers=2, persistent_workers=False)
        values = np.concatenate([batch.numpy() for batch in dataset])
    expected = (np.arange(2)[:, None] * 8 + rank * 2 + np.arange(2)).reshape(-1)
    np.testing.assert_array_equal(values, expected)


def test_dataset_reset_wakes_worker_announcing_into_full_queue():
    result = run_child_script('''
import faulthandler
import jittor as jt
from jittor.dataset import Dataset
from jittor.dataset.dataset import mp
jt.flags.use_cuda = 0
ready = mp.Event()
calls = mp.Value('i', 0)
original = jt.RingBuffer
class FullQueue:
    def __init__(self, size):
        self.inner = original(16)
    def __getattr__(self, name):
        return getattr(self.inner, name)
    def push(self, value):
        with calls.get_lock():
            calls.value += 1
            if calls.value == 2:
                ready.set()
        return self.inner.push(value)
jt.RingBuffer = lambda size: FullQueue(size) if size == 2048 else original(size)
class Source(Dataset):
    def __init__(self):
        super().__init__(batch_size=4, num_workers=2)
        self.set_attrs(total_len=8, keep_numpy_array=True)
    def __getitem__(self, index):
        return index
faulthandler.dump_traceback_later(30, exit=True)
ds = Source()
ds._init_workers(ds._get_index_list())
assert ready.wait(10)
ds.set_epoch(1)
assert not hasattr(ds, 'workers')
faulthandler.cancel_dump_traceback_later()
print('RESET_OK')
''', text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESET_OK" in result.stdout
    assert "RuntimeError: stop" not in result.stderr


@pytest.mark.parametrize("failure", ["error", "exit"])
def test_dataset_reset_reports_worker_failure(failure):
    source = '''
import faulthandler
import os
import jittor as jt
from jittor.dataset import Dataset
from jittor.dataset.dataset import mp
jt.flags.use_cuda = 0
ready = mp.Event()
class Source(Dataset):
    def __init__(self):
        super().__init__(batch_size=4, num_workers=2)
        self.set_attrs(total_len=8, keep_numpy_array=True)
    def __getitem__(self, index):
        if index >= 4:
            ready.set()
            FAILURE
        return index
faulthandler.dump_traceback_later(30, exit=True)
ds = Source()
ds._init_workers(ds._get_index_list())
assert ready.wait(10)
try:
    ds.reset()
except RuntimeError as error:
    assert EXPECTED in str(error), str(error)
else:
    raise AssertionError('reset concealed a failed worker')
assert not hasattr(ds, 'workers')
faulthandler.cancel_dump_traceback_later()
print('FAILURE_REPORTED')
'''
    if failure == "error":
        source = source.replace("            FAILURE\n", '            raise ValueError("reset data error")\n')
        source = source.replace("EXPECTED", '"ValueError: reset data error"')
    else:
        source = source.replace("            FAILURE\n", "            os._exit(3)\n")
        source = source.replace("EXPECTED", '"exit code 3"')
    result = run_child_script(source, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REPORTED" in result.stdout


def test_dataset_workers_observe_new_epoch_after_early_reset():
    class EpochDataset(_IndexDataset):
        def __getitem__(self, index):
            return index + 100 * self.epoch_id

    with jt.flag_scope(use_cuda=0):
        dataset = EpochDataset(total_len=32, batch_size=4, seed=20261012)
        dataset.set_attrs(num_workers=2)
        try:
            next(iter(dataset))
            for epoch in (1, 2, 3):
                dataset.set_epoch(epoch)
                values = np.concatenate([batch.numpy() for batch in dataset])
                np.testing.assert_array_equal(np.sort(values), np.arange(32) + 100 * epoch)
        finally:
            dataset.reset()
