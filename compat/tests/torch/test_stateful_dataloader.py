"""Bounded opt-in DataLoader checkpoint and worker lifecycle contracts."""

import copy
import random

import numpy as np
import pytest

import jittor as jt
import torch
from torchdata.stateful_dataloader import StatefulDataLoader


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, size=12):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def _loader(seed=31, workers=0, **kwargs):
    return StatefulDataLoader(
        _Dataset(), batch_size=2, num_workers=workers,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        collate_fn=list, **kwargs,
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_default_loader_does_not_retain_consumed_batch_plan(workers):
    loader = torch.utils.data.DataLoader(
        _Dataset(64), batch_size=2, num_workers=workers, collate_fn=list,
    )
    iterator = iter(loader)
    assert list(iterator) == [list(range(i, i + 2)) for i in range(0, 64, 2)]
    assert not getattr(iterator, "_batch_plan", None)
    assert not getattr(iterator, "_worker_plan", None)


def test_pending_loaded_state_roundtrips_before_iteration():
    source = _loader(shuffle=True)
    iterator = iter(source)
    next(iterator)
    state = source.state_dict()
    resumed = _loader(seed=99, shuffle=True)
    resumed.load_state_dict(state)
    second_state = resumed.state_dict()
    assert second_state["active"] is True
    assert second_state["cursor"] == state["cursor"]
    assert second_state["batches"] == state["batches"]
    twice_resumed = _loader(seed=777, shuffle=True)
    twice_resumed.load_state_dict(second_state)
    assert list(twice_resumed) == list(iterator)


def test_sampler_generator_restored_before_eager_iterator_creation():
    class _EagerSampler(torch.utils.data.Sampler):
        def __init__(self, generator):
            self.generator = generator

        def __len__(self):
            return 12

        def __iter__(self):
            return iter(torch.randperm(12, generator=self.generator).tolist())

    class _EagerBatchSampler:
        batch_size = 2
        drop_last = False

        def __init__(self, sampler):
            self.sampler = sampler

        def __len__(self):
            return 6

        def __iter__(self):
            indices = list(self.sampler)
            return iter([indices[i:i + 2] for i in range(0, 12, 2)])

    def make(seed):
        sampler = _EagerSampler(torch.Generator().manual_seed(seed))
        return StatefulDataLoader(_Dataset(), batch_sampler=_EagerBatchSampler(sampler),
                                  collate_fn=list)

    state = make(31).state_dict()
    expected = list(make(31))
    resumed = make(99)
    resumed.load_state_dict(state)
    assert list(resumed) == expected


@pytest.mark.parametrize("change", ["dataset", "batch_size", "drop_last", "persistent_workers"])
def test_resume_rejects_changed_dataset_or_loader_configuration(change):
    state = _loader().state_dict()
    args = {"dataset": _Dataset(), "batch_size": 2, "drop_last": False,
            "persistent_workers": False}
    if change == "dataset":
        args[change] = _Dataset(10)
    elif change == "batch_size":
        args[change] = 3
    else:
        args[change] = True
    resumed = StatefulDataLoader(
        **args, generator=torch.Generator().manual_seed(31), collate_fn=list,
    )
    with pytest.raises(ValueError, match="configuration|dataset"):
        resumed.load_state_dict(state)


@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_plan_has_explicit_index_limit(workers):
    loader = _loader(workers=workers, max_stateful_indices=6)
    with pytest.raises(ValueError, match="max_stateful_indices"):
        iter(loader)


def test_stateful_plan_does_not_drain_an_infinite_sampler():
    class _BadSampler:
        batch_size = 1
        drop_last = False

        def __len__(self):
            return 1

        def __iter__(self):
            # The guard proves a broken or dishonest sampler is bounded.
            for count in range(8):
                if count >= 5:
                    raise AssertionError("sampler was drained without a bound")
                yield [0]

    loader = StatefulDataLoader(_Dataset(), batch_sampler=_BadSampler(),
                                collate_fn=list, max_stateful_indices=4)
    with pytest.raises(ValueError, match="batch count|plan|indices"):
        iter(loader)


def test_oversized_random_sampler_is_rejected_before_permutation():
    class _UncalledRandomSampler(torch.utils.data.RandomSampler):
        def __iter__(self):
            raise AssertionError("oversized sampler allocated its permutation")

    dataset = _Dataset(100)
    loader = StatefulDataLoader(
        dataset, batch_size=100, sampler=_UncalledRandomSampler(dataset),
        max_stateful_indices=4, collate_fn=list,
    )
    with pytest.raises(ValueError, match="max_stateful_indices"):
        iter(loader)


@pytest.mark.parametrize("workers", [0, 2])
def test_snapshot_before_first_batch_restores_cursor_and_next_epoch(workers):
    source = _loader(workers=workers, shuffle=True)
    iterator = iter(source)
    state = source.state_dict()
    assert state["cursor"] == 0
    expected = list(iterator)
    expected_next_epoch = list(source)
    resumed = _loader(seed=99, workers=workers, shuffle=True)
    resumed.load_state_dict(state)
    assert list(resumed) == expected
    assert list(resumed) == expected_next_epoch


def test_loaded_state_can_replace_an_active_persistent_iterator():
    source = _loader(workers=2, persistent_workers=True, shuffle=True)
    iterator = iter(source)
    next(iterator)
    state = source.state_dict()
    expected = list(iterator)
    resumed = _loader(seed=99, workers=2, persistent_workers=True, shuffle=True)
    next(iter(resumed))
    resumed.load_state_dict(state)
    assert list(resumed) == expected


def test_mutating_dataset_size_after_iteration_is_rejected():
    source = _loader()
    next(iter(source))
    source.dataset.size = 10
    with pytest.raises(ValueError, match="configuration changed"):
        source.state_dict()


@pytest.mark.parametrize("kind", ["iterable", "mutable", "unsized"])
def test_unsupported_dataset_contracts_are_rejected(kind):
    if kind == "iterable":
        class _Unsupported(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter(range(12))
    elif kind == "mutable":
        class _Unsupported(_Dataset):
            def state_dict(self):
                return {"cursor": 0}
    else:
        class _Unsupported(torch.utils.data.Dataset):
            def __getitem__(self, index):
                return index
    with pytest.raises((ValueError, NotImplementedError), match="finite|mutable"):
        StatefulDataLoader(_Unsupported())


@pytest.mark.parametrize("kwargs", [
    {"in_order": False}, {"snapshot_every_n_steps": 3},
    {"max_stateful_indices": 0}, {"max_stateful_indices": True},
])
def test_unsupported_stateful_options_are_rejected(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        _loader(**kwargs)


def test_serial_state_rejects_worker_payload_and_boolean_cursor():
    source = _loader()
    next(iter(source))
    state = source.state_dict()
    for invalid in ({**state, "workers": [{"seed": 1}]},
                    {**state, "cursor": True},
                    {**state, "batches": [[999]]}):
        with pytest.raises(ValueError):
            _loader().load_state_dict(invalid)


@pytest.mark.parametrize("field,value", [
    ("seed", "bad"), ("initialized", "yes"),
    ("python_rng_state", None), ("numpy_rng_state", None),
])
def test_worker_state_is_validated_before_starting_threads(field, value):
    source = _loader(workers=2)
    iterator = iter(source)
    next(iterator)
    state = source.state_dict()
    list(iterator)
    invalid = copy.deepcopy(state)
    invalid["workers"][0][field] = value
    with pytest.raises(ValueError):
        _loader(workers=2).load_state_dict(invalid)


def test_persistent_workers_reuse_logical_rng_across_epochs():
    class _RandomDataset(_Dataset):
        def __getitem__(self, index):
            info = torch.utils.data.get_worker_info()
            return info.id, info.seed, random.random(), float(np.random.random())

    loader = StatefulDataLoader(
        _RandomDataset(4), num_workers=2, persistent_workers=True,
        generator=torch.Generator().manual_seed(31), collate_fn=list,
    )
    first_iterator = iter(loader)
    first = [row[0] for row in first_iterator]
    second_iterator = iter(loader)
    second = [row[0] for row in second_iterator]
    assert second_iterator is first_iterator
    for worker_id in range(2):
        left = [row for row in first if row[0] == worker_id]
        right = [row for row in second if row[0] == worker_id]
        assert left[0][1] == right[0][1]
        expected = random.Random(left[0][1])
        draws = [expected.random() for _ in range(4)]
        assert [row[2] for row in left + right] == draws
