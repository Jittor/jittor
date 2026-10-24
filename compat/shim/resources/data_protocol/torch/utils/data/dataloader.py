"""Explicit map DataLoader aliases; unsupported worker entry points fail loud."""
from typing import Any, Callable, List, TypeVar
from jittor.compat.torch.installers.data import (
    _DataLoader as DataLoader, _DatasetKind, _InfiniteConstantSampler,
    _BaseDataLoaderIter, _SingleProcessDataLoaderIter,
)
from ._utils.collate import default_collate, default_convert
from ._utils.worker import get_worker_info

_T = TypeVar('_T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[_T]], Any]


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    raise NotImplementedError('DataLoader worker-process sharding is not implemented')


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        raise NotImplementedError('multiprocessing DataLoader iterators are not implemented')
