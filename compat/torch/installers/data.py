"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from typing import Generic as _Generic, TypeVar as _TypeVar

_T_co = _TypeVar("_T_co", covariant=True)

import collections as _collections_data

import jittor as jt

from ..context import registry_for, get_install_context
from ..fidelity import Fidelity, register_fidelity
from ... import stub_policy as _stub_policy_data
from ...diagnostics import EXPECTED, swallowed


class _TorchDataset(_Generic[_T_co]):
    def __getitem__(self, i):
        raise NotImplementedError
    def __add__(self, other):
        return _ConcatDataset([self, other])


class _IterableDataset(_TorchDataset[_T_co]):
    def __iter__(self):
        raise NotImplementedError


class _TensorDataset(_TorchDataset):
    def __init__(self, *tensors):
        self.tensors = tensors
    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)
    def __len__(self):
        return len(self.tensors[0]) if self.tensors else 0


class _ConcatDataset(_TorchDataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    def __getitem__(self, idx):
        import bisect as _bisect
        dataset_idx = _bisect.bisect_right(self.cumulative_sizes, idx)
        prev = self.cumulative_sizes[dataset_idx - 1] if dataset_idx else 0
        return self.datasets[dataset_idx][idx - prev]


class _Subset(_TorchDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class _Sampler(_Generic[_T_co]):
    def __init__(self, data_source=None):
        self.data_source = data_source
    def __iter__(self):
        raise NotImplementedError


class _SequentialSampler(_Sampler):
    def __iter__(self):
        return iter(range(len(self.data_source)))
    def __len__(self):
        return len(self.data_source)


class _RandomSampler(_Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
    @property
    def num_samples(self):
        return len(self.data_source) if self._num_samples is None else self._num_samples
    def __iter__(self):
        """Every index comes from `self.generator` when one was given.

        A sampler handed a generator promises three things: the indices came
        from that generator, it advanced by exactly those draws, and restoring
        its state replays them. Drawing from python's global `random` instead
        -- which is what this did, while storing `self.generator` and never
        reading it -- keeps none of them, and says nothing on the first run: the
        order only comes out different after a resume, which is the run nobody
        is watching.

        `torch.randint` / `torch.randperm` already draw from a generator's own
        stream, so this routes through them rather than growing a second
        sampling owner with its own state rules.
        """
        import random as _random
        n = len(self.data_source)
        if self.generator is None:
            if self.replacement:
                return iter(_random.randrange(n) for _ in range(self.num_samples))
            indices = list(range(n))
            _random.shuffle(indices)
            return iter(indices[:self.num_samples])
        if self.replacement:
            draws = _torch_ns().randint(0, n, (int(self.num_samples),),
                                        generator=self.generator)
            return iter(int(i) for i in draws.tolist())
        order = _torch_ns().randperm(n, generator=self.generator).tolist()
        return iter(int(i) for i in order[:self.num_samples])
    def __len__(self):
        return self.num_samples


def _torch_ns():
    """The installed torch namespace, for the generator-aware factories.

    Imported at call time rather than read out of the module registry:
    publishing names into that registry is the publication layer's business,
    not an installer's (``compat/tests/structure/test_torch_compat_structure.py``
    fails any installer that touches it, as a substring), and a plain call-time
    ``import`` gets the same object the caller sees -- this frontend when it is
    active, the real PyTorch when it is not.
    """
    import torch
    return torch


class _SubsetRandomSampler(_Sampler):
    def __init__(self, indices, generator=None):
        self.indices = list(indices)
        self.generator = generator
    def __iter__(self):
        # Same contract as RandomSampler: a generator that was handed in is the
        # one the permutation comes from.
        import random as _random
        indices = list(self.indices)
        if self.generator is None:
            _random.shuffle(indices)
            return iter(indices)
        order = _torch_ns().randperm(len(indices), generator=self.generator).tolist()
        return iter(indices[int(i)] for i in order)
    def __len__(self):
        return len(self.indices)


class _BatchSampler(_Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


class _DistributedSampler(_Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True,
                 seed=0, drop_last=False):
        import math as _math
        self.dataset = dataset
        self.num_replicas = 1 if num_replicas is None else int(num_replicas)
        self.rank = 0 if rank is None else int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = _math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = _math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        import random as _random
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = _random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        if not self.drop_last:
            padding = self.total_size - len(indices)
            if padding > 0:
                indices += (indices * ((padding + len(indices) - 1) // len(indices)))[:padding]
        else:
            indices = indices[:self.total_size]
        return iter(indices[self.rank:self.total_size:self.num_replicas])
    def __len__(self):
        return self.num_samples
    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _default_collate(batch):
    from ..tensor_state import compatibility_owner
    g = compatibility_owner(jt)
    import numpy as _np
    elem = batch[0]
    if isinstance(elem, jt.Var):
        from ..frontend import tensor_frontend
        with tensor_frontend(g.Var):
            result = jt.stack(list(batch), dim=0)
            if not any(value.requires_grad for value in batch):
                result.requires_grad_(False)
            return result
    if isinstance(elem, (_np.ndarray, _np.generic)):
        values = _np.stack(batch)
        return g.as_tensor(values, dtype=_jittor_dtype_name(values.dtype)).requires_grad_(False)
    if isinstance(elem, bool):
        return g.as_tensor(batch, dtype=g.bool).requires_grad_(False)
    if isinstance(elem, type(0)):
        return g.as_tensor(batch, dtype=g.int64).requires_grad_(False)
    if isinstance(elem, type(0.0)):
        return g.as_tensor(batch, dtype=g.float64).requires_grad_(False)
    if isinstance(elem, (tuple, list)):
        return [_default_collate(list(items)) for items in zip(*batch)]
    if isinstance(elem, dict):
        return {key: _default_collate([d[key] for d in batch]) for key in elem}
    return batch


class _MapDatasetFetcher:
    """Map-style fetch owner; checkpoint/cursor state belongs to torchdata."""

    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            getitems = getattr(self.dataset, "__getitems__", None)
            if getitems:
                data = getitems(possibly_batched_index)
            else:
                data = [self.dataset[index] for index in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        if kind == _DatasetKind.Iterable:
            raise NotImplementedError("IterableDataset fetch protocol is not implemented")
        raise ValueError("unknown DataLoader dataset kind")


class _InfiniteConstantSampler(_Sampler):
    def __iter__(self):
        while True:
            yield None


class _BaseDataLoaderIter:
    """Single-process map protocol used by real, separately installed torchdata.

    No state_dict keys, cursor reconstruction or third-party module replacement
    are implemented here. Unsupported capabilities fail before sampler/RNG use.
    """

    def __init__(self, loader):
        if loader._dataset_kind != _DatasetKind.Map:
            raise NotImplementedError("only map-style DataLoader iterators are implemented")
        if loader.num_workers != 0 or loader.persistent_workers:
            raise NotImplementedError("stateful worker-process protocol is not implemented")
        if loader.pin_memory or loader.pin_memory_device:
            raise NotImplementedError("pinned DataLoader protocol is not implemented")
        if loader.timeout != 0:
            raise ValueError("single-process DataLoader requires timeout=0")
        self._dataset = loader.dataset
        self._shared_seed = None
        self._pg = None
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        dist = _torch_ns().distributed
        if dist.is_available() and dist.is_initialized():
            self._world_size, self._rank = dist.get_world_size(), dist.get_rank()
        else:
            self._world_size, self._rank = 1, 0
        self._pin_memory = False
        self._pin_memory_device = None
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        # Match the upstream iterator's one real generator draw, including the
        # generator object supplied by the caller. Do not invent a Python seed.
        torch = _torch_ns()
        self._base_seed = torch.empty((), dtype=torch.int64).random_(
            generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0

    def __iter__(self):
        return self

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def _next_index(self):
        return next(self._sampler_iter)

    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        return data

    def __len__(self):
        return len(self._index_sampler)

    def __getstate__(self):
        raise NotImplementedError("DataLoader iterators cannot be pickled")


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation,
            self._collate_fn, self._drop_last)

    def _next_data(self):
        return self._dataset_fetcher.fetch(self._next_index())


def _get_worker_info():
    return _torch_ns().utils.data._utils.worker.get_worker_info()


class _DataLoader(_Generic[_T_co]):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, generator=None, prefetch_factor=None,
                 persistent_workers=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = kwargs.get("multiprocessing_context", None)
        self.pin_memory_device = kwargs.get("pin_memory_device", "")
        self.in_order = kwargs.get("in_order", True)
        self._dataset_kind = (_DatasetKind.Iterable if isinstance(dataset, _IterableDataset)
                              else _DatasetKind.Map)
        self._IterableDataset_len_called = None
        self.shuffle = shuffle
        self.collate_fn = (collate_fn if collate_fn is not None else
                           (_default_collate if batch_size is not None or batch_sampler is not None
                            else _default_convert))
        self.worker_init_fn = worker_init_fn
        self.generator = generator
        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
            self.sampler = None
        else:
            self.sampler = sampler if sampler is not None else (
                _RandomSampler(dataset, generator=generator) if shuffle else _SequentialSampler(dataset)
            )
            if batch_size is None:
                if drop_last:
                    raise ValueError("batch_size=None is incompatible with drop_last")
                self.batch_sampler = None
            else:
                self.batch_sampler = _BatchSampler(self.sampler, batch_size, drop_last)
        self._iterator = None
    def __iter__(self):
        self.check_worker_number_rationality()
        self._iterator = _SingleProcessDataLoaderIter(self)
        return self._iterator
    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        return self.batch_sampler if self._auto_collation else self.sampler

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, value):
        if value is not None:
            if self.num_workers == 0:
                raise ValueError("multiprocessing_context requires num_workers > 0")
            raise NotImplementedError("worker-process context protocol is not implemented")
        self.__multiprocessing_context = None

    def check_worker_number_rationality(self):
        # The real torchdata constructor calls this before choosing its worker
        # iterator. Do not pretend our legacy thread pool is worker processes.
        if self.num_workers != 0 or self.persistent_workers:
            raise NotImplementedError("stateful worker-process protocol is not implemented")

    def __len__(self):
        if self._dataset_kind != _DatasetKind.Map:
            raise NotImplementedError("IterableDataset length protocol is not implemented")
        return len(self._index_sampler)


def _checkpoint(fn, *args, use_reentrant=None, **kwargs):
    """Run fn directly: correct values, but no activation recompute.

    jittor has no gradient-checkpointing primitive, so the activation
    memory this API exists to save is NOT saved and `use_reentrant`
    has no meaning here. The result is numerically identical, which is
    why this stays a pass-through rather than an error -- but the
    memory guarantee the caller asked for is absent.
    """
    _stub_policy_data.degraded(
        "torch.utils.checkpoint.checkpoint",
        "the wrapped function is run directly, so activations are kept "
        "and no memory is saved",
        "use_reentrant has no effect on jittor.")
    return fn(*args, **kwargs)


def _default_convert(value):
    return _torch_ns().utils.data._utils.collate.default_convert(value)


_DATA_APIS = {
    "Dataset": _TorchDataset, "IterableDataset": _IterableDataset,
    "TensorDataset": _TensorDataset, "ConcatDataset": _ConcatDataset,
    "Subset": _Subset, "Sampler": _Sampler,
    "SequentialSampler": _SequentialSampler, "RandomSampler": _RandomSampler,
    "SubsetRandomSampler": _SubsetRandomSampler, "BatchSampler": _BatchSampler,
    "DistributedSampler": _DistributedSampler, "DataLoader": _DataLoader,
    "default_collate": _default_collate, "default_convert": _default_convert,
    "get_worker_info": _get_worker_info,
}
for _name, _implementation in _DATA_APIS.items():
    register_fidelity(
        "torch.utils.data." + _name, _implementation, Fidelity.APPROXIMATE,
        "Map-style ordered workers0 batching; genuine generic conversion; "
        "worker-process, IterableDataset and pinned-memory execution explicitly "
        "reject unsupported use. Full TorchData/DataPipe behavior is not provided.",
    )
register_fidelity(
    "torch.utils.checkpoint.checkpoint", _checkpoint, Fidelity.APPROXIMATE,
    "evaluates the function directly; no activation recomputation or memory saving",
)

def install(ctx):
    import types as _types2
    from ..data_protocol import install as install_protocol
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    install_protocol(ctx)
    if "torch.utils.checkpoint" not in _modules:
        _ckpt = _types2.ModuleType("torch.utils.checkpoint")
        _ckpt.checkpoint = _checkpoint
        _modules["torch.utils.checkpoint"] = _ckpt
        g.utils.checkpoint = _ckpt
