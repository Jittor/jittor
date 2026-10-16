"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import collections as _collections_data
import concurrent.futures as _futures_data
import copy as _copy_data
import itertools as _itertools_data
import operator as _operator_data
import threading as _threading_data

import jittor as jt

from ..context import registry_for, get_install_context
from ..fidelity import Fidelity, register_fidelity
from ... import stub_policy as _stub_policy_data
from ...diagnostics import EXPECTED, swallowed


def _install_torchdata_stateful_dataloader(g, registry=None):
    """Provide torchdata.stateful_dataloader for verl trainer imports.

    The opt-in loader saves a bounded map-style batch plan and supported RNG
    state. It never publishes no-op checkpoint hooks.
    """
    _modules = registry_for(g, registry).module_map
    import types as _types

    torchdata = _modules.get("torchdata")
    if torchdata is None:
        torchdata = _types.ModuleType("torchdata")
        torchdata.__version__ = "0.0.jittor"
        _modules["torchdata"] = torchdata

    stateful = _types.ModuleType("torchdata.stateful_dataloader")
    sampler_mod = _types.ModuleType("torchdata.stateful_dataloader.sampler")
    data_mod = getattr(getattr(g, "utils", None), "data", None)
    base_loader = getattr(data_mod, "DataLoader", object)
    if base_loader is _DataLoader:
        stateful.StatefulDataLoader = StatefulDataLoader
    else:
        stateful.StatefulDataLoader = _stateful_loader_type(
            base_loader, get_install_context(g))
    if data_mod is not None:
        for name in ("RandomSampler", "SequentialSampler", "BatchSampler", "Sampler"):
            if hasattr(data_mod, name):
                setattr(sampler_mod, name, getattr(data_mod, name))
    _modules["torchdata.stateful_dataloader"] = stateful
    _modules["torchdata.stateful_dataloader.sampler"] = sampler_mod
    setattr(torchdata, "stateful_dataloader", stateful)


class _TorchDataset:
    def __getitem__(self, i):
        raise NotImplementedError
    def __add__(self, other):
        return _ConcatDataset([self, other])


class _IterableDataset(_TorchDataset):
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


class _Sampler:
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
            if self.generator.device.type != "cpu":
                raise RuntimeError("RandomSampler replacement requires a CPU Generator")
            blocks = []
            for _ in range(int(self.num_samples) // 32):
                blocks.extend(_torch_ns().randint(
                    0, n, (32,), device="cpu", generator=self.generator).tolist())
            remainder = int(self.num_samples) % 32
            if remainder:
                blocks.extend(_torch_ns().randint(
                    0, n, (remainder,), device="cpu",
                    generator=self.generator).tolist())
            return iter(int(i) for i in blocks)
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


def _loader_generators(loader):
    """Return the explicit loader and sampler generators without duplicates."""
    result = {}
    loader_generator = getattr(loader, "generator", None)
    if loader_generator is not None:
        result["loader"] = loader_generator
    sampler = getattr(loader, "sampler", None)
    if sampler is None:
        sampler = getattr(loader, "batch_sampler", None)
    for _ in range(4):
        if sampler is None:
            break
        generator = getattr(sampler, "generator", None)
        if generator is not None:
            result["sampler"] = generator
            break
        sampler = getattr(sampler, "sampler", None)
    return result


def _save_generator_states(loader):
    return {
        name: _copy_data.deepcopy(generator.get_state())
        for name, generator in _loader_generators(loader).items()
        if callable(getattr(generator, "get_state", None))
    }


def _restore_generator_states(loader, states):
    for name, state in states.items():
        generator = _loader_generators(loader).get(name)
        if generator is None or not callable(getattr(generator, "set_state", None)):
            raise ValueError("saved DataLoader generator is missing: %s" % name)
        generator.set_state(_copy_data.deepcopy(state))


def _stateful_configuration(loader):
    """Portable shape/configuration guard; dataset contents remain immutable."""
    sampler = loader.sampler
    if sampler is None:
        sampler = getattr(loader.batch_sampler, "sampler", None)
    def type_name(value):
        kind = type(value)
        return kind.__module__ + "." + kind.__qualname__
    return {
        "dataset_type": type_name(loader.dataset),
        "dataset_size": len(loader.dataset),
        "batch_size": loader.batch_size,
        "drop_last": bool(loader.drop_last),
        "num_workers": int(loader.num_workers or 0),
        "persistent_workers": bool(loader.persistent_workers),
        "batch_sampler_type": type_name(loader.batch_sampler),
        "batch_count": len(loader.batch_sampler),
        "sampler_size": len(sampler) if sampler is not None else None,
        "sampler_type": type_name(sampler),
        "sampler_replacement": getattr(sampler, "replacement", None),
        "batch_sampler_size": getattr(loader.batch_sampler, "batch_size", None),
        "batch_sampler_drop_last": getattr(loader.batch_sampler, "drop_last", None),
    }


def _bounded_stateful_plan(loader):
    """Bound both iteration count and batch size, even for dishonest samplers."""
    limit = loader.max_stateful_indices
    expected = len(loader.batch_sampler)
    if expected > limit:
        raise ValueError("DataLoader batch plan exceeds max_stateful_indices=%d" % limit)
    if isinstance(loader.batch_sampler, _BatchSampler):
        # Reject oversized standard samplers before RandomSampler.__iter__
        # allocates a permutation or draws replacement indices.
        planned = min(expected * loader.batch_sampler.batch_size,
                      len(loader.batch_sampler.sampler))
        if planned > limit:
            raise ValueError("DataLoader batch plan exceeds max_stateful_indices=%d" % limit)
    batches = []
    count = 0
    for batch in _itertools_data.islice(iter(loader.batch_sampler), expected + 1):
        if len(batches) >= expected:
            raise ValueError("DataLoader batch count exceeds its declared finite plan")
        indices = list(_itertools_data.islice(iter(batch), limit - count + 1))
        count += len(indices)
        if count > limit:
            raise ValueError("DataLoader batch plan exceeds max_stateful_indices=%d" % limit)
        if not indices:
            raise ValueError("DataLoader stateful batch plan cannot contain empty batches")
        checked = []
        for index in indices:
            try:
                integer = _operator_data.index(index)
            except TypeError as exc:
                raise ValueError("DataLoader stateful indices must be integers") from exc
            if isinstance(index, bool) or not 0 <= integer < len(loader.dataset):
                raise ValueError("DataLoader stateful index is outside the dataset")
            checked.append(integer)
        batches.append(checked)
    if len(batches) != expected:
        raise ValueError("DataLoader batch count does not match its declared finite plan")
    return batches


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


class _BaseDataLoaderIter:
    def __iter__(self):
        return self


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        self._loader = loader
        self._configuration = _stateful_configuration(loader) if loader._stateful_enabled else None
        state = loader._resume_state
        self._generator_states = state.get("generators", {}) if state is not None else {}
        if self._generator_states:
            _restore_generator_states(loader, self._generator_states)
        if state is not None and state.get("active"):
            if int(state.get("num_workers", -1)) != 0:
                raise ValueError("DataLoader state worker count does not match this loader")
            self._batch_plan = list(state["batches"])
            self._cursor = int(state["cursor"])
            self._plan_complete = True
            self._batch_iter = iter(self._batch_plan[self._cursor:])
        else:
            self._batch_plan = _bounded_stateful_plan(loader) if loader._stateful_enabled else []
            self._cursor = 0
            self._plan_complete = loader._stateful_enabled
            self._batch_iter = iter(self._batch_plan) if self._plan_complete else iter(loader.batch_sampler)
        loader._resume_state = None

    def __next__(self):
        batch_indices = next(self._batch_iter)
        batch = self._loader.collate_fn([self._loader.dataset[i] for i in batch_indices])
        self._cursor += 1
        return batch

    def checkpoint_state(self):
        if not self._loader._stateful_enabled:
            raise RuntimeError("checkpoint state requires explicit StatefulDataLoader")
        return {
            "version": 1,
            "active": True,
            "num_workers": 0,
            "batches": _copy_data.deepcopy(self._batch_plan),
            "cursor": self._cursor,
            "workers": [],
            "worker_ids": [],
            "generators": _save_generator_states(self._loader),
        }


class _WorkerInfo:
    def __init__(self, id, num_workers, seed, dataset, worker_init_fn):
        self.id = id
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = dataset
        self.worker_init_fn = worker_init_fn
        self.initialized = False
        self.python_rng_state = None
        self.numpy_rng_state = None


_worker_state = _threading_data.local()
_worker_rng_lock = _threading_data.RLock()


def _get_worker_info():
    return getattr(_worker_state, "info", None)


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    """Real background fetching for DataLoader(num_workers>0).

    This class used to be ``pass``: DataLoader recorded num_workers,
    prefetch_factor, worker_init_fn and persistent_workers and then
    built a single-process iterator regardless, so every input pipeline
    silently ran serially in the training thread -- which reads as "the
    framework is slow" rather than "the flag did nothing".

    Batches are now prepared by ``num_workers`` background workers with
    a bounded look-ahead of ``prefetch_factor`` batches each, delivered
    strictly in order.  The workers are threads, not processes (see the
    one-time warning): datasets here hold jittor Vars, which do not
    survive a fork.  Each submitted batch is assigned to a stable logical
    worker, and batches for each worker are serialized so random transforms
    advance that worker's stream in submission order.
    """

    def __init__(self, loader):
        self._loader = loader
        self._configuration = _stateful_configuration(loader) if loader._stateful_enabled else None
        self._num_workers = max(1, int(loader.num_workers or 0))
        prefetch = loader.prefetch_factor
        self._prefetch = max(1, int(prefetch if prefetch else 2))
        self._timeout = float(loader.timeout or 0) or None
        self._pending = _collections_data.deque()
        state = loader._resume_state
        self._generator_states = state.get("generators", {}) if state is not None else {}
        if state is not None and state.get("active"):
            if int(state.get("num_workers", -1)) != self._num_workers:
                raise ValueError("DataLoader state worker count does not match this loader")
            worker_states = state["workers"]
            if len(worker_states) != self._num_workers:
                raise ValueError("DataLoader state has an invalid worker-state count")
            self._batch_plan = list(state["batches"])
            self._worker_plan = list(state["worker_ids"])
            if len(self._worker_plan) != len(self._batch_plan):
                raise ValueError("DataLoader state has an invalid worker assignment plan")
            self._cursor = int(state["cursor"])
            self._submitted = self._cursor
            self._plan_complete = True
            self._worker_infos = []
            for worker_id, saved in enumerate(worker_states):
                info = _new_worker_info(loader, int(saved["seed"]), worker_id)
                info.seed = int(saved["seed"])
                info.initialized = bool(saved["initialized"])
                info.python_rng_state = _copy_data.deepcopy(saved["python_rng_state"])
                info.numpy_rng_state = _copy_data.deepcopy(saved["numpy_rng_state"])
                self._worker_infos.append(info)
            self._next_worker_id = self._submitted % self._num_workers
            self._batch_iter = iter(self._batch_plan[self._submitted:])
        else:
            # Match PyTorch's DataLoader contract: the loader generator (or the
            # default CPU stream when it is absent) owns the base seed draw.
            if self._generator_states:
                _restore_generator_states(loader, self._generator_states)
            base_seed = int(_torch_ns().randint(
                0, 2**63 - 1, (1,), dtype=_torch_ns().int64,
                generator=loader.generator,
            ).item())
            self._worker_infos = [
                _new_worker_info(loader, base_seed, worker_id)
                for worker_id in range(self._num_workers)
            ]
            self._next_worker_id = 0
            self._cursor = 0
            self._submitted = 0
            self._batch_plan = _bounded_stateful_plan(loader) if loader._stateful_enabled else []
            self._worker_plan = [i % self._num_workers for i in range(len(self._batch_plan))]
            self._plan_complete = loader._stateful_enabled
            self._batch_iter = iter(()) if self._plan_complete else iter(loader.batch_sampler)
        if state is not None and state.get("active") and self._generator_states:
            _restore_generator_states(loader, self._generator_states)
        loader._resume_state = None
        # One single-thread executor per logical worker preserves that
        # worker's task order while still overlapping separate workers.
        self._pools = [
            _futures_data.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="jt-dataloader-%d" % worker_id
            )
            for worker_id in range(self._num_workers)
        ]
        _stub_policy_data.degraded(
            "torch.utils.data.DataLoader(num_workers>0)",
            "batches are prefetched by %d worker THREADS rather than "
            "worker processes" % self._num_workers,
            "Datasets holding jittor Vars cannot be forked; "
            "legacy Python/NumPy RNG isolation serializes dataset/collate "
            "calls; pass num_workers=0 for strictly serial fetching.")
        self._fill()

    def _fetch(self, batch_indices, worker_info, task_state):
        loader = self._loader
        previous_info = getattr(_worker_state, "info", None)
        previous_task_state = getattr(_worker_state, "task_state", None)
        _worker_state.info = worker_info
        _worker_state.task_state = task_state
        try:
            return _run_with_worker_rng(
                worker_info,
                lambda: loader.collate_fn(
                    [loader.dataset[i] for i in batch_indices]
                ),
            )
        finally:
            if previous_info is None:
                if hasattr(_worker_state, "info"):
                    del _worker_state.info
            else:
                _worker_state.info = previous_info
            if previous_task_state is None:
                if hasattr(_worker_state, "task_state"):
                    del _worker_state.task_state
            else:
                _worker_state.task_state = previous_task_state

    def _fill(self):
        want = self._num_workers * self._prefetch
        while len(self._pending) < want:
            if self._plan_complete:
                if self._submitted >= len(self._batch_plan):
                    return
                batch_indices = self._batch_plan[self._submitted]
                worker_id = self._worker_plan[self._submitted]
            else:
                try:
                    batch_indices = next(self._batch_iter)
                except StopIteration:
                    return
                worker_id = self._next_worker_id
            worker_info = self._worker_infos[worker_id]
            worker_pool = self._pools[worker_id]
            self._next_worker_id = (self._next_worker_id + 1) % self._num_workers
            self._submitted += 1
            task_state = {}
            future = worker_pool.submit(
                self._fetch, batch_indices, worker_info, task_state
            )
            self._pending.append((future, worker_id, task_state))

    def __next__(self):
        if not self._pending:
            self._shutdown()
            raise StopIteration
        future, _, _ = self._pending.popleft()
        try:
            batch = future.result(timeout=self._timeout)
        except _futures_data.TimeoutError:
            self._shutdown()
            raise RuntimeError(
                "DataLoader timed out after %s seconds waiting for a "
                "worker batch" % self._timeout)
        self._fill()
        self._cursor += 1
        return batch

    def checkpoint_state(self):
        """Capture the next unconsumed batch and each logical worker's RNG state."""
        if not self._loader._stateful_enabled:
            raise RuntimeError("checkpoint state requires explicit StatefulDataLoader")
        pending = list(self._pending)
        for future, _, _ in pending:
            try:
                future.result(timeout=self._timeout)
            except _futures_data.TimeoutError:
                raise RuntimeError("DataLoader timed out while capturing worker state")

        self._batch_iter = iter(())
        workers = [
            {
                "seed": int(info.seed),
                "initialized": bool(info.initialized),
                "python_rng_state": _copy_data.deepcopy(info.python_rng_state),
                "numpy_rng_state": _copy_data.deepcopy(info.numpy_rng_state),
            }
            for info in self._worker_infos
        ]
        first_pending_worker = set()
        for _, worker_id, task_state in pending:
            if worker_id in first_pending_worker:
                continue
            first_pending_worker.add(worker_id)
            if "worker_before" not in task_state:
                raise RuntimeError("worker did not publish its checkpoint RNG boundary")
            workers[worker_id] = _copy_data.deepcopy(task_state["worker_before"])
        return {
            "version": 1,
            "active": True,
            "num_workers": self._num_workers,
            "batches": _copy_data.deepcopy(self._batch_plan),
            "worker_ids": list(self._worker_plan),
            "cursor": self._cursor,
            "workers": workers,
            "generators": _save_generator_states(self._loader),
        }

    def _reset(self):
        # Persistent workers retain their streams and worker_init_fn lifecycle.
        # Finish old prefetches before assigning the next epoch's batches.
        for future, _, _ in self._pending:
            future.result(timeout=self._timeout)
        self._pending.clear()
        self._batch_plan = _bounded_stateful_plan(self._loader) if self._loader._stateful_enabled else []
        self._worker_plan = [i % self._num_workers for i in range(len(self._batch_plan))]
        self._plan_complete = self._loader._stateful_enabled
        self._batch_iter = iter(()) if self._plan_complete else iter(self._loader.batch_sampler)
        self._cursor = self._submitted = self._next_worker_id = 0
        self._configuration = _stateful_configuration(self._loader) if self._loader._stateful_enabled else None
        self._fill()

    def _shutdown(self, force=False, wait=False):
        pools = getattr(self, "_pools", None)
        if pools is not None and (force or not self._loader.persistent_workers):
            self._pools = None
            for pool in pools:
                pool.shutdown(wait=wait)

    def __del__(self):
        try:
            pools = getattr(self, "_pools", None)
            if pools is not None:
                for pool in pools:
                    pool.shutdown(wait=False)
        except EXPECTED as exc:
            swallowed("torch/installers/data.py __del__: pools = getattr(self, '_pools', None)", exc)


def _new_worker_info(loader, base_seed, worker_id):
    import random as _random_worker
    import numpy as _np_worker

    seed = int(base_seed) + int(worker_id)
    info = _WorkerInfo(
        worker_id, int(loader.num_workers or 0), seed, loader.dataset,
        loader.worker_init_fn,
    )
    info.python_rng_state = _random_worker.Random(seed).getstate()
    info.numpy_rng_state = _np_worker.random.RandomState(
        seed % (2 ** 32)
    ).get_state()
    return info


def _run_with_worker_rng(info, fn):
    """Run one batch with the logical worker's process-global RNG states.

    Jittor's compatibility workers are threads, so Python's ``random`` and
    NumPy's legacy global RandomState cannot be isolated by thread-local
    module state.  Swap both states under one process-wide lock, save the
    updated worker states, and restore the caller's states before returning.
    This covers the common dataset transform contract while keeping the
    explicit warning that these are not PyTorch worker processes.
    """
    import random as _random_worker
    import numpy as _np_worker

    with _worker_rng_lock:
        caller_python_state = _random_worker.getstate()
        caller_numpy_state = _np_worker.random.get_state()
        try:
            _random_worker.setstate(info.python_rng_state)
            _np_worker.random.set_state(info.numpy_rng_state)
            task_state = getattr(_worker_state, "task_state", None)
            if task_state is not None:
                task_state["worker_before"] = {
                    "seed": int(info.seed),
                    "initialized": bool(info.initialized),
                    "python_rng_state": _copy_data.deepcopy(info.python_rng_state),
                    "numpy_rng_state": _copy_data.deepcopy(info.numpy_rng_state),
                }
            if not info.initialized:
                if info.worker_init_fn is not None:
                    info.worker_init_fn(info.id)
                info.initialized = True
            return fn()
        finally:
            info.python_rng_state = _random_worker.getstate()
            info.numpy_rng_state = _np_worker.random.get_state()
            _random_worker.setstate(caller_python_state)
            _np_worker.random.set_state(caller_numpy_state)


class _DataLoader:
    _stateful_enabled = False

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
        self.shuffle = shuffle
        self.collate_fn = collate_fn if collate_fn is not None else _default_collate
        self.worker_init_fn = worker_init_fn
        self.generator = generator
        self._resume_state = None
        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
            self.sampler = None
        else:
            self.sampler = sampler if sampler is not None else (
                _RandomSampler(dataset, generator=generator) if shuffle else _SequentialSampler(dataset)
            )
            self.batch_sampler = _BatchSampler(self.sampler, batch_size, drop_last)
        self._iterator = None
    def __iter__(self):
        if int(self.num_workers or 0) > 0:
            if self.persistent_workers and self._iterator is not None and self._resume_state is None:
                self._iterator._reset()
            else:
                self._iterator = _MultiProcessingDataLoaderIter(self)
        else:
            self._iterator = _SingleProcessDataLoaderIter(self)
        return self._iterator
    def __len__(self):
        return len(self.batch_sampler)

    def checkpoint_state(self):
        if not self._stateful_enabled:
            raise RuntimeError("checkpoint state requires explicit StatefulDataLoader")
        configuration = _stateful_configuration(self)
        if self._resume_state is not None:
            if self._resume_state["configuration"] != configuration:
                raise ValueError("DataLoader dataset or loader configuration changed after loading state")
            return _copy_data.deepcopy(self._resume_state)
        iterator = self._iterator
        if iterator is None or not hasattr(iterator, "checkpoint_state"):
            return {
                "version": 1,
                "active": False,
                "num_workers": int(self.num_workers or 0),
                "batches": [],
                "cursor": 0,
                "workers": [],
                "worker_ids": [],
                "generators": _save_generator_states(self),
                "configuration": configuration,
            }
        if configuration != iterator._configuration:
            raise ValueError("DataLoader dataset or loader configuration changed during iteration")
        state = iterator.checkpoint_state()
        state["configuration"] = configuration
        return state

    def load_checkpoint_state(self, state):
        if not self._stateful_enabled:
            raise RuntimeError("checkpoint state requires explicit StatefulDataLoader")
        if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 1:
            raise ValueError("unsupported DataLoader checkpoint state")
        if not isinstance(state.get("active"), bool):
            raise ValueError("DataLoader state has an invalid active flag")
        if type(state.get("num_workers")) is not int or state["num_workers"] != int(self.num_workers or 0):
            raise ValueError("DataLoader state worker count does not match this loader")
        if state.get("configuration") != _stateful_configuration(self):
            raise ValueError("DataLoader state dataset or loader configuration does not match")
        if not isinstance(state.get("batches"), list) or type(state.get("cursor")) is not int:
            raise ValueError("DataLoader state has invalid batch cursor data")
        if state["cursor"] < 0 or state["cursor"] > len(state["batches"]):
            raise ValueError("DataLoader state cursor is outside the saved batch plan")
        if not isinstance(state.get("workers"), list):
            raise ValueError("DataLoader state has invalid worker data")
        if not isinstance(state.get("generators", {}), dict):
            raise ValueError("DataLoader state has invalid generator data")
        if any(name not in ("loader", "sampler") for name in state.get("generators", {})):
            raise ValueError("DataLoader state has an invalid generator name")
        total_indices = 0
        for batch in state["batches"]:
            if not isinstance(batch, list) or not batch:
                raise ValueError("DataLoader state has an invalid batch plan")
            total_indices += len(batch)
            if total_indices > self.max_stateful_indices:
                raise ValueError("DataLoader state exceeds max_stateful_indices")
            if any(type(index) is not int or not 0 <= index < len(self.dataset) for index in batch):
                raise ValueError("DataLoader state index is outside the dataset")
        if state["active"] and len(state["batches"]) != len(self.batch_sampler):
            raise ValueError("DataLoader state batch count does not match loader configuration")
        if not state["active"]:
            if state["batches"] or state["cursor"] or state["workers"]:
                raise ValueError("inactive DataLoader state must not contain iterator data")
            if state.get("worker_ids") not in (None, []):
                raise ValueError("inactive DataLoader state cannot contain worker ids")
        elif int(state.get("num_workers", 0)) > 0:
            if not isinstance(state.get("worker_ids"), list) or len(state["worker_ids"]) != len(state["batches"]):
                raise ValueError("DataLoader state has invalid worker assignment data")
            if any(
                type(worker_id) is not int or worker_id < 0
                or worker_id >= int(state["num_workers"])
                for worker_id in state["worker_ids"]
            ):
                raise ValueError("DataLoader state has an invalid worker id")
            if len(state["workers"]) != int(state["num_workers"]):
                raise ValueError("DataLoader state has an invalid worker-state count")
            import random as _random_worker
            import numpy as _np_worker
            for worker in state["workers"]:
                if not isinstance(worker, dict) or type(worker.get("seed")) is not int:
                    raise ValueError("DataLoader state has an invalid worker seed")
                if not isinstance(worker.get("initialized"), bool):
                    raise ValueError("DataLoader state has an invalid worker initialized flag")
                try:
                    _random_worker.Random().setstate(worker.get("python_rng_state"))
                    _np_worker.random.RandomState().set_state(worker.get("numpy_rng_state"))
                except (TypeError, ValueError, IndexError, KeyError, OverflowError) as exc:
                    raise ValueError("DataLoader state has invalid worker RNG data") from exc
        elif state.get("worker_ids") not in (None, []) or state["workers"]:
            raise ValueError("single-process DataLoader state cannot contain worker data")
        iterator = self._iterator
        if iterator is not None and hasattr(iterator, "_shutdown"):
            iterator._shutdown(force=True, wait=True)
        self._iterator = None
        self._resume_state = _copy_data.deepcopy(state)
        return None


def _generate_state(base_seed, worker_id):
    import random as _random_worker
    rng = _random_worker.Random(int(base_seed) + int(worker_id))
    return [rng.randrange(0, 2**32) for _ in range(4)]


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
    return value


class _StatefulLoaderState:
    """Bounded immutable map-style data with Python/NumPy legacy worker RNG.

    Dataset-local mutable state, custom RNG objects and worker Torch RNG are
    outside this thread-worker contract. Dataset state hooks and iterable data
    are rejected rather than silently pretending to checkpoint that state.
    """
    _stateful_enabled = True

    def __init__(self, *args, snapshot_every_n_steps=1, in_order=True,
                 max_stateful_indices=100000, **kwargs):
        if in_order is not True:
            raise NotImplementedError("StatefulDataLoader requires in_order=True")
        if type(snapshot_every_n_steps) is not int or snapshot_every_n_steps != 1:
            raise NotImplementedError("StatefulDataLoader requires snapshot_every_n_steps=1")
        if type(max_stateful_indices) is not int or max_stateful_indices <= 0:
            raise ValueError("max_stateful_indices must be a positive integer")
        self.max_stateful_indices = max_stateful_indices
        self.snapshot_every_n_steps = snapshot_every_n_steps
        self.in_order = in_order
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, _IterableDataset):
            raise NotImplementedError("StatefulDataLoader only supports finite map-style datasets")
        if any(callable(getattr(self.dataset, name, None)) for name in ("state_dict", "load_state_dict")):
            raise NotImplementedError("StatefulDataLoader does not checkpoint mutable dataset state")
        try:
            _stateful_configuration(self)
        except TypeError as exc:
            raise ValueError("StatefulDataLoader requires a sized finite map-style batch sampler") from exc

    def state_dict(self):
        return _DataLoader.checkpoint_state(self)

    def load_state_dict(self, state_dict):
        return _DataLoader.load_checkpoint_state(self, state_dict)


class StatefulDataLoader(_StatefulLoaderState, _DataLoader):
    pass


def _stateful_loader_type(base_loader, context):
    """Retain an explicitly supplied loader base without copying its behavior."""
    adapters = context.state.setdefault("stateful_loader_types", {})
    if base_loader not in adapters:
        adapters[base_loader] = type("StatefulDataLoader", (_StatefulLoaderState, base_loader),
                                     {"__module__": "torchdata.stateful_dataloader"})
    return adapters[base_loader]


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
        "Jittor datasets and ordered batching; workers use threads rather than "
        "processes, default_convert preserves its input, and pinned-memory "
        "and complete Torch worker semantics are not provided",
    )
register_fidelity(
    "torch.utils.checkpoint.checkpoint", _checkpoint, Fidelity.APPROXIMATE,
    "evaluates the function directly; no activation recomputation or memory saving",
)
register_fidelity(
    "torchdata.stateful_dataloader.StatefulDataLoader", StatefulDataLoader,
    Fidelity.APPROXIMATE,
    "ordered data loading; iterator cursor and supported RNG state are restored, "
    "with at most 100000 indices by default and immutable map-style datasets; "
    "worker Python/NumPy legacy RNG is supported, worker Torch/custom RNG and "
    "mutable dataset state are unsupported; threads are not process-equivalent",
)


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    import types as _types2
    if "torch.utils.data" not in _modules:
        _data = _types2.ModuleType("torch.utils.data")

        for _name, _value in _DATA_APIS.items():
            setattr(_data, _name, _value)
        _modules["torch.utils.data"] = _data
        g.utils.data = _data
        _du = _types2.ModuleType("torch.utils.data._utils")
        _duc = _types2.ModuleType("torch.utils.data._utils.collate")
        _duw = _types2.ModuleType("torch.utils.data._utils.worker")
        _duc.default_collate = _default_collate
        _du.collate = _duc
        _duw._generate_state = _generate_state
        _du.worker = _duw
        _modules["torch.utils.data._utils"] = _du
        _modules["torch.utils.data._utils.collate"] = _duc
        _modules["torch.utils.data._utils.worker"] = _duw
        _data._utils = _du
        _dist_data = _types2.ModuleType("torch.utils.data.distributed")
        _dist_data.DistributedSampler = _DistributedSampler
        _modules["torch.utils.data.distributed"] = _dist_data
        _data.distributed = _dist_data
        _dataset_mod = _types2.ModuleType("torch.utils.data.dataset")
        for _name in ("Dataset", "IterableDataset", "TensorDataset", "ConcatDataset", "Subset"):
            setattr(_dataset_mod, _name, getattr(_data, _name))
        _modules["torch.utils.data.dataset"] = _dataset_mod
        _data.dataset = _dataset_mod
        _sampler_mod = _types2.ModuleType("torch.utils.data.sampler")
        for _name in ("Sampler", "SequentialSampler", "RandomSampler", "SubsetRandomSampler", "BatchSampler", "DistributedSampler"):
            setattr(_sampler_mod, _name, getattr(_data, _name))
        _modules["torch.utils.data.sampler"] = _sampler_mod
        _data.sampler = _sampler_mod
        _dataloader_mod = _types2.ModuleType("torch.utils.data.dataloader")
        _dataloader_mod.DataLoader = _DataLoader
        _dataloader_mod.default_collate = _default_collate
        _dataloader_mod._DatasetKind = type("_DatasetKind", (), {"Iterable": 0, "Map": 1})
        _dataloader_mod._BaseDataLoaderIter = _BaseDataLoaderIter
        _dataloader_mod._SingleProcessDataLoaderIter = _SingleProcessDataLoaderIter
        _dataloader_mod._MultiProcessingDataLoaderIter = _MultiProcessingDataLoaderIter
        _modules["torch.utils.data.dataloader"] = _dataloader_mod
        _data.dataloader = _dataloader_mod
    else:
        g.utils.data = _modules["torch.utils.data"]
    if "torch.utils.checkpoint" not in _modules:
        _ckpt = _types2.ModuleType("torch.utils.checkpoint")
        _ckpt.checkpoint = _checkpoint
        _modules["torch.utils.checkpoint"] = _ckpt
        g.utils.checkpoint = _ckpt
    _install_torchdata_stateful_dataloader(g, ctx.registry)
