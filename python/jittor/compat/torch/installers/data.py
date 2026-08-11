"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt

from ..context import registry_for


def _install_torchdata_stateful_dataloader(g, registry=None):
    """Provide torchdata.stateful_dataloader for verl trainer imports.

    Newer torchdata packages may omit the stateful_dataloader namespace while
    verl still imports it.  A single-process fallback can use the installed
    torch.utils.data.DataLoader and expose no-op state_dict hooks.
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

    class StatefulDataLoader(base_loader):
        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            return None

    stateful.StatefulDataLoader = StatefulDataLoader
    if data_mod is not None:
        for name in ("RandomSampler", "SequentialSampler", "BatchSampler", "Sampler"):
            if hasattr(data_mod, name):
                setattr(sampler_mod, name, getattr(data_mod, name))
    _modules["torchdata.stateful_dataloader"] = stateful
    _modules["torchdata.stateful_dataloader.sampler"] = sampler_mod
    setattr(torchdata, "stateful_dataloader", stateful)


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    import types as _types2
    if "torch.utils.data" not in _modules:
        _data = _types2.ModuleType("torch.utils.data")
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
                import random as _random
                n = len(self.data_source)
                if self.replacement:
                    return iter(_random.randrange(n) for _ in range(self.num_samples))
                indices = list(range(n))
                _random.shuffle(indices)
                return iter(indices[:self.num_samples])
            def __len__(self):
                return self.num_samples
        class _SubsetRandomSampler(_Sampler):
            def __init__(self, indices, generator=None):
                self.indices = list(indices)
                self.generator = generator
            def __iter__(self):
                import random as _random
                indices = list(self.indices)
                _random.shuffle(indices)
                return iter(indices)
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
            import numpy as _np
            elem = batch[0]
            if isinstance(elem, jt.Var):
                return jt.stack(list(batch), dim=0)
            if isinstance(elem, _np.ndarray):
                return jt.array(_np.stack(batch))
            if isinstance(elem, (type(0), type(0.0), _np.number)):
                return jt.array(_np.array(batch))
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
                self._batch_iter = iter(loader.batch_sampler)

            def __next__(self):
                batch_indices = next(self._batch_iter)
                return self._loader.collate_fn([self._loader.dataset[i] for i in batch_indices])

        class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
            pass

        class _DataLoader:
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
                self._iterator = _SingleProcessDataLoaderIter(self)
                return self._iterator
            def __len__(self):
                return len(self.batch_sampler)
        for _name, _value in {
            "Dataset": _TorchDataset,
            "IterableDataset": _IterableDataset,
            "TensorDataset": _TensorDataset,
            "ConcatDataset": _ConcatDataset,
            "Subset": _Subset,
            "Sampler": _Sampler,
            "SequentialSampler": _SequentialSampler,
            "RandomSampler": _RandomSampler,
            "SubsetRandomSampler": _SubsetRandomSampler,
            "BatchSampler": _BatchSampler,
            "DistributedSampler": _DistributedSampler,
            "DataLoader": _DataLoader,
            "default_collate": _default_collate,
            "default_convert": lambda x: x,
            "get_worker_info": lambda: None,
        }.items():
            setattr(_data, _name, _value)
        _modules["torch.utils.data"] = _data
        g.utils.data = _data
        _du = _types2.ModuleType("torch.utils.data._utils")
        _duc = _types2.ModuleType("torch.utils.data._utils.collate")
        _duw = _types2.ModuleType("torch.utils.data._utils.worker")
        def _generate_state(base_seed, worker_id):
            import random as _random_worker
            rng = _random_worker.Random(int(base_seed) + int(worker_id))
            return [rng.randrange(0, 2**32) for _ in range(4)]
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
        def _checkpoint(fn, *args, use_reentrant=None, **kwargs):
            return fn(*args, **kwargs)
        _ckpt.checkpoint = _checkpoint
        _modules["torch.utils.checkpoint"] = _ckpt
        g.utils.checkpoint = _ckpt
    _install_torchdata_stateful_dataloader(g, ctx.registry)
