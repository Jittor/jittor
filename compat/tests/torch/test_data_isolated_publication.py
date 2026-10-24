"""Isolated publication contracts; genuine torchdata import needs activation."""
import sys
import types

import pytest

from jittor.compat.torch.context import InstallContext, ModuleRegistry
from jittor.compat.torch.installers import data as owner
from jittor.compat.torch import data_protocol


def _context():
    root = types.ModuleType('_isolated_data_publication_target')
    return InstallContext(root, ModuleRegistry(root, {}))


def _torch_modules():
    return {name: module for name, module in sys.modules.items()
            if name == 'torch' or name.startswith('torch.')
            or name == 'torchdata' or name.startswith('torchdata.')}


def test_isolated_data_publication_preserves_single_owners_and_global_imports():
    context = _context()
    before = _torch_modules()
    owner.install(context)
    data = context.registry.get('torch.utils.data')
    assert context.target_namespace.utils.data is data
    for name, implementation in owner._DATA_APIS.items():
        assert vars(data)[name] is implementation
    assert data.dataset.Dataset is owner._TorchDataset
    assert data.sampler.Sampler is owner._Sampler
    assert data.sampler.BatchSampler is owner._BatchSampler
    assert data.dataloader._DatasetKind is owner._DatasetKind
    assert data.dataloader._BaseDataLoaderIter is owner._BaseDataLoaderIter
    assert data.dataloader._SingleProcessDataLoaderIter is owner._SingleProcessDataLoaderIter
    assert data._utils.fetch._MapDatasetFetcher is owner._MapDatasetFetcher
    assert context.target_namespace.utils.checkpoint.checkpoint is owner._checkpoint
    assert '__file__' not in vars(data)
    assert data.__spec__.loader is None
    assert not any(name == 'torchdata' or name.startswith('torchdata.')
                   for name in context.registry._modules)
    assert 'StatefulDataLoader' not in vars(data)
    assert _torch_modules() == before
    assert context.state == {}


def test_isolated_data_publication_is_idempotent_without_recreating_behavior():
    first, second = _context(), _context()
    owner.install(first)
    snapshot = dict(first.registry._modules)
    owner.install(first)
    owner.install(second)
    assert first.registry._modules == snapshot
    a, b = first.target_namespace.utils.data, second.target_namespace.utils.data
    assert a is not b
    assert a.DataLoader is b.DataLoader is owner._DataLoader
    assert a.dataloader._BaseDataLoaderIter is b.dataloader._BaseDataLoaderIter
    assert a._utils.fetch._MapDatasetFetcher is b._utils.fetch._MapDatasetFetcher
    fetch = a._utils.fetch._MapDatasetFetcher([2, 3, 4], False, lambda value: (value,), False)
    assert fetch.fetch(1) == (3,)


def test_isolated_data_publication_does_not_fake_import_capabilities():
    context = _context()
    data = data_protocol.install(context)
    for name in ('IterDataPipe', 'MapDataPipe', 'graph', 'datapipes'):
        with pytest.raises(NotImplementedError, match='real Torch namespace activation'):
            getattr(data, name)
    with pytest.raises(AttributeError):
        getattr(data, 'StatefulDataLoader')
    foreign = object()
    data.DataLoader = foreign
    with pytest.raises(RuntimeError, match='foreign owner'):
        data_protocol.install(context)
    assert data.DataLoader is foreign
