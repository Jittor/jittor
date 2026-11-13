"""Grouped communication preserves gradients and propagates backend errors."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jittor.compat import collectives
from jittor.compat.torch.installers.tensor.distributed import _ddp_all_reduce_grouped


def test_grouped_means_preserve_gradient_objects_and_independent_storage(monkeypatch):
    gradients = [torch.arange(n, dtype=torch.float32).reshape(shape)
                 for n, shape in [(4, (2, 2)), (6, (2, 3)), (8, (2, 4))]]
    identities = [id(g) for g in gradients]
    calls = []

    def grouped(values, mean=False):
        calls.append(tuple(np.array(value.numpy(), copy=True) for value in values))
        return [value + index if mean else value*2 + index*2 for index, value in enumerate(values)]

    monkeypatch.setattr(collectives, '_world_size', lambda: 2)
    monkeypatch.setattr(collectives, '_nccl_ops', lambda: SimpleNamespace(nccl_all_reduce_many=grouped))
    _ddp_all_reduce_grouped([(i, grad, None) for i, grad in enumerate(gradients)])
    assert len(calls) == 1 and len(calls[0]) == 3
    assert [id(g) for g in gradients] == identities
    for index, grad in enumerate(gradients):
        np.testing.assert_array_equal(grad.numpy(), np.arange(grad.numel()).reshape(grad.shape) + index)


def test_collective_errors_are_not_hidden(monkeypatch):
    def fail(values, **kwargs):
        raise RuntimeError('collective backend failure')
    monkeypatch.setattr(collectives, '_world_size', lambda: 2)
    monkeypatch.setattr(collectives, '_nccl_ops', lambda: SimpleNamespace(nccl_all_reduce_many=fail))
    with pytest.raises(RuntimeError, match='collective backend failure'):
        collectives._all_reduce_mean_many([torch.ones(4)])


def test_single_rank_and_older_backend_keep_the_existing_semantics(monkeypatch):
    values = [torch.ones(4), torch.ones(2, dtype=torch.float64)]
    monkeypatch.setattr(collectives, '_world_size', lambda: 1)
    assert all(a is b for a, b in zip(values, collectives._all_reduce_mean_many(values)))
    monkeypatch.setattr(collectives, '_world_size', lambda: 2)
    monkeypatch.setattr(collectives, '_nccl_ops', lambda: None)
    monkeypatch.setattr(collectives, '_all_reduce_mean', lambda value: value+3)
    for value in collectives._all_reduce_mean_many(values):
        np.testing.assert_array_equal(value.numpy(), np.full(value.shape, 4))


def test_bucketed_means_join_every_bucket_and_preserve_each_shape(monkeypatch):
    gradients = [torch.arange(n, dtype=torch.float32).reshape(shape)
                 for n, shape in [(4, (2, 2)), (5, (5,)), (2, (1, 2))]]
    calls = []
    def start(value, mean=False):
        assert mean
        calls.append(tuple(value.shape))
        return value + 3
    def wait(values):
        assert len(values) == 2
        return [value + 10 for value in values]
    monkeypatch.setattr(collectives, '_world_size', lambda: 2)
    monkeypatch.setattr(collectives, '_nccl_ops', lambda: SimpleNamespace(
        nccl_all_reduce_deferred=start, nccl_wait_many=wait))
    results = collectives._all_reduce_mean_bucketed(gradients, 32)
    assert calls == [(9,), (2,)]
    for value, original in zip(results, gradients):
        assert value.shape == original.shape and value.dtype == original.dtype
        np.testing.assert_array_equal(value.numpy(), original.numpy()+13)
