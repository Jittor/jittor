"""Persistent Torch BatchNorm state, including an adopted native child."""

import os

import numpy as np
import pytest
import torch

from _release_test_helpers import on_device, public_device_scope, values


_DEVICES = tuple(name for name in os.environ.get(
    "JITTOR_TEST_DEVICES", "cpu").split(",") if name in ("cpu", "cuda"))


@pytest.fixture(params=_DEVICES)
def batchnorm_device(request):
    with public_device_scope(request.param, set_default=True) as name:
        yield name


@pytest.mark.parametrize("rank", (1, 2, 3))
def test_batchnorm_counter_survives_saved_state_and_continues_training(batchnorm_device, tmp_path, rank):
    layer = getattr(torch.nn, "BatchNorm%dd" % rank)
    shape = (4, 3) + (2,) * rank
    inputs = torch.tensor(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100,
                          device=batchnorm_device)
    source = layer(3).to(batchnorm_device).train()
    for _ in range(3):
        source(inputs).sum().item()
    state = source.state_dict()
    counter = state["num_batches_tracked"]
    assert counter.dtype == torch.int64
    assert tuple(counter.shape) == ()
    assert int(counter.item()) == 3
    assert "num_batches_tracked" not in dict(source.named_parameters())
    on_device(counter, batchnorm_device)
    path = tmp_path / "batchnorm.pt"
    torch.save(state, path)
    restored = layer(3).to(batchnorm_device).train()
    restored.load_state_dict(torch.load(path, weights_only=False), strict=True)
    assert int(restored.num_batches_tracked.item()) == 3
    for _ in range(2):
        restored(inputs).sum().item()
    assert int(restored.num_batches_tracked.item()) == 5
    assert int(source.num_batches_tracked.item()) == 3
    on_device(restored.num_batches_tracked, batchnorm_device)


def test_adopted_batchnorm_counter_keeps_native_source_and_external_module_unchanged(batchnorm_device):
    if not hasattr(torch, "_torch_compat_install_context"):
        pytest.skip("checks Jittor frontend ownership")
    import jittor as jt

    source, supplied = jt.nn.BatchNorm2d(3), jt.nn.BatchNorm2d(3)
    source.num_batches_tracked.update(source.num_batches_tracked + 11)
    source_counter, supplied_counter = source.num_batches_tracked, supplied.num_batches_tracked
    original_names = set(source._non_persistent_buffer_names)

    class NativeParent(jt.nn.Module):
        def __init__(self, external):
            self.owned, self.external = source, external

    owner = torch.nn.BatchNorm2d._nn_frontend_owner
    adapted = owner.adapt_class(NativeParent)(supplied)
    assert adapted.external is supplied
    assert adapted.owned is not source
    counter = adapted.owned.state_dict()["num_batches_tracked"]
    assert counter.dtype == torch.int64 and tuple(counter.shape) == ()
    assert int(counter.item()) == 11
    assert source.num_batches_tracked is source_counter
    assert supplied.num_batches_tracked is supplied_counter
    assert set(source._non_persistent_buffer_names) == original_names
    assert "num_batches_tracked" not in source.state_dict()
    assert "num_batches_tracked" not in supplied.state_dict()
    assert tuple(source_counter.shape) == (1,)
    np.testing.assert_array_equal(values(source_counter), [11])
