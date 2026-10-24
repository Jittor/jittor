"""Real private packed-jagged contract; not full PPO gradient/performance proof."""
import importlib.util
import os
import pickle

import numpy as np
import pytest
import jittor as jt
import torch

from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks

HAS_TENSORDICT = importlib.util.find_spec("tensordict") is not None
if not HAS_TENSORDICT and (os.environ.get("JITTOR_REQUIRE_TENSORDICT") == "1"
                           or os.environ.get("JITTOR_REQUIRE_OPTIONAL_DEPS") == "1"):
    raise ModuleNotFoundError("tensordict is required for consolidation verification")


def _resident(value, device):
    assert value.device.type == device
    native = getattr(value, "_var", value)
    native.sync()
    backend, ordinal = jt.core.dispatch_context([native])
    assert backend == device
    assert native.location() == ("device" if device == "cuda" else "cpu")
    if device == "cuda":
        assert ordinal == native.device_id >= 0
        assert native.placement_backend == 1
    else:
        assert native.placement_backend == 0


def _host(value):
    return np.array(value.detach().cpu().numpy(), copy=True)


def _exercise(device, ragged, holes):
    from tensordict import TensorDict
    from tensordict.utils import _rebuild_njt_from_njt

    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        raw = np.arange(21, dtype=np.float32).reshape((7, 3) if ragged == 1 else (3, 7))
        values = torch.tensor(raw, device=device)
        offsets = torch.tensor([0, 3, 7], device=device, dtype=torch.int64)
        lengths = torch.tensor([2, 3], device=device, dtype=torch.int64) if holes else None
        nested = torch.nested.nested_tensor_from_jagged(
            values, offsets, lengths=lengths, jagged_dim=ragged)
        data = TensorDict({"nested": nested, "dense": torch.tensor([2, 5], device=device)},
                          batch_size=[2], device=device)
        packed = data.consolidate()
        assert packed.is_consolidated() and packed.is_locked
        assert not data.is_consolidated()
        rebuilt = packed["nested"]
        assert rebuilt.is_contiguous() is (not holes)
        storage = packed._consolidated["storage"]
        _resident(storage, device)
        base, size = storage.data_ptr(), storage.numel() * storage.element_size()
        leaves = (rebuilt.values(), rebuilt.offsets(), packed["dense"])
        for value in leaves:
            _resident(value, device)
            assert base <= value.data_ptr() < base + size
        np.testing.assert_array_equal(_host(leaves[0]), raw)
        np.testing.assert_array_equal(_host(leaves[1]), [0, 3, 7])
        np.testing.assert_array_equal(_host(leaves[2]), [2, 5])
        if holes:
            _resident(rebuilt.lengths(), device)
            assert base <= rebuilt.lengths().data_ptr() < base + size
            np.testing.assert_array_equal(_host(rebuilt.lengths()), [2, 3])
        if ragged == 1:
            from tensordict._reductions import _reduce_td
            rebuild_transport, arguments = pickle.loads(pickle.dumps(_reduce_td(packed)))
            restored = rebuild_transport(*arguments)
            assert restored.is_consolidated()
            assert restored.is_locked == arguments[0]['cls_metadata'].get('is_locked', False)
            restored_storage = restored._consolidated['storage']
            _resident(restored_storage, device)
            start = restored_storage.data_ptr()
            end = start + restored_storage.numel() * restored_storage.element_size()
            transported = (restored['nested'].values(), restored['nested'].offsets(), restored['dense'])
            for value in transported:
                _resident(value, device)
                assert start <= value.data_ptr() < end
            np.testing.assert_array_equal(_host(transported[0]), raw)
            np.testing.assert_array_equal(_host(transported[1]), [0, 3, 7])
            np.testing.assert_array_equal(_host(transported[2]), [2, 5])
            if holes:
                _resident(restored['nested'].lengths(), device)
                assert start <= restored['nested'].lengths().data_ptr() < end
                np.testing.assert_array_equal(_host(restored['nested'].lengths()), [2, 3])
        leaf = torch.tensor(raw, device=device, requires_grad=True)
        new_offsets = offsets.clone()
        new_lengths = lengths.clone() if holes else None
        view = _rebuild_njt_from_njt(nested, leaf, new_offsets, new_lengths)
        _resident(view.values(), device)
        assert view.values().data_ptr() == leaf.data_ptr()
        assert view.offsets() is new_offsets
        assert not view.requires_grad and not view.values().requires_grad
        assert leaf.grad is None
        leaf.square().sum().backward()
        assert leaf.grad is not None
        _resident(leaf.grad, device)
        np.testing.assert_array_equal(_host(leaf.grad), 2 * raw)
    assert jt.core.backend_fallback_count() == before


@pytest.mark.cpu
@pytest.mark.skipif(not HAS_TENSORDICT, reason="tensordict is not installed")
@pytest.mark.parametrize("ragged", [1, 2])
@pytest.mark.parametrize("holes", [False, True])
def test_tensordict_consolidate_cpu(ragged, holes):
    with jt.flag_scope(use_cuda=0):
        _exercise("cpu", ragged, holes)


@pytest.mark.cuda
@pytest.mark.skipif(not HAS_TENSORDICT, reason="tensordict is not installed")
@pytest.mark.parametrize("ragged", [1, 2])
@pytest.mark.parametrize("holes", [False, True])
def test_tensordict_consolidate_cuda(ragged, holes):
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("No CUDA found")
    with jt.flag_scope(use_cuda=1):
        _exercise("cuda", ragged, holes)
