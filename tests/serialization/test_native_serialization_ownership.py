"""Native runtime round trips for the checkpoint implementation move."""

import importlib
import pickle
import sys
import types

import jittor as jt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_execution_mode():
    with jt.flag_scope(use_cuda=jt.introspection.policy.runtime.use_cuda):
        yield


def test_native_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "native.pkl")
    source = {"weight": jt.array([[1.25, -2.0], [3.5, 4.0]]), "epoch": 3}
    jt.save(source, path)
    restored = jt.load(path)
    assert restored["epoch"] == 3
    np.testing.assert_array_equal(restored["weight"].numpy(), source["weight"].numpy())


def test_torch_archive_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    path = str(tmp_path / "weights.pth")
    source = {"weight": jt.array([[1.25, -2.0], [3.5, 4.0]]),
              "steps": jt.array([1, 3, 7]).int32()}
    jt.save(source, path)
    restored = jt.load(path)
    for name in source:
        assert restored[name].dtype == source[name].dtype
        np.testing.assert_array_equal(restored[name].numpy(), source[name].numpy())


def test_historical_rebuild_pickle_roundtrip():
    args = (np.arange(8, dtype=np.int32), 2, (2, 2), (2, 1), False, {})
    payload = b"cjittor_utils.load_pytorch\njittor_rebuild\n"
    payload += pickle.dumps(args, protocol=0)[:-1] + b"R."
    restored = pickle.loads(payload)
    np.testing.assert_array_equal(restored.numpy(), [[2, 3], [4, 5]])
    legacy = importlib.import_module("jittor_utils.load_pytorch")
    canonical = importlib.import_module("jittor.serialization.load_pytorch")
    assert legacy.jittor_rebuild is canonical.jittor_rebuild
    assert pickle.loads(pickle.dumps(legacy.jittor_rebuild)) is canonical.jittor_rebuild


def test_stock_pytorch_loads_exported_scalar_tensor_archive(tmp_path):
    torch = pytest.importorskip("torch")
    save_pytorch = importlib.import_module("jittor.serialization.save_pytorch").save_pytorch
    path = str(tmp_path / "stock-pytorch-scalar.pt")

    with jt.runtime.scope(use_cuda=0, backend_fallback="error"):
        source = {
            "step": jt.array(7).int64(),
            "vector": jt.array([1, 2, 3]).int32(),
            "matrix": jt.array([[1.5, -2.0], [3.0, 4.5]]),
        }
        save_pytorch(path, source)

    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["step"].shape == torch.Size([])
    assert restored["step"].item() == 7
    torch.testing.assert_close(restored["vector"], torch.tensor([1, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(restored["matrix"], torch.tensor([[1.5, -2.0], [3.0, 4.5]]))


def test_save_pytorch_restores_pickle_targets_after_failure(tmp_path):
    pytest.importorskip("torch")
    serializer = importlib.import_module("jittor.serialization.save_pytorch")
    bindings = []
    for target in serializer.targets:
        module = sys.modules[target.__module__]
        bindings.append((module, target.__name__, module.__dict__.get(target.__name__)))

    class FailingPickle:
        def __reduce__(self):
            raise RuntimeError("forced pickle failure")

    path = str(tmp_path / "must-not-be-written.pt")
    with pytest.raises(RuntimeError, match="forced pickle failure"):
        serializer.save_pytorch(path, {"payload": FailingPickle()})

    for module, name, original in bindings:
        assert module.__dict__.get(name) is original


@pytest.mark.parametrize("binding_kind", ["external-owner", "missing"])
def test_save_pytorch_restores_torch_targets_after_failure(binding_kind, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    serializer = importlib.reload(importlib.import_module("jittor.serialization.save_pytorch"))
    targets_before = list(serializer.targets)
    fake_torch = types.ModuleType("torch")
    fake_utils = types.ModuleType("torch._utils")
    shim_owner_name = "_test_jittor_torch_storage_owner"
    shim_owner = types.ModuleType(shim_owner_name)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch._utils", fake_utils)
    monkeypatch.setitem(sys.modules, shim_owner_name, shim_owner)

    missing = object()
    bindings = []
    for target in serializer.targets:
        namespace = fake_utils.__dict__ if target.__module__ == "torch._utils" else fake_torch.__dict__
        if binding_kind == "external-owner":
            if target.__name__ == "_rebuild_tensor_v2":
                def original(*args):
                    return None
                original.__name__ = target.__name__
                original.__qualname__ = target.__name__
                original.__module__ = shim_owner_name
            else:
                original = type(target.__name__, (), {"__module__": shim_owner_name})
            setattr(shim_owner, target.__name__, original)
            namespace[target.__name__] = original
        else:
            original = missing
        bindings.append((namespace, target.__name__, original))

    class FailingPickle:
        def __reduce__(self):
            raise RuntimeError("forced pickle failure")

    path = str(tmp_path / "must-not-be-written.pt")
    with pytest.raises(RuntimeError, match="forced pickle failure"):
        serializer.save_pytorch(path, {"payload": FailingPickle()})
    assert not (tmp_path / "must-not-be-written.pt").exists()
    assert serializer.targets == targets_before

    for namespace, name, original in bindings:
        if original is missing:
            assert name not in namespace
        else:
            assert namespace.get(name) is original
