"""Native runtime round trips for the checkpoint implementation move."""

import importlib
import pickle
import sys

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
