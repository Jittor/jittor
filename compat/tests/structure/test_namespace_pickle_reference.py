"""Exercise the real namespace's module-reference contract without native imports."""
import importlib.util
from pathlib import Path
import pickle
import sys
import threading
import types

import pytest


@pytest.fixture
def frontend(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "torch" / "namespace.py"
    spec = importlib.util.spec_from_file_location("_namespace_pickle_contract", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    owner = types.ModuleType("unpublished_native_owner")
    owner.unserializable_lock = threading.RLock()
    value = module.TorchNamespace(owner)
    monkeypatch.setitem(sys.modules, "torch", value)
    return value


@pytest.fixture(params=["pickle", "cloudpickle"])
def serializer(request):
    return pickle if request.param == "pickle" else pytest.importorskip("cloudpickle")


def test_published_namespace_roundtrip_preserves_identity(frontend, serializer):
    payload = serializer.dumps(frontend)
    assert serializer.loads(payload) is frontend
    assert len(payload) < 512
    assert b"unserializable_lock" not in payload


def test_cloudpickle_closure_reuses_published_namespace(frontend):
    cloudpickle = pytest.importorskip("cloudpickle")

    def capture():
        return frontend

    assert cloudpickle.loads(cloudpickle.dumps(capture))() is frontend


def test_replaced_namespace_is_rejected(frontend, serializer, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    with pytest.raises(TypeError, match="unpublished TorchNamespace"):
        serializer.dumps(frontend)


def test_missing_publication_is_rejected(frontend, serializer, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch")
    with pytest.raises(TypeError, match="unpublished TorchNamespace"):
        serializer.dumps(frontend)


def test_renamed_namespace_is_rejected(frontend, serializer):
    frontend.__name__ = "other_frontend"
    with pytest.raises(TypeError, match="unpublished TorchNamespace"):
        serializer.dumps(frontend)
