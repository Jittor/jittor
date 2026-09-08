"""Native runtime round trips for the checkpoint implementation move."""

import importlib
import pickle

import jittor as jt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def restore_execution_mode():
    with jt.flag_scope(use_cuda=jt.flags.use_cuda):
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
