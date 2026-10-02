"""Native HCCL WORLD status: CPU refusal and opt-in real NPU execution.

Run in native Jittor mode. Default JITTOR_HCCL_STATUS_TEST_MODE=cpu requires
an unloaded HCCL runtime; fake launcher variables must not imply readiness.
For hardware coverage set JITTOR_HCCL_STATUS_TEST_MODE=npu and launch using
jittor.distributed.launch --backend hccl. That mode requires a real initialized
communicator and collective; missing hardware or bootstrap fails, never skips.
Mocked modules below test error handling only and are not hardware evidence.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

import jittor as jt
from jittor import distributed
from jittor.distributed import backend_status
from jittor._runtime.backend_libraries import get_library
from jittor._runtime.fallback import forbid_backend_fallbacks


NOT_INITIALIZED = {"initialized": False, "rank": None, "world_size": None}


def test_hccl_public_status_identity():
    assert distributed.get_hccl_world_info is backend_status.get_hccl_world_info
    assert "get_hccl_world_info" in distributed.__all__


def test_hccl_status_never_requests_library_load(monkeypatch):
    calls = []

    def lookup(name, *, load):
        calls.append((name, load))
        return None

    monkeypatch.setattr(backend_status, "get_library", lookup)
    assert distributed.get_hccl_world_info() == NOT_INITIALIZED
    assert calls == [("hccl", False)]


def test_hccl_loaded_module_is_not_initialized_world(monkeypatch):
    # No rank/size functions: querying either before initialization must fail.
    module = SimpleNamespace(hccl_is_initialized=lambda: False, ops=object())
    monkeypatch.setattr(backend_status, "get_library", lambda *a, **k: module)
    assert distributed.get_hccl_world_info() == NOT_INITIALIZED


@pytest.mark.parametrize("failing_query", [
    "hccl_is_initialized", "hccl_process_group_rank", "hccl_process_group_size"])
def test_hccl_query_errors_propagate(monkeypatch, failing_query):
    failure = RuntimeError("native HCCL query failed")

    def fail(*args):
        raise failure

    module = SimpleNamespace(hccl_is_initialized=lambda: True,
                             hccl_process_group_rank=lambda group: 0,
                             hccl_process_group_size=lambda group: 1)
    setattr(module, failing_query, fail)
    monkeypatch.setattr(backend_status, "get_library", lambda *a, **k: module)
    with pytest.raises(RuntimeError) as caught:
        distributed.get_hccl_world_info()
    assert caught.value is failure


@pytest.mark.parametrize("rank,size", [(-1, 1), (1, 1), (0, 0)])
def test_hccl_invalid_world_metadata_rejected(monkeypatch, rank, size):
    module = SimpleNamespace(hccl_is_initialized=lambda: True,
                             hccl_process_group_rank=lambda group: rank,
                             hccl_process_group_size=lambda group: size)
    monkeypatch.setattr(backend_status, "get_library", lambda *a, **k: module)
    with pytest.raises(RuntimeError, match="invalid rank/size"):
        distributed.get_hccl_world_info()


def test_hccl_status_on_declared_runtime(monkeypatch):
    mode = os.environ.get("JITTOR_HCCL_STATUS_TEST_MODE", "cpu")
    assert mode in ("cpu", "npu"), "select cpu or npu explicitly"
    if mode == "cpu":
        assert jt.flags.use_cuda == 0, "CPU gate requires CPU mode"
        assert get_library("hccl", load=False) is None
        assert distributed.get_hccl_world_info() == NOT_INITIALIZED
        # Set after Jittor import: a configuration claim cannot create a comm.
        monkeypatch.setenv("JT_HCCL_WORLD_SIZE", "8")
        monkeypatch.setenv("JT_HCCL_RANK", "3")
        assert distributed.get_hccl_world_info() == NOT_INITIALIZED
        return

    assert jt.compiler.has_acl, "NPU gate requires the real ACL runtime"
    # Backend selection loads ACL; native Jittor still requires explicit device mode.
    with jt.flag_scope(use_cuda=1):
        info = distributed.get_hccl_world_info()
        assert info["initialized"] is True, "HCCL WORLD must be initialized"
        rank, size = info["rank"], info["world_size"]
        assert rank == int(os.environ["JT_HCCL_RANK"])
        assert size == int(os.environ["JT_HCCL_WORLD_SIZE"])
        # The independent expected sum uses every rank's known input, rather than
        # assuming an identity reduction (which would miss a broken multi-rank op).
        source = np.array([rank + 1, 2 * (rank + 1)], dtype=np.float32)
        expected = np.array([size * (size + 1) / 2, size * (size + 1)],
                            dtype=np.float32)
        module = get_library("hccl", load=False)
        with forbid_backend_fallbacks():
            result = module.ops.hccl_all_reduce(jt.array(source), "sum", 0)
            result.sync()
            np.testing.assert_array_equal(result.numpy(), expected)
            jt.sync_all(True)
        assert distributed.get_hccl_world_info() == info
