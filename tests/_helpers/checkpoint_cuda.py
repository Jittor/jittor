"""Shared real-CUDA acceptance support for activation storage and replay tests."""
import os
import pytest


@pytest.fixture(autouse=True)
def accelerator():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0] != "cuda":
        pytest.skip("backend: CUDA activation checkpoint contracts")
    if not torch.cuda.is_available():
        pytest.fail("checkpoint acceptance requires a real CUDA device")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1, auto_flush_ops=0, auto_graph_replay=0), forbid_backend_fallbacks():
        yield
        jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before


def snapshot(value):
    return value.detach().cpu().numpy().copy()


def curand_position():
    import jittor as jt
    backend = jt.compile_extern.get_library("curand", load=True)
    return (int(backend.curand_generator_seed(0)),
            int(backend.curand_generator_offset(0)))
