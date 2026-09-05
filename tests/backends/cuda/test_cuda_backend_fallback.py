"""Real CPU-only kernel fallback, distinct from explicit host/device copies."""

import gc
import json
import re

import numpy as np
import pytest


FALLBACK_MESSAGE = r"Backend fallback: op=code backend=cuda target=cpu reason=\S[^\n]*"


@pytest.fixture
def cuda_runtime():
    import jittor as jt

    if not jt.has_cuda or jt.get_device_count() < 1:
        pytest.skip("backend fallback policy requires a real CUDA device")
    return jt


def _cpu_only_add_five(jt, value, marker):
    # The marker distinguishes rejecting a fallback from reporting it after
    # the CPU implementation has already run and touched input storage.
    marker_literal = json.dumps(str(marker))
    return jt.code(
        value.shape,
        value.dtype,
        [value],
        cpu_header="#include <cstdio>\n#include <stdexcept>",
        cpu_src="""
            FILE* marker = std::fopen(%s, "ab");
            if (!marker) throw std::runtime_error("cannot open fallback marker");
            std::fputs("executed\\n", marker);
            std::fclose(marker);
            for (int i = 0; i < in0_shape0; ++i)
                out0_p[i] = in0_p[i] + 5.0f;
        """ % marker_literal,
    )


def _assert_cuda_value(jt, value, expected):
    value.sync()
    jt.sync_all(True)
    assert value.location() == "device"
    assert value.device_id == 0
    np.testing.assert_array_equal(value.numpy(), expected)


@pytest.mark.parametrize("policy", ("error", "warn", "allow"))
def test_cpu_only_code_obeys_cuda_fallback_policy(cuda_runtime, policy, tmp_path, capfd):
    jt = cuda_runtime
    marker = tmp_path / "cpu-kernel-executed.txt"
    values = np.arange(8, dtype="float32")
    with jt.flag_scope(
        use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0,
        use_cuda_managed_allocator=0, backend_fallback=policy,
        log_silent=0, log_sync=1,
    ):
        x = jt.array(values) + 1
        x.sync()
        jt.sync_all(True)
        assert x.location() == "device"
        original_pointer = x.device_raw_ptr
        before = jt.core.backend_fallback_count()
        capfd.readouterr()
        output = None
        try:
            if policy == "error":
                with pytest.raises(RuntimeError, match=FALLBACK_MESSAGE) as caught:
                    output = _cpu_only_add_five(jt, x, marker)
                    output.sync()
                    jt.sync_all(True)
                assert "reason=" in str(caught.value)
                assert not marker.exists(), "rejected CPU kernel was executed"
                assert x.location() == "device", "input migrated before fallback rejection"
                assert x.device_raw_ptr == original_pointer
                del caught
            else:
                output = _cpu_only_add_five(jt, x, marker)
                _assert_cuda_value(jt, output, values + 6)
                assert marker.read_text() == "executed\n"
            assert jt.core.backend_fallback_count() == before + 1
            captured = capfd.readouterr()
            log = captured.out + captured.err
            if policy == "warn":
                assert len(re.findall(FALLBACK_MESSAGE, log)) == 1
            elif policy == "allow":
                assert "Backend fallback:" not in log
        finally:
            output = None
            gc.collect()
            jt.gc()

        # The rejected graph must not remain pending after its last reference
        # is released. Recovery itself runs with fallback forbidden.
        after = jt.core.backend_fallback_count()
        with jt.flag_scope(backend_fallback="error"):
            _assert_cuda_value(jt, jt.array(values) + 2, values + 2)
        assert jt.core.backend_fallback_count() == after
        if policy == "error":
            assert not marker.exists(), "cleanup executed the rejected CPU kernel"


def test_host_staging_and_explicit_transfers_are_not_backend_fallbacks(cuda_runtime):
    jt = cuda_runtime
    values = np.arange(8, dtype="float32")
    with jt.flag_scope(
        use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0,
        use_cuda_managed_allocator=0, use_cuda_host_allocator=1,
        backend_fallback="error",
    ):
        before = jt.core.backend_fallback_count()
        staged = jt.array(values)
        staged.sync()
        gpu = staged + 1
        gpu.sync()
        jt.sync_all(True)
        assert gpu.location() == "device"

        host = gpu.cpu()
        host.sync()
        assert host.location() == "cpu"
        np.testing.assert_array_equal(host.numpy(), values + 1)
        back = host.cuda(0)
        back.sync()
        assert back.location() == "device"
        assert back.device_id == 0
        assert host.location() == "cpu"

        fetched = []
        jt.fetch(back, lambda result: fetched.append(result.copy()))
        jt.sync_all(True)
        assert len(fetched) == 1
        np.testing.assert_array_equal(fetched[0], values + 1)
        _assert_cuda_value(jt, back + 2, values + 3)
        assert jt.core.backend_fallback_count() == before
