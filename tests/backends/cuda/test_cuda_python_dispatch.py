"""Real-CUDA execution and multi-device Python routing probes."""

from _helpers import capability as _test_capability

import numpy as np
import pytest

from _helpers.python_dispatch import DISPATCH_PROBES, assert_result, registered_kernels


def _require_cuda(jt, count=1):
    if not _test_capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip("CUDA Python dispatch requires a CUDA build")
    if _test_capability.device_count('cuda', backend=jt) < count:
        pytest.skip("CUDA Python dispatch requires %d visible CUDA devices" % count)


@pytest.mark.parametrize("probe", DISPATCH_PROBES, ids=lambda probe: probe.__name__)
def test_cuda_python_dispatch(probe):
    import jittor as jt

    _require_cuda(jt)
    with jt.flag_scope(use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0):
        probe(jt, "cuda", 0)


def test_input_device_wins_over_current_device():
    import jittor as jt
    from jittor._runtime.dispatch import dispatch_context, select_kernel

    _require_cuda(jt, 2)
    values = np.arange(8, dtype="float32")
    op = "test.python_dispatch.nondefault_device"

    def kernel(value):
        return value + 2

    with jt.flag_scope(use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0):
        with jt.flag_scope(device_id=1):
            x = jt.array(values)
        with registered_kernels(op, "cuda") as register:
            register(kernel)
            for materialized in (False, True):
                if materialized:
                    x.sync()
                before = (x.location(), x.device_id, jt.introspection.policy.runtime.device_id)
                context = dispatch_context(x)
                assert (context.backend, context.device_id) == ("cuda", 1)
                assert select_kernel(op, x) is kernel
                assert (x.location(), x.device_id, jt.introspection.policy.runtime.device_id) == before
                assert_result(jt, kernel(x), values + 2, "cuda", 1)


def test_mixed_devices_are_rejected_before_kernel_or_graph_execution():
    import jittor as jt
    from jittor._runtime.dispatch import dispatch_context, select_kernel

    _require_cuda(jt, 2)
    values = np.arange(8, dtype="float32")
    op = "test.python_dispatch.mixed_devices"
    calls = []

    def kernel(left, right):
        calls.append((left, right))
        return left + right

    with jt.flag_scope(use_cuda=1, device_id=0, lazy_execution=1, auto_flush_ops=0):
        x = jt.array(values)
        with jt.flag_scope(device_id=1):
            y = jt.array(values + 1)
        with registered_kernels(op, "cuda") as register:
            register(kernel)
            for materialized in (False, True):
                if materialized:
                    x.sync()
                    y.sync()
                before = (x.location(), y.location(), jt.introspection.counters.exec_calls)
                with pytest.raises(RuntimeError, match="same device"):
                    dispatch_context(x, y)
                with pytest.raises(RuntimeError, match="same device"):
                    select_kernel(op, x, y)
                assert (x.location(), y.location(), jt.introspection.counters.exec_calls) == before
                assert calls == []
            assert_result(jt, x + 1, values + 1, "cuda", 0)
            assert_result(jt, y + 1, values + 2, "cuda", 1)
