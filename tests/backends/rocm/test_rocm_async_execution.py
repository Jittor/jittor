"""Native HIP worker execution with visible handovers and ordered host reads."""

import ctypes
import numpy as np
import pytest
import jittor as jt
from jittor._runtime import forbid_backend_fallbacks

pytestmark = pytest.mark.skipif(not jt.compiler.has_rocm, reason='ROCm build required')


def test_inference_handovers_match_inline_results_and_host_reads():
    rng = np.random.RandomState(97)
    raw = rng.normal(size=(32, 32)).astype('float32') / 8
    weight = rng.normal(size=(32, 32)).astype('float32') / 8
    before_fallback = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_rocm=1, use_parallel_op_compiler=0), forbid_backend_fallbacks():
        x, w = jt.array(raw), jt.array(weight)
        jt.sync_all(True)
        outputs = []
        for async_on in (0, 1):
            with jt.flag_scope(async_execution=async_on, auto_flush_ops=23,
                               auto_flush_bytes=0, async_min_caller_us=0, auto_graph_replay=0), jt.no_grad():
                before = jt.core.async_exec_batches()
                value, checkpoints, kept = x, [], []
                for i in range(40):
                    transformed = jt.tanh(value * 0.5 + 0.25)
                    # A live result with no consumer is a submission root;
                    # the single consumed chain alone has none at the
                    # holder-construction boundary. Read every kept result.
                    kept.append(jt.matmul(transformed, w))
                    value = jt.matmul(transformed, w)
                    if i in (9, 23):
                        checkpoints.append(value.numpy().copy())
                value.sync()
                assert value.location() == 'device'
                outputs.append((value.numpy().copy(), checkpoints + [v.numpy().copy() for v in kept]))
                ran = jt.core.async_exec_batches() - before
                assert ran > 0 if async_on else ran == 0
        for actual, expected in zip(outputs[1][1] + [outputs[1][0]],
                                    outputs[0][1] + [outputs[0][0]]):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        ref = raw.copy()
        for actual in outputs[1][1][-40:]:
            ref = np.tanh(ref * 0.5 + 0.25) @ weight
            np.testing.assert_allclose(actual, ref, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(outputs[1][0], ref, rtol=2e-5, atol=2e-5)
    assert jt.core.backend_fallback_count() == before_fallback


def test_new_host_thread_binds_the_runtime_device():
    # A second Python thread disables future handovers in its process. Keep
    # this device-binding check in a child so the inference tests retain the
    # same single-caller contract as user inference.
    from _helpers.child_process import run_child_script
    result = run_child_script(r'''
import ctypes
from concurrent.futures import ThreadPoolExecutor
import jittor as jt
sdk = ctypes.CDLL('libamdhip64.so')
sdk.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
sdk.hipGetDevice.restype = ctypes.c_int
for device in range(jt.core.backend_device_count('rocm')):
    with jt.runtime.scope(use_rocm=1, device_id=device):
        def check():
            assert jt.current_device() == device
            actual = ctypes.c_int(-1)
            assert sdk.hipGetDevice(ctypes.byref(actual)) == 0
            assert actual.value == device
        with ThreadPoolExecutor(1) as pool:
            pool.submit(check).result()
print('HIP_THREAD_DEVICE_OK')
''', timeout=180, text=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert 'HIP_THREAD_DEVICE_OK' in result.stdout
