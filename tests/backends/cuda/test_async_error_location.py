"""Actual delayed CUDA faults preserve source candidates without debug sync."""
import textwrap

import pytest

from _helpers.child_process import run_child_script
from _helpers.capability import require_accelerator


def _require_cuda():
    require_accelerator("cuda")


def test_launch_history_survives_graph_release():
    _require_cuda()
    result = run_child_script(textwrap.dedent('''
        import gc, inspect
        import jittor as jt
        with jt.runtime.scope(use_cuda=1, trace_py_var=0, auto_flush_ops=0):
            line = inspect.currentframe().f_lineno + 1
            value = jt.array([2.0]) + 3
            assert value.numpy().tolist() == [5.0]
            del value
            gc.collect()
            history = jt.core.async_launch_history("cuda", 0)
            assert __file__ + ":" + str(line) in history, history
            assert "candidates are not proof" in history
            assert "not-found" in jt.core.async_launch_history("cuda", 0, 987654)
            print("NORMAL-RELEASE-PASS", flush=True)
    '''), env={"JT_SYNC": "0", "trace_py_var": "0"}, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NORMAL-RELEASE-PASS" in result.stdout
    assert "CUDA error at" not in result.stderr


@pytest.mark.parametrize("boundary", ["device_wait", "readback"])
def test_asynchronous_fault_reports_creation_line(boundary):
    _require_cuda()
    source = textwrap.dedent('''
        import inspect
        import jittor as jt
        with jt.runtime.scope(use_cuda=1, trace_py_var=0, auto_flush_ops=0):
            x = jt.ones((1,), "float32")
            assert x.numpy().tolist() == [1.0]
            fault_line = inspect.currentframe().f_lineno + 1
            y = jt.code(x.shape, x.dtype, [x], cuda_src=r"""
                __global__ void delayed_illegal_write(float* p) {
                    unsigned long long start = clock64();
                    while (clock64() - start < 200000000ULL) {}
                    ((volatile float*)p)[1<<28] = 1.0f;
                }
                delayed_illegal_write<<<1,1>>>(out0_p);
            """)
            y.sync()  # issue only; poison deliberately occurs later
            print("ISSUED-WITHOUT-WAIT", flush=True)
            try:
                BOUNDARY
            except RuntimeError as error:
                report = str(error)
                assert "cudaErrorIllegalAddress" in report, report
                assert "[Recent launch candidates]" in report, report
                assert __file__ + ":" + str(fault_line) in report, report
                assert "op=code" in report, report
                assert "candidates are not proof" in report, report
                assert "device=0" in report, report
                print("ASYNC-ORIGIN-PASS", flush=True)
            else:
                raise AssertionError("asynchronous poison was never reported")
    ''').replace("BOUNDARY", "jt.sync_all(True)" if boundary == "device_wait" else "y.numpy()")
    result = run_child_script(source, env={"JT_SYNC": "0", "trace_py_var": "0"},
                              text=True, timeout=180, crash_isolated=True,
                              name="async_origin_" + boundary)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISSUED-WITHOUT-WAIT" in result.stdout
    assert "ASYNC-ORIGIN-PASS" in result.stdout
