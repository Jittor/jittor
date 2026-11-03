"""Array shape metadata survives queue reuse and process boundaries."""

import os

import pytest

from _helpers.child_process import run_child_script


_CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "JT_BUILD_NVCC_PATH": "",
            "nvcc_path": "", "JT_USE_CUDA": "0", "use_cuda": "0"}
_PREAMBLE = """
import numpy as np
import jittor as jt
from _helpers.native_runtime import assert_native_runtime
assert_native_runtime(jt)
"""


def test_compact_and_high_rank_array_shapes_survive_queue_reuse():
    result = run_child_script(_PREAMBLE + """
queue = jt.RingBuffer(65536)
for index in range(24):
    rank = (2, 11, 3)[index % 3]
    expected = np.arange(2**rank, dtype=np.float32).reshape((2,) * rank)
    value = expected if index % 2 == 0 else jt.array(expected)
    queue.push(value)
    actual = queue.pop()
    if isinstance(actual, jt.Var):
        actual = actual.numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
assert_native_runtime(jt)
print("ARRAY_SHAPE_ROUNDTRIPS_OK", flush=True)
""", env=_CPU_ENV, without_torch_mode=True, crash_isolated=True,
        name="ring_shape_roundtrips", text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ARRAY_SHAPE_ROUNDTRIPS_OK" in result.stdout


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_high_rank_array_shape_survives_shared_queue_transfer():
    result = run_child_script(_PREAMBLE + """
import os
import select
import signal

queue = jt.RingBuffer(65536)
expected = np.arange(2048, dtype=np.float32).reshape((2,) * 11)
read_fd, write_fd = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(read_fd)
    jt.jt_init_subprocess()
    actual = queue.pop()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape
    os.write(write_fd, b"OK")
    os._exit(0)
os.close(write_fd)
reaped = False
try:
    queue.push(expected)
    ready, _, _ = select.select([read_fd], [], [], 30)
    if not ready:
        raise AssertionError("shared array receiver exceeded bounded wait")
    message = os.read(read_fd, 2)
    _, status = os.waitpid(pid, 0)
    reaped = True
    assert message == b"OK"
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
finally:
    os.close(read_fd)
    if not reaped:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
assert_native_runtime(jt)
print("SHARED_ARRAY_SHAPE_OK", flush=True)
""", env=_CPU_ENV, without_torch_mode=True, crash_isolated=True,
        name="ring_shape_shared_transfer", text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SHARED_ARRAY_SHAPE_OK" in result.stdout
