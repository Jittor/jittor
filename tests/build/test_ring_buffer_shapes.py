"""RingBuffer serializes shape values without C++ object lifetime or pointers."""

import os
import textwrap

import pytest

from _helpers.child_process import run_python_child


def _check_child(body):
    child = run_python_child(
        ['-c', textwrap.dedent(body)], text=False, merge_stderr=True,
        crash_isolated=True,
    )
    assert child.returncode == 0, child.stdout.decode('utf8', 'replace')


def test_shape_write_survives_reused_nonzero_payload():
    _check_child('''
        import jittor as jt
        import numpy as np
        ring = jt.RingBuffer(2048)
        ring.keep_numpy_array(True)
        payload = np.full(1800, 255, dtype='uint8')
        ring.push(payload)
        np.testing.assert_array_equal(ring.pop(), payload)
        for _ in range(30):
            ring.push(-1)
            assert ring.pop() == -1
        value = np.arange(6, dtype='float32').reshape(2, 3)
        ring.push(value)
        np.testing.assert_array_equal(ring.pop(), value)
        print('REUSED_SHAPE_PASS', flush=True)
    ''')


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork is unavailable")
@pytest.mark.parametrize("rank", [11, 13, 32])
def test_spilled_shape_crosses_process_boundary_without_heap_pointers(rank):
    _check_child('''
        import os
        rank = RANK
        import jittor as jt
        import numpy as np
        if not hasattr(os, 'fork'):
            raise RuntimeError('this shared-memory test requires fork')
        ring = jt.RingBuffer(2048)
        ring.keep_numpy_array(True)
        pid = os.fork()
        if pid == 0:
            value = np.arange(6, dtype='float32').reshape((1,) * (rank - 2) + (2, 3))
            ring.push(value)
            os._exit(0)
        got = ring.pop_for(10000)
        _, status = os.waitpid(pid, 0)
        assert status == 0, status
        assert got.shape == (1,) * (rank - 2) + (2, 3), got.shape
        np.testing.assert_array_equal(got.reshape(-1), np.arange(6, dtype='float32'))
        print('SPILLED_SHAPE_PASS', flush=True)
    '''.replace('RANK', str(rank)))


def test_empty_shape_preserves_large_int64_extent():
    _check_child('''
        import jittor as jt
        import numpy as np
        ring = jt.RingBuffer(2048)
        ring.keep_numpy_array(True)
        value = np.empty((0, 1 << 62), dtype='uint8')
        ring.push(value)
        restored = ring.pop()
        assert restored.shape == value.shape
        assert restored.dtype == value.dtype and restored.nbytes == 0
    ''')
