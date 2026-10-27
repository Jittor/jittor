"""Exercise the NumPy C-API with both the 1.x and 2.x runtime layouts.

Run this file once per NumPy version in separate environments. These cases
used to crash at CopyInto or silently copy zero bytes under NumPy 2.
"""

import inspect
import textwrap

import jittor as jt

from _helpers.child_process import run_python_child


def test_numpy_array_and_ring_buffer_copy_preserve_values_and_dtypes():
    # A bad C-API slot can kill the interpreter. Keep that failure observable
    # to pytest, including Jittor's SIGCHLD handler, with the crash boundary.
    process = run_python_child(
        ["-c", textwrap.dedent("""
            import numpy as np
            import jittor as jt

            assert jt.flags.use_cuda == 0
            dtypes = (
                'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32',
                'uint32', 'int64', 'uint64', 'float16', 'float32',
                'float64', 'complex64', 'complex128',
            )
            for dtype in dtypes:
                source = np.arange(24).reshape(2, 3, 4).astype(dtype)
                if dtype.startswith('complex'):
                    source += 1j * source
                layouts = (
                    source, source[:, ::-1, ::2], np.asfortranarray(source),
                )
                for array in layouts:
                    # Explicit dtype bypasses Jittor's documented default
                    # auto_convert_64_to_32, so these check the ABI copy.
                    result = jt.array(array, dtype=str(array.dtype)).numpy()
                    assert result.dtype == array.dtype
                    np.testing.assert_array_equal(result, array)
                    ring = jt.RingBuffer(1 << 16)
                    ring.push(array)
                    restored = ring.pop()
                    assert restored.dtype == array.dtype
                    np.testing.assert_array_equal(restored, array)

            for source in (np.array(3.5), np.empty((2, 0, 3), np.float32)):
                result = jt.array(source, dtype=str(source.dtype)).numpy()
                assert result.shape == source.shape
                np.testing.assert_array_equal(result, source)
            print('NUMPY_ABI_PASSED', np.__version__)
        """)],
        env={"JT_BACKEND": "cpu", "JT_USE_CUDA": "0", "use_cuda": "0"},
        merge_stderr=True,
        crash_isolated=True,
    )
    assert process.returncode == 0, process.stdout
    assert "NUMPY_ABI_PASSED" in process.stdout


def test_native_shape_is_a_property_with_variable_rank():
    # The binding returns NanoVector, an integer sequence. Tuple[int, ...]
    # in the generated stub must not be mistaken for a runtime tuple promise.
    descriptor = inspect.getattr_static(jt.Var, "shape")
    assert hasattr(descriptor, "__get__")
    for dimensions in ((), (5,), (2, 3, 4), (1,) * 10):
        var = jt.zeros(dimensions)
        assert not callable(var.shape)
        assert tuple(var.shape) == dimensions
        assert all(type(size) is int for size in var.shape)
