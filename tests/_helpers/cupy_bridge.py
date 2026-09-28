"""The CuPy bridge every CUDA numpy-code operator goes through.

``py_converter.h`` hands the callback ``cupy`` instead of ``numpy`` whenever
``use_cuda`` is on, and it does so by importing the module outright. Without
CuPy the operator raises a bare ``ModuleNotFoundError`` from inside execution,
at whatever ``.numpy()`` happened to force it -- and the failed execution
leaves the CUDA operators pending, so the *next* device transition flushes them
while the flag says CPU and an unrelated test dies with "No kernel registered
for cublas_matmul on cpu". One missing optional dependency, several tests, none
of the messages naming it.

So: ask before running the CUDA half, and say which dependency is missing.
"""
import importlib.util
import sys
import unittest


def cuda_numpy_code_available():
    """Whether a numpy-code operator can execute on CUDA on this machine.

    Asked with ``find_spec`` rather than through ``_helpers.torch_runtime``:
    that module reaches for ``torch`` to decide what owns the name, and a
    compatibility-mode test that imports it before ``import jittor`` makes the
    shim refuse to install ("cannot install Jittor Torch compatibility over an
    existing Torch module graph"). This question has nothing to do with Torch.
    """
    if "cupy" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("cupy") is not None
    except (ImportError, ValueError):
        return False


def requires_cuda_numpy_code():
    """Skip unless the CUDA numpy-code bridge can actually run."""
    if not cuda_numpy_code_available():
        raise unittest.SkipTest(
            "CUDA numpy-code operators need CuPy: py_converter hands the "
            "callback `cupy` when use_cuda is on, and it is not installed")
